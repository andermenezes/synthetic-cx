"""Deterministic real-browser regressions for native website action authority."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

import pytest
from playwright.async_api import Route, async_playwright

from syntheticcx.adapters.channel.broker import NativeChatBroker
from syntheticcx.adapters.channel.website import WebsiteNavigator
from syntheticcx.config.models import ExecutionLimits, WebsiteNavigationStep
from syntheticcx.errors import ChannelError

URL = "https://service.example/"
PHONE = "15551234567"
ACCOUNT_CANARY = "ACCOUNT DOM MUST NEVER ENTER WEBSITE EVIDENCE"


@asynccontextmanager
async def website(pages: dict[str, str]):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        context = await browser.new_context(viewport={"width": 1000, "height": 700})
        visits: list[str] = []

        async def serve(route: Route) -> None:
            parsed = urlsplit(route.request.url)
            visits.append(route.request.url)
            if parsed.hostname == "web.whatsapp.com":
                body = f"<body>{ACCOUNT_CANARY}</body>"
            elif parsed.hostname == "api.whatsapp.com":
                body = pages["landing"]
            elif parsed.hostname == "service.example" and parsed.path in pages:
                body = pages[parsed.path]
            else:
                pytest.fail(f"unexpected navigation: {route.request.url}")
            await route.fulfill(content_type="text/html", body=body)

        await context.route("**/*", serve)
        page = await context.new_page()
        await page.goto(URL)
        try:
            yield browser, context, page, visits, serve
        finally:
            await context.close()
            await browser.close()


def limits(**overrides: Any) -> ExecutionLimits:
    return ExecutionLimits(
        max_turns=3,
        journey_timeout_seconds=60,
        greeting_grace_seconds=0,
        browser_action_timeout_seconds=3,
        response_timeout_seconds=5,
        model_timeout_seconds=5,
        evaluator_timeout_seconds=5,
        persistence_timeout_seconds=5,
        close_timeout_seconds=5,
        **overrides,
    )


async def choose(broker: NativeChatBroker, label: str, action: str = "open") -> dict[str, Any]:
    observation = await broker.observe_website()
    candidate = next(
        item for item in observation["catalog"]["candidates"] if item["description"] == label
    )
    assert action in candidate["actions"]
    assert action in observation["available_actions"]
    await broker.act_website(observation["revision"], action, candidate["id"])
    return observation


@pytest.mark.parametrize("replay", [False, True])
def test_discovery_native_guard_refuses_portability_before_click_or_replay(replay):
    async def exercise():
        async with website(
            {"/": '<button id="port" onclick="window.effects=1">Portabilidade</button>'}
        ) as resources:
            _, _, page, _, _ = resources
            broker = NativeChatBroker(URL, limits(), page=page)
            broker.discovery_only = True
            await broker.start()
            try:
                with pytest.raises(ChannelError, match="forbidden in discovery"):
                    if replay:
                        await broker.website_navigator.replay(
                            [
                                WebsiteNavigationStep.model_validate(
                                    {
                                        "action": "click",
                                        "locator": {"strategy": "css", "value": "#port"},
                                    }
                                )
                            ]
                        )
                    else:
                        await choose(broker, "Portabilidade")
                assert await page.evaluate("window.effects || 0") == 0
            finally:
                await broker.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("target", ["webchat", "whatsapp"])
def test_mixed_route_backtracking_lazy_scroll_popup_and_exact_fresh_replay(target: str) -> None:
    async def exercise() -> None:
        final = (
            '<a id="handoff" target="_blank" '
            f'href="https://web.whatsapp.com/send?phone={PHONE}">Talk to team</a>'
            if target == "whatsapp"
            else '<button id="handoff" '
            "onclick=\"document.querySelector('#chat').hidden=false\">Talk to team</button>"
        )
        pages = {
            "/": """<body><p>Choose your service</p>
                <button id="products"
                 onmouseenter="document.querySelector('#routes').hidden=false">Products</button>
                <nav id="routes" hidden><a id="residential" href="/dead">Residential support</a>
                <a id="mobile" href="/mobile">Mobile support</a></nav></body>""",
            "/dead": (
                "<body><h1>Residential maintenance</h1>"
                "<p>No online support for this service.</p></body>"
            ),
            "/mobile": (
                "<body><h1>Mobile customers</h1>"
                '<a id="channels" href="/channels">Choose a channel</a></body>'
            ),
            "/channels": (
                "<body><h1>Service channels</h1>"
                '<a id="details" href="/details">More service options</a></body>'
            ),
            "/details": (
                '<body><h1>Contact details</h1><a id="continue" href="/lazy">Continue</a></body>'
            ),
            "/lazy": f"""<body style="min-height:1800px"><h1>Online assistance</h1>
                <div id="mount" style="position:absolute;top:900px"></div>
                <section id="chat" role="dialog" aria-label="Chat assistant" hidden>
                <textarea aria-label="Message"></textarea></section>
                <script>let scheduled=false;addEventListener('scroll',()=>{{
                  if(scrollY>300&&!scheduled){{scheduled=true;setTimeout(()=>{{
                    document.querySelector('#mount').innerHTML={json.dumps(final)};
                  }},180)}}
                }})</script></body>""",
        }
        async with website(pages) as (browser, context, page, visits, serve):
            broker = NativeChatBroker(
                URL,
                page=page,
                limits=limits(),
                website_recipient=PHONE if target == "whatsapp" else None,
            )
            await broker.start()
            try:
                await choose(broker, "Products", "hover")
                await choose(broker, "Residential support")
                dead = await broker.observe_website()
                assert "back" in dead["available_actions"]
                assert "Residential maintenance" in dead["page_context"]
                await broker.act_website(dead["revision"], "back")
                await choose(broker, "Products", "hover")
                reopened = await broker.observe_website()
                branch = next(
                    item
                    for item in reopened["catalog"]["candidates"]
                    if item["description"] == "Residential support"
                )
                assert "open" in branch["tried_actions"]
                assert "open" not in branch["actions"]
                assert any(item["action"] == "back" for item in reopened["history"])
                await choose(broker, "Mobile support")
                await choose(broker, "Choose a channel")
                await choose(broker, "More service options")
                await choose(broker, "Continue")
                lazy = await broker.observe_website()
                assert "scroll_down" in lazy["available_actions"]
                assert not any(
                    item["description"] == "Talk to team" for item in lazy["catalog"]["candidates"]
                )
                await broker.act_website(lazy["revision"], "scroll_down")
                waiting = await broker.observe_website()
                await broker.act_website(waiting["revision"], "wait")
                ready = await broker.observe_website()
                assert "scroll_up" in ready["available_actions"]
                await broker.act_website(ready["revision"], "scroll_up")
                above = await broker.observe_website()
                offscreen = next(
                    item
                    for item in above["catalog"]["candidates"]
                    if item["description"] == "Talk to team"
                )
                assert offscreen["actions"] == []
                await broker.act_website(above["revision"], "scroll_down")
                await choose(broker, "Talk to team")
                assert (await broker.observe_website())["terminal"] == target
                steps = [
                    WebsiteNavigationStep.model_validate(step.model_dump(mode="json"))
                    for step in broker.website_navigator.steps
                ]
                assert len(steps) > 5
                assert {step.action for step in steps} >= {
                    "hover",
                    "click",
                    "back",
                    "scroll_down",
                    "scroll_up",
                    "wait",
                }
                assert all(
                    step.locator is not None for step in steps if step.action in {"click", "hover"}
                )
                assert not any(
                    "ACCOUNT DOM" in json.dumps(item) for item in broker.website_navigator._history
                )
                if target == "whatsapp":
                    assert broker.page is not page
                    assert await page.title() == ""
                replay_context = await browser.new_context(viewport={"width": 1000, "height": 700})
                await replay_context.route("**/*", serve)
                replay_page = await replay_context.new_page()
                await replay_page.goto(URL)
                replay_broker = NativeChatBroker(
                    URL,
                    page=replay_page,
                    limits=limits(),
                    website_recipient=PHONE if target == "whatsapp" else None,
                )
                await replay_broker.start()
                try:
                    await replay_broker.website_navigator.replay(steps)
                    replay_broker.page = replay_broker.website_navigator.page
                    await replay_broker._remember_website_opener()
                    assert (await replay_broker.observe_website())["terminal"] == target
                    assert replay_broker.website_navigator.steps == steps
                finally:
                    await replay_broker.close()
                    await replay_context.close()
                assert sum(urlsplit(value).path == "/dead" for value in visits) == 2
            finally:
                await broker.close()

    asyncio.run(exercise())


def test_duplicate_id_shared_title_landing_uses_visible_exact_semantic_choice() -> None:
    async def exercise() -> None:
        destination = f"https://web.whatsapp.com/send?phone={PHONE}"
        pages = {
            "/": (
                '<body><a id="entry" '
                f'href="https://api.whatsapp.com/send?phone={PHONE}">Contact us</a></body>'
            ),
            "landing": f"""<body><h1>Continue your conversation</h1>
                <a id="action-button" title="Share on WhatsApp" href="{destination}">Open app</a>
                <a id="action-button" role="button" title="Share on WhatsApp"
                 href="{destination}">Continue to WhatsApp Web</a>
                <a id="action-button" title="Share on WhatsApp" hidden
                 href="{destination}">Download app</a>
                <a id="action-button" role="button" title="Share on WhatsApp" hidden
                 href="{destination}">Continue to WhatsApp Web</a></body>""",
        }
        async with website(pages) as (browser, context, page, visits, serve):
            broker = NativeChatBroker(URL, page=page, limits=limits(), website_recipient=PHONE)
            await broker.start()
            try:
                await choose(broker, "Contact us")
                observed = await broker.observe_website()
                await choose(broker, "Continue to WhatsApp Web", "follow_link")
                assert (await broker.observe_website())["terminal"] == "whatsapp"
                assert ACCOUNT_CANARY not in json.dumps(observed)
                fresh = await context.new_page()
                await fresh.goto(URL)
                navigator = WebsiteNavigator(fresh, URL, limits(), recipient=PHONE)
                try:
                    await navigator.replay(broker.website_navigator.steps)
                    assert await navigator.terminal()
                finally:
                    await navigator.close()
            finally:
                await broker.close()

    asyncio.run(exercise())


def test_hidden_duplicates_use_persisted_visible_css_for_badge_icon() -> None:
    async def exercise() -> None:
        pages = {
            "/": """<body>
            <button id="icon" title="Messaging support"
             onclick="document.querySelector('#result').hidden=false"><span>7</span></button>
            <button id="icon" title="Messaging support" hidden><span>7</span></button>
            <p id="result" hidden>Channel options ready</p></body>"""
        }
        async with website(pages) as (browser, context, page, visits, serve):
            navigator = WebsiteNavigator(page, URL, limits())
            try:
                observed = await navigator.observe()
                candidate = next(
                    item
                    for item in observed["catalog"]["candidates"]
                    if item["description"] == "Messaging support"
                )
                await navigator.act(observed["revision"], "open", candidate["id"])
                assert await page.locator("#result").is_visible()
                assert navigator.steps[-1].locator.value == "#icon:visible"
                fresh = await context.new_page()
                await fresh.goto(URL)
                replay = WebsiteNavigator(fresh, URL, limits())
                try:
                    await replay.replay(navigator.steps)
                    assert await fresh.locator("#result").is_visible()
                finally:
                    await replay.close()
            finally:
                await navigator.close()

    asyncio.run(exercise())


def test_visible_duplicate_controls_and_original_unsafe_text_fail_closed() -> None:
    async def exercise() -> None:
        pages = {
            "/": """<body>
            <button id="duplicate" title="Help">Explore</button>
            <button id="duplicate" title="Help">Explore</button>
            <button id="unsafe" title="Messaging support">Confirm purchase</button>
            <button id="form" form="payment" title="Help">Help</button>
            <button id="send" title="Messaging support"
             onclick="window.sent=true">Send</button>
            <button id="send-icon" title="Messaging support"
             onclick="window.sent=true"><img alt="Send message"></button>
            <button id="safe"
             onclick="document.querySelector('#status').hidden=false">Browse</button>
            <p id="status" hidden>Browse options</p></body>"""
        }
        async with website(pages) as (browser, context, page, visits, serve):
            navigator = WebsiteNavigator(page, URL, limits())
            try:
                observed = await navigator.observe()
                assert [item["description"] for item in observed["catalog"]["candidates"]] == [
                    "Browse"
                ]
                candidate = observed["catalog"]["candidates"][0]
                await page.locator("#safe").evaluate(
                    "node => node.textContent = 'Confirm purchase'"
                )
                with pytest.raises(ChannelError, match="unsafe"):
                    await navigator.act(observed["revision"], "open", candidate["id"])
                assert navigator.steps == []
                assert not await page.locator("#status").is_visible()
                duplicate = WebsiteNavigationStep(
                    action="click", locator={"strategy": "css", "value": "#duplicate"}
                )
                with pytest.raises(ChannelError, match="ambiguous"):
                    await navigator.replay([duplicate])
            finally:
                await navigator.close()

    asyncio.run(exercise())


def test_modal_context_and_blocker_dismissal_without_force_clicks() -> None:
    async def exercise() -> None:
        hidden = (
            "<div hidden>"
            + "".join(
                f'<button id="hidden-{index}">Other service {index}</button>'
                for index in range(200)
            )
            + "</div>"
        )
        pages = {
            "/": """<body><p>Ordinary introductory content</p>
            <button id="target" style="position:absolute;top:80px;left:30px"
             onclick="document.querySelector('#result').hidden=false">Support options</button>"""
            + hidden
            + """<div role="dialog" aria-label="Choose your service" id="overlay"
             style="position:fixed;inset:0;background:white;z-index:9">
              <h2>Mobile or residential service?</h2>
              <p>Choose the mobile team for a mobile contract.</p>
              <button id="dismiss"
               onclick="document.querySelector('#overlay').remove()">Close</button>
              <textarea aria-label="Service question"></textarea>
            </div><p id="result" hidden>Support choices ready</p></body>"""
        }
        async with website(pages) as (browser, context, page, visits, serve):
            broker = NativeChatBroker(URL, page=page, limits=limits())
            await broker.start()
            try:
                observed = await broker.observe_website()
                assert observed["terminal"] is None
                assert observed["page_context"].startswith("Mobile or residential service?")
                target = next(
                    item
                    for item in observed["catalog"]["candidates"]
                    if item["description"] == "Support options"
                )
                assert target["actions"] == []
                assert "Mobile or residential" in target["blocked_by"]
                with pytest.raises(ChannelError, match="unavailable"):
                    await broker.act_website(observed["revision"], "open", target["id"])
                await choose(broker, "Close")
                await choose(broker, "Support options")
                assert await page.locator("#result").is_visible()
            finally:
                await broker.close()

    asyncio.run(exercise())


def test_no_change_exhaustion_state_changes_reenable_and_action_limit_is_native() -> None:
    async def exercise() -> None:
        pages = {
            "/": """<body><button id="browse">Browse</button>
            <button id="reveal"
             onclick="document.querySelector('#extra').hidden=false">Reveal</button>
            <p id="extra" hidden>A newly revealed choice</p></body>"""
        }
        async with website(pages) as (browser, context, page, visits, serve):
            navigator = WebsiteNavigator(page, URL, limits(website_action_limit=3))
            try:
                observed = await navigator.observe()
                browse = next(
                    item
                    for item in observed["catalog"]["candidates"]
                    if item["description"] == "Browse"
                )
                await navigator.act(observed["revision"], "open", browse["id"])
                unchanged = await navigator.observe()
                browse = next(
                    item
                    for item in unchanged["catalog"]["candidates"]
                    if item["description"] == "Browse"
                )
                assert "open" not in browse["actions"]
                assert unchanged["feedback"] == "no_visible_change_yet"
                reveal = next(
                    item
                    for item in unchanged["catalog"]["candidates"]
                    if item["description"] == "Reveal"
                )
                await navigator.act(unchanged["revision"], "open", reveal["id"])
                changed = await navigator.observe()
                browse = next(
                    item
                    for item in changed["catalog"]["candidates"]
                    if item["description"] == "Browse"
                )
                assert "open" in browse["actions"]
                await navigator.act(changed["revision"], "open", browse["id"])
                exhausted = await navigator.observe()
                assert exhausted["available_actions"] == ["observe", "stop"]
                with pytest.raises(ChannelError, match="unavailable"):
                    await navigator.act(exhausted["revision"], "wait")
                assert len(navigator.steps) == 3
            finally:
                await navigator.close()

    asyncio.run(exercise())


def test_wrong_recipient_and_changed_destination_are_never_activated() -> None:
    async def exercise() -> None:
        pages = {
            "/": f"""<body>
            <a id="wrong" href="https://web.whatsapp.com/send?phone=15550000000">Other team</a>
            <a id="right" href="https://web.whatsapp.com/send?phone={PHONE}">Our team</a>
            <a id="external" href="https://unrelated.example/">External options</a></body>"""
        }
        async with website(pages) as (browser, context, page, visits, serve):
            navigator = WebsiteNavigator(page, URL, limits(), recipient=PHONE)
            try:
                observed = await navigator.observe()
                assert [item["description"] for item in observed["catalog"]["candidates"]] == [
                    "Our team"
                ]
                item = observed["catalog"]["candidates"][0]
                await page.locator("#right").evaluate(
                    "node => node.href='https://web.whatsapp.com/send?phone=15550000000'"
                )
                with pytest.raises(ChannelError, match="changed"):
                    await navigator.act(observed["revision"], "open", item["id"])
                assert navigator.steps == []
                assert visits == [URL]
            finally:
                await navigator.close()

    asyncio.run(exercise())


def test_cancelled_dispatch_cannot_be_retried_and_observation_limit_is_native() -> None:
    async def exercise() -> None:
        pages = {
            "/": """<body><button id="browse"
            onclick="window.clicks=(window.clicks||0)+1">Browse</button></body>"""
        }
        async with website(pages) as (browser, context, page, visits, serve):
            navigator = WebsiteNavigator(page, URL, limits())
            try:
                observed = await navigator.observe()
                candidate = observed["catalog"]["candidates"][0]
                dispatch = asyncio.create_task(
                    navigator.act(observed["revision"], "open", candidate["id"])
                )
                await page.wait_for_function("window.clicks === 1")
                dispatch.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await dispatch
                uncertain = await navigator.observe()
                assert uncertain["available_actions"] == ["observe", "stop"]
                assert uncertain["history"][-1]["result"] == "uncertain"
                with pytest.raises(ChannelError, match="unavailable"):
                    await navigator.act(uncertain["revision"], "wait")
                assert await page.evaluate("window.clicks") == 1
            finally:
                await navigator.close()
            bounded = WebsiteNavigator(page, URL, limits(website_observation_limit=1))
            try:
                await bounded.observe()
                with pytest.raises(ChannelError, match="observation limit"):
                    await bounded.observe()
            finally:
                await bounded.close()

    asyncio.run(exercise())


def test_ancestor_context_distinguishes_identically_labelled_service_choices() -> None:
    async def exercise() -> None:
        pages = {
            "/": """<body><section><h2>Residential services</h2><ul><li>
                <a id="home" href="/home">Get support</a></li></ul></section>
                <section><h2>Mobile services</h2><ul><li>
                <a id="mobile" href="/mobile">Get support</a></li></ul></section></body>""",
            "/home": "<body>Residential service choices</body>",
            "/mobile": "<body>Mobile service choices</body>",
        }
        async with website(pages) as (browser, context, page, visits, serve):
            navigator = WebsiteNavigator(page, URL, limits())
            try:
                observed = await navigator.observe()
                options = observed["catalog"]["candidates"]
                assert len(options) == 2
                assert {item["description"] for item in options} == {"Get support"}
                mobile = next(item for item in options if "Mobile services" in item["context"])
                await navigator.act(observed["revision"], "open", mobile["id"])
                assert page.url == URL + "mobile"
                assert URL + "home" not in visits
            finally:
                await navigator.close()

    asyncio.run(exercise())


def test_service_popup_back_returns_only_to_observed_owned_tab_and_replays() -> None:
    async def exercise() -> None:
        pages = {
            "/": """<body><a id="branch" target="_blank" href="/dead">Service information</a>
                <a id="team" href="/team">Team options</a></body>""",
            "/dead": "<body>No available contact channel here.</body>",
            "/team": "<body>Our service team options</body>",
        }
        async with website(pages) as (browser, context, page, visits, serve):
            navigator = WebsiteNavigator(page, URL, limits())
            try:
                observed = await navigator.observe()
                branch = next(
                    item
                    for item in observed["catalog"]["candidates"]
                    if item["description"] == "Service information"
                )
                await navigator.act(observed["revision"], "open", branch["id"])
                popup = navigator.page
                assert popup is not page
                assert popup.url == URL + "dead"
                dead = await navigator.observe()
                await navigator.act(dead["revision"], "back")
                assert popup.is_closed()
                assert navigator.page is page
                returned = await navigator.observe()
                branch = next(
                    item
                    for item in returned["catalog"]["candidates"]
                    if item["description"] == "Service information"
                )
                assert "open" not in branch["actions"]
                team = next(
                    item
                    for item in returned["catalog"]["candidates"]
                    if item["description"] == "Team options"
                )
                await navigator.act(returned["revision"], "open", team["id"])
                assert page.url == URL + "team"
                fresh = await context.new_page()
                await fresh.goto(URL)
                replay = WebsiteNavigator(fresh, URL, limits())
                try:
                    await replay.replay(navigator.steps)
                    assert replay.page is fresh
                    assert fresh.url == URL + "team"
                finally:
                    await replay.close()
            finally:
                await navigator.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("destination", ["service", "whatsapp", "unrelated"])
def test_document_commit_between_url_capture_and_read_is_reattested(
    destination: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def exercise() -> None:
        pages = {
            "/": """<body><button id="continue"
                onclick="window.continueRequested=true">Continue</button></body>""",
            "/next": "<body>Service contact choices ready</body>",
        }
        async with website(pages) as (browser, context, page, visits, serve):
            navigator = WebsiteNavigator(page, URL, limits(), recipient=PHONE)
            original_locator = page.locator
            committed = False
            if destination == "service":
                next_url = URL + "next"
            elif destination == "whatsapp":
                next_url = f"https://web.whatsapp.com/send?phone={PHONE}"
            else:
                next_url = "https://unrelated.example/"
                await context.route(
                    next_url,
                    lambda route: route.fulfill(
                        content_type="text/html", body="<body>UNAUTHORIZED CONTENT</body>"
                    ),
                )

            def scheduled_locator(selector: str, **kwargs: Any) -> Any:
                locator = original_locator(selector, **kwargs)
                if selector != "body":
                    return locator

                async def evaluate(expression: str, arg: Any = None) -> Any:
                    nonlocal committed
                    if not committed and await page.evaluate("window.continueRequested === true"):
                        # Commit an actual Chromium navigation in the exact race
                        # window, after _state captured the old URL but before its
                        # guarded body read. All DOM/URL checks remain native.
                        committed = True
                        await page.goto(next_url)
                    return await locator.evaluate(expression, arg)

                return SimpleNamespace(evaluate=evaluate)

            try:
                observed = await navigator.observe()
                candidate = observed["catalog"]["candidates"][0]
                monkeypatch.setattr(page, "locator", scheduled_locator)
                if destination == "unrelated":
                    with pytest.raises(ChannelError, match="unrelated destination"):
                        await navigator.act(observed["revision"], "open", candidate["id"])
                    assert navigator._history[-1]["result"] == "uncertain"
                else:
                    await navigator.act(observed["revision"], "open", candidate["id"])
                    assert committed
                    assert len(navigator.steps) == 1
                    if destination == "whatsapp":
                        assert await navigator.terminal()
                        assert navigator.feedback == "recipient_attested_whatsapp_handoff"
                    else:
                        assert page.url == next_url
                        assert navigator.feedback == "navigation_completed"
                        assert (
                            "Service contact choices ready"
                            in (await navigator.observe())["page_context"]
                        )
                assert ACCOUNT_CANARY not in json.dumps(navigator._history)
                assert "UNAUTHORIZED CONTENT" not in json.dumps(navigator._history)
            finally:
                monkeypatch.setattr(page, "locator", original_locator)
                await navigator.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("target", ["webchat", "whatsapp"])
def test_click_only_reveal_reopens_after_dead_branch_back_and_fresh_replays(
    target: str,
) -> None:
    async def exercise() -> None:
        final = (
            f'<a id="team" href="https://web.whatsapp.com/send?phone={PHONE}">Team support</a>'
            if target == "whatsapp"
            else '<button id="team" '
            "onclick=\"document.querySelector('#chat').hidden=false\">Team support</button>"
        )
        pages = {
            "/": """<body><button id="options"
                onclick="document.querySelector('#routes').hidden=false">Options</button>
                <nav id="routes" hidden>
                <a id="branch" href="/dead">Service information</a>"""
            + final
            + """</nav><section id="chat" role="dialog" aria-label="Chat assistant" hidden>
                <textarea aria-label="Message"></textarea></section></body>""",
            "/dead": "<body>No contact options here.</body>",
        }
        async with website(pages) as (browser, context, page, visits, serve):
            broker = NativeChatBroker(
                URL,
                page=page,
                limits=limits(),
                website_recipient=PHONE if target == "whatsapp" else None,
            )
            await broker.start()
            try:
                await choose(broker, "Options")
                await choose(broker, "Service information")
                dead = await broker.observe_website()
                await broker.act_website(dead["revision"], "back")
                returned = await broker.observe_website()
                menu = next(
                    item
                    for item in returned["catalog"]["candidates"]
                    if item["description"] == "Options"
                )
                assert "open" in menu["actions"]
                await broker.act_website(returned["revision"], "open", menu["id"])
                revealed = await broker.observe_website()
                branch = next(
                    item
                    for item in revealed["catalog"]["candidates"]
                    if item["description"] == "Service information"
                )
                assert "open" in branch["tried_actions"]
                assert "open" not in branch["actions"]
                await choose(broker, "Team support")
                assert (await broker.observe_website())["terminal"] == target
                steps = list(broker.website_navigator.steps)
                assert [step.action for step in steps] == [
                    "click",
                    "click",
                    "back",
                    "click",
                    "click",
                ]
                fresh = await context.new_page()
                await fresh.goto(URL)
                replay = WebsiteNavigator(
                    fresh, URL, limits(), recipient=PHONE if target == "whatsapp" else None
                )
                try:
                    await replay.replay(steps)
                    assert replay.steps == steps
                    if target == "whatsapp":
                        assert await replay.terminal()
                    else:
                        assert await fresh.locator("#chat").is_visible()
                finally:
                    await replay.close()
            finally:
                await broker.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("target", ["webchat", "whatsapp"])
def test_follow_observed_link_is_a_reproducible_action_not_a_click_retry(target: str) -> None:
    async def exercise() -> None:
        destination = (
            f"https://api.whatsapp.com/send?phone={PHONE}"
            if target == "whatsapp"
            else "/conversation"
        )
        pages = {
            "/": (
                f'<body><a id="route" href="{destination}" '
                "onclick=\"event.preventDefault();fetch('/forbidden-handler')\">"
                "Contact our team</a></body>"
            ),
            "/conversation": '<body><section role="dialog" aria-label="Support chat">'
            '<textarea aria-label="Message"></textarea></section></body>',
            "landing": (
                '<body><a id="web" target="_blank" '
                f'href="https://web.whatsapp.com/send?phone={PHONE}" '
                "onclick=\"event.preventDefault();fetch('/forbidden-handler')\">"
                "Continue to WhatsApp Web</a></body>"
            ),
        }
        async with website(pages) as (browser, context, page, visits, serve):
            broker = NativeChatBroker(
                URL,
                page=page,
                limits=limits(),
                website_recipient=PHONE if target == "whatsapp" else None,
            )
            await broker.start()
            navigator = None
            try:
                await choose(broker, "Contact our team", "follow_link")
                if target == "whatsapp":
                    await choose(broker, "Continue to WhatsApp Web", "follow_link")
                assert (await broker.observe_website())["terminal"] == target
                steps = list(broker.website_navigator.steps)
                learned_visits = list(visits)
                fresh = await context.new_page()
                await fresh.goto(URL)
                replay = NativeChatBroker(
                    URL,
                    page=fresh,
                    limits=limits(),
                    website_recipient=PHONE if target == "whatsapp" else None,
                )
                await replay.start()
                try:
                    navigator = replay.website_navigator
                    await navigator.replay(steps)
                    assert (await replay.observe_website())["terminal"] == target
                    assert visits == learned_visits * 2
                finally:
                    await replay.close()
            finally:
                await broker.close()

    asyncio.run(exercise())


def test_scroll_precedes_offscreen_action_and_late_overlay_never_dispatches_click() -> None:
    async def exercise() -> None:
        pages = {
            "/": """<body style="min-height:1800px">
            <button id="details" style="position:absolute;top:900px"
             onclick="document.querySelector('#result').hidden=false">Open details</button>
            <p id="result" hidden>Contact details are available</p></body>""",
        }
        async with website(pages) as (browser, context, page, visits, serve):
            broker = NativeChatBroker(URL, page=page, limits=limits())
            await broker.start()
            try:
                initial = await broker.observe_website()
                offscreen = next(
                    item
                    for item in initial["catalog"]["candidates"]
                    if item["description"] == "Open details"
                )
                assert offscreen["actions"] == []
                with pytest.raises(ChannelError, match="unavailable"):
                    await broker.act_website(initial["revision"], "open", offscreen["id"])
                await broker.act_website(initial["revision"], "scroll_down")
                visible = await broker.observe_website()
                selected = next(
                    item
                    for item in visible["catalog"]["candidates"]
                    if item["description"] == "Open details"
                )
                await page.evaluate("""() => {
                  const overlay=document.createElement('div'); overlay.id='late-overlay';
                  overlay.style='position:fixed;inset:0;background:white;z-index:99';
                  overlay.innerHTML='<button id="close">Close browsing prompt</button>';
                  overlay.querySelector('button').onclick=()=>overlay.remove();
                  document.body.append(overlay);
                }""")
                with pytest.raises(ChannelError) as error:
                    await broker.act_website(visible["revision"], "open", selected["id"])
                assert error.value.operation == "website.readiness"
                assert not await page.locator("#result").is_visible()
                assert [step.action for step in broker.website_navigator.steps] == ["scroll_down"]
                await choose(broker, "Close browsing prompt")
                await choose(broker, "Open details")
                assert await page.locator("#result").is_visible()
            finally:
                await broker.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("transient", [True, False])
def test_exact_replay_waits_for_transient_cover_without_dismissing_unknown_overlays(
    transient: bool,
) -> None:
    async def exercise() -> None:
        pages = {
            "/": """<body><button id="contact"
             onclick="document.querySelector('#result').hidden=false">Contact options</button>
             <p id="result" hidden>Available contact options</p>
             <div id="cover" style="position:fixed;inset:0;background:white;z-index:99">
             Preparing options</div></body>""",
        }
        async with website(pages) as (browser, context, page, visits, serve):
            broker = NativeChatBroker(URL, page=page, limits=limits())
            await broker.start()
            steps = [
                WebsiteNavigationStep.model_validate(
                    {"action": "click", "locator": {"strategy": "css", "value": "#contact"}}
                )
            ]
            try:
                if transient:
                    await page.evaluate(
                        "() => setTimeout(() => document.querySelector('#cover').remove(), 500)"
                    )
                    await broker.website_navigator.replay(steps)
                    assert await page.locator("#result").is_visible()
                    assert broker.website_navigator.steps == steps
                else:
                    with pytest.raises(ChannelError, match="blocked|not ready"):
                        await broker.website_navigator.replay(steps)
                    assert not await page.locator("#result").is_visible()
                    assert await page.locator("#cover").is_visible()
                    assert broker.website_navigator.steps == []
            finally:
                await broker.close()

    asyncio.run(exercise())

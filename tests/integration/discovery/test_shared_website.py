"""Both channel discoveries navigate the same real Chromium website surface."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

import pytest
from playwright.async_api import Route, async_playwright

from syntheticcx.adapters.channel.broker import NativeChatBroker
from syntheticcx.config.models import ExecutionLimits, PrivacyConfig
from syntheticcx.discovery.budget import DiscoveryBudget
from syntheticcx.discovery.learn import explore_website, learn
from syntheticcx.discovery.replay import replay
from syntheticcx.errors import ChannelError, ModelError
from syntheticcx.privacy import Redactor

URL = "https://service.example/"
PHONE = "15551234567"
FIELDS = (
    "target_id",
    "assistant_id",
    "customer_id",
    "input_id",
    "send_id",
    "option_id",
    "send_mode",
    "probe_id",
    "stop_reason",
)


class WebsiteProvider:
    def __init__(self, *, invalid: bool = False) -> None:
        self.requests: list[dict[str, Any]] = []
        self.invalid = invalid

    async def generate(self, request: Any) -> Any:
        payload = json.loads(request.messages[-1].content)
        self.requests.append(payload)
        inventory = payload["inventory"]
        offered_opens = {
            choice["target_id"]
            for choice in payload["action_choices"]
            if choice["action"] == "open"
        }
        candidates = [
            item for item in inventory["catalog"]["candidates"] if item["id"] in offered_opens
        ]
        preferred = next(
            (
                item
                for label in ("WhatsApp Claro", "Explore", "Next", "Continue")
                for item in candidates
                if item["description"] == label
            ),
            None,
        )
        choice = (
            {"action": "open", "target_id": preferred["id"]} if preferred else {"action": "wait"}
        )
        if "target_channel" not in payload:
            choice = next(
                item
                for action in ("finish", "bind_scope", "bind_reader", "bind_state", "submit_probe")
                for item in payload["action_choices"]
                if item["action"] == action
            )
        if self.invalid:
            choice = {"action": "open", "target_id": "a[href]"}
        decision = {
            **dict.fromkeys(FIELDS),
            "revision": inventory["revision"],
            **choice,
        }
        return SimpleNamespace(content=json.dumps(decision))


def redactor() -> Redactor:
    return Redactor(PrivacyConfig(), resolved_secrets=[PHONE, "PRIVATE-CANARY"])


@pytest.mark.parametrize("target", ["webchat", "whatsapp"])
@pytest.mark.parametrize("lazy", [False, True])
def test_shared_engine_inspects_generic_homepage_intermediate_widget_and_lazy_control(
    target: str,
    lazy: bool,
) -> None:
    async def exercise() -> None:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            broker = None
            visits: list[str] = []
            try:
                page = await browser.new_page()

                async def route(request: Route) -> None:
                    path = urlsplit(request.request.url).path
                    visits.append(path)
                    if path == "/":
                        launcher = '<a id="explore" href="/services">Explore</a>'
                        if lazy:
                            launcher = (
                                "<script>setTimeout(() => document.body.insertAdjacentHTML("
                                f"'beforeend', {json.dumps(launcher)}), 120)</script>"
                            )
                        body = "<body><p>Find your service options PRIVATE-CANARY</p>" + launcher
                    elif path == "/services":
                        final = (
                            f'<a id="continue" href="https://web.whatsapp.com/send?phone={PHONE}">Continue</a>'
                            if target == "whatsapp"
                            else (
                                '<section role="dialog" aria-label="Chat assistant">'
                                '<textarea aria-label="Message"></textarea></section>'
                            )
                        )
                        body = (
                            "<body><p>Choose how to continue with our team</p>"
                            '<button id="next" '
                            "onclick=\"document.querySelector('#widget').hidden=false\">"
                            "Next</button>"
                            f'<div id="widget" hidden>{final}</div>'
                        )
                    else:
                        body = "<body>ACCOUNT CONTENT MUST NOT BE OBSERVED</body>"
                    await request.fulfill(content_type="text/html", body=body)

                await page.route("**/*", route)
                await page.goto(URL)
                broker = NativeChatBroker(
                    URL,
                    page=page,
                    website_recipient=PHONE if target == "whatsapp" else None,
                )
                await broker.start()
                provider = WebsiteProvider()
                await explore_website(
                    broker,
                    provider,
                    redactor(),
                    target=target,
                    budget=DiscoveryBudget.create(15),
                    browser_action_timeout_seconds=5,
                )
                assert visits[:2] == ["/", "/services"]
                snapshots = [request["inventory"] for request in provider.requests]
                descriptions = {
                    item["description"]
                    for snapshot in snapshots
                    for item in snapshot["catalog"]["candidates"]
                }
                assert {"Explore", "Next"} <= descriptions
                assert any(
                    "Choose how to continue" in item.get("page_context", "") for item in snapshots
                )
                assert "PRIVATE-CANARY" not in json.dumps(provider.requests)
                assert "ACCOUNT CONTENT" not in json.dumps(provider.requests)
                assert all(request["target_channel"] == target for request in provider.requests)
                assert (await broker.observe_website())["terminal"] == target
            finally:
                if broker is not None:
                    await broker.close()
                await browser.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("target", ["webchat", "whatsapp"])
@pytest.mark.parametrize("icon_only", [False, True])
def test_dense_homepage_keeps_late_generic_missing_type_launcher_in_shared_inventory(
    target: str,
    icon_only: bool,
) -> None:
    async def exercise() -> None:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            broker = None
            try:
                page = await browser.new_page()
                unrelated = "".join(
                    f'<a href="/company-page-{index}">Company information {index}</a>'
                    for index in range(250)
                )
                label = (
                    '<svg aria-hidden="true" width="24" height="24">'
                    '<circle cx="12" cy="12" r="8"/></svg>'
                    if icon_only
                    else "Explore"
                )
                activation = (
                    f"location.href='https://web.whatsapp.com/send?phone={PHONE}'"
                    if target == "whatsapp"
                    else "document.querySelector('#chat').hidden=false"
                )
                website = (
                    f"<body>{unrelated}<section><p>Find your service options</p>"
                    '<button id="late-launcher" aria-label="Explore" '
                    f'style="position:fixed;bottom:12px;right:12px" onclick="{activation}">'
                    f"{label}</button></section>"
                    '<section id="chat" role="dialog" aria-label="Chat assistant" hidden>'
                    '<textarea aria-label="Message"></textarea></section></body>'
                )

                async def route(request: Route) -> None:
                    if request.request.url == URL:
                        await request.fulfill(content_type="text/html", body=website)
                    elif urlsplit(request.request.url).hostname == "web.whatsapp.com":
                        await request.fulfill(body="<body>PRIVATE ACCOUNT CONTENT</body>")
                    else:
                        pytest.fail("unrelated homepage navigation was activated")

                await page.route("**/*", route)
                await page.goto(URL)
                broker = NativeChatBroker(
                    URL,
                    page=page,
                    website_recipient=PHONE if target == "whatsapp" else None,
                )
                await broker.start()
                provider = WebsiteProvider()
                await explore_website(
                    broker,
                    provider,
                    redactor(),
                    target=target,
                    budget=DiscoveryBudget.create(15),
                    browser_action_timeout_seconds=5,
                )
                assert len(provider.requests) == 1
                inventory = provider.requests[0]["inventory"]
                assert inventory["catalog"]["has_more"]
                assert len(inventory["catalog"]["candidates"]) <= 160
                launcher = next(
                    item
                    for item in inventory["catalog"]["candidates"]
                    if item["description"] == "Explore"
                )
                assert {"action": "open", "target_id": launcher["id"]} in (
                    provider.requests[0]["action_choices"]
                )
                assert [step.locator.value for step in broker.website_navigator.steps] == [
                    "#late-launcher"
                ]
                assert (await broker.observe_website())["terminal"] == target
                assert "PRIVATE ACCOUNT CONTENT" not in json.dumps(provider.requests)
            finally:
                if broker is not None:
                    await broker.close()
                await browser.close()

    asyncio.run(exercise())


def test_exact_id_hint_pins_shared_initial_navigation_not_model_preference() -> None:
    async def exercise() -> None:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            broker = None
            try:
                page = await browser.new_page()
                visits: list[str] = []

                async def route(request: Route) -> None:
                    visits.append(request.request.url)
                    body = (
                        '<a id="other" href="/wrong">Explore</a>'
                        f'<a id="chosen" href="https://web.whatsapp.com/send?phone={PHONE}">'
                        "Explore</a>"
                        if urlsplit(request.request.url).hostname == "service.example"
                        else "PRIVATE ACCOUNT CONTENT"
                    )
                    await request.fulfill(content_type="text/html", body=body)

                await page.route("**/*", route)
                await page.goto(URL)
                broker = NativeChatBroker(URL, page=page, website_recipient=PHONE)
                await broker.start()
                broker.chat_hint = "id=chosen"
                provider = WebsiteProvider()
                await explore_website(
                    broker,
                    provider,
                    redactor(),
                    target="whatsapp",
                    budget=DiscoveryBudget.create(10),
                    browser_action_timeout_seconds=3,
                )
                assert len(provider.requests) == 1
                assert visits == [URL, f"https://web.whatsapp.com/send?phone={PHONE}"]
                assert [step.locator.value for step in broker.website_navigator.steps] == [
                    "#chosen"
                ]
                assert "PRIVATE ACCOUNT CONTENT" not in json.dumps(provider.requests)
            finally:
                if broker is not None:
                    await broker.close()
                await browser.close()

    asyncio.run(exercise())


def test_whatsapp_exploration_identifies_widget_without_authorizing_its_send() -> None:
    async def exercise() -> None:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            broker = None
            visits: list[str] = []
            try:
                page = await browser.new_page()

                async def route(request: Route) -> None:
                    visits.append(request.request.url)
                    body = (
                        '<section role="dialog" aria-label="Chat assistant">'
                        '<textarea aria-label="Message">Unsubmitted draft</textarea>'
                        '<button id="send" onclick="fetch(\'/forbidden-send\')">Send</button>'
                        f'<a id="continue" href="https://web.whatsapp.com/send?phone={PHONE}">'
                        "Continue</a></section>"
                        if urlsplit(request.request.url).hostname == "service.example"
                        else "PRIVATE ACCOUNT CONTENT"
                    )
                    await request.fulfill(content_type="text/html", body=body)

                await page.route("**/*", route)
                await page.goto(URL)
                broker = NativeChatBroker(URL, page=page, website_recipient=PHONE)
                await broker.start()
                observation = await broker.observe_website()
                assert observation["observed_channel"] == "webchat"
                assert [item["description"] for item in observation["catalog"]["candidates"]] == [
                    "Continue"
                ]
                await explore_website(
                    broker,
                    WebsiteProvider(),
                    redactor(),
                    target="whatsapp",
                    budget=DiscoveryBudget.create(10),
                    browser_action_timeout_seconds=3,
                )
                assert visits == [URL, f"https://web.whatsapp.com/send?phone={PHONE}"]
            finally:
                if broker is not None:
                    await broker.close()
                await browser.close()

    asyncio.run(exercise())


def test_shared_engine_rejects_unoffered_selector_without_activating_browser_control() -> None:
    async def exercise() -> None:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            broker = None
            try:
                page = await browser.new_page()
                await page.route(
                    "**/*",
                    lambda route: route.fulfill(
                        body="""<body>
                    <button id="explore" onclick="window.clicks++">Explore</button>
                    <button id="pay" onclick="window.clicks++">Pay</button>
                    <form><button onclick="window.clicks++">Continue</button></form>
                    <a href="https://unrelated.example/">Elsewhere</a>
                    <a href="https://wa.me/15557654321">Wrong recipient</a>
                    <script>window.clicks=0;</script></body>"""
                    ),
                )
                await page.goto(URL)
                broker = NativeChatBroker(URL, page=page, website_recipient=PHONE)
                await broker.start()
                provider = WebsiteProvider(invalid=True)
                with pytest.raises(ModelError):
                    await explore_website(
                        broker,
                        provider,
                        redactor(),
                        target="whatsapp",
                        budget=DiscoveryBudget.create(5),
                        max_discovery_attempts=2,
                    )
                assert await page.evaluate("window.clicks") == 0
                assert len(provider.requests) == 2
                assert provider.requests[1]["correction"]
                assert {
                    item["description"]
                    for request in provider.requests
                    for item in request["inventory"]["catalog"]["candidates"]
                } == {"Explore"}
                assert broker.website_navigator.steps == []
            finally:
                if broker is not None:
                    await broker.close()
                await browser.close()

    asyncio.run(exercise())


def test_native_shared_controls_recheck_revision_destination_and_consequences() -> None:
    async def exercise() -> None:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            broker = None
            try:
                page = await browser.new_page()
                await page.route(
                    "**/*",
                    lambda route: route.fulfill(
                        body="""<body>
                    <a id="explore" href="/services">Explore</a>
                    <button id="next" onclick="window.clicks++">Next</button>
                    <script>window.clicks=0;</script></body>"""
                    ),
                )
                await page.goto(URL)
                broker = NativeChatBroker(URL, page=page)
                await broker.start()
                old = await broker.observe_website()
                old_control = old["catalog"]["candidates"][0]
                await broker.observe_website()
                with pytest.raises(ChannelError):
                    await broker.act_website(old["revision"], "open", old_control["id"])
                current = await broker.observe_website()
                link = next(
                    item
                    for item in current["catalog"]["candidates"]
                    if item["description"] == "Explore"
                )
                await page.locator("#explore").evaluate(
                    "node => node.href='https://unrelated.example/'"
                )
                with pytest.raises(ChannelError):
                    await broker.act_website(current["revision"], "open", link["id"])
                current = await broker.observe_website()
                button = next(
                    item
                    for item in current["catalog"]["candidates"]
                    if item["description"] == "Next"
                )
                await page.locator("#next").evaluate("node => node.textContent='Pay'")
                with pytest.raises(ChannelError):
                    await broker.act_website(current["revision"], "open", button["id"])
                assert page.url == URL
                assert await page.evaluate("window.clicks") == 0
                assert broker.website_navigator.steps == []
            finally:
                if broker is not None:
                    await broker.close()
                await browser.close()

    asyncio.run(exercise())


def test_active_webchat_learning_replays_generic_navigation_and_owned_exchange() -> None:
    async def exercise() -> None:
        async with async_playwright() as pw:
            learning_browser = await pw.chromium.launch()
            fresh_browser = None
            learning = fresh = None
            visits: list[str] = []
            submissions: list[str] = []
            provider = WebsiteProvider()
            limits = ExecutionLimits(
                max_turns=3,
                journey_timeout_seconds=30,
                greeting_grace_seconds=0,
                browser_action_timeout_seconds=3,
                response_timeout_seconds=2,
                model_timeout_seconds=2,
                evaluator_timeout_seconds=2,
                persistence_timeout_seconds=2,
                close_timeout_seconds=2,
            )

            async def route(request: Route) -> None:
                path = urlsplit(request.request.url).path
                if path == "/reply":
                    submissions.append(request.request.post_data_json["text"])
                    await request.fulfill(json={"text": "Fresh information for this request."})
                    return
                visits.append(path)
                if path == "/":
                    body = '<body><a id="explore" href="/services">Explore</a></body>'
                else:
                    body = """<body><button id="next"
                    onclick="document.querySelector('#chat').hidden=false">Next</button>
                    <section id="chat" role="dialog" aria-label="Chat assistant" hidden>
                    <article class="assistant">Welcome to our information service.</article>
                    <textarea id="composer" aria-label="Message"></textarea></section>
                    <script>
                    document.querySelector('textarea').onkeydown=async event=>{
                      if(event.key!=='Enter') return;
                      event.preventDefault();
                      const text=event.target.value; event.target.value='';
                      const customer=document.createElement('article');
                      customer.className='customer';customer.textContent=text;
                      document.querySelector('#chat').append(customer);
                      const response=await fetch('/reply',{method:'POST',
                        headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
                      const assistant=document.createElement('article');
                      assistant.className='assistant';
                      assistant.textContent=(await response.json()).text;
                      document.querySelector('#chat').append(assistant);
                    };
                    </script></body>"""
                await request.fulfill(content_type="text/html", body=body)

            try:
                page = await learning_browser.new_page()
                await page.route("**/*", route)
                await page.goto(URL)
                learning = NativeChatBroker(URL, limits, page=page)
                await learning.start()
                await explore_website(
                    learning,
                    provider,
                    redactor(),
                    budget=DiscoveryBudget.create(10),
                    browser_action_timeout_seconds=3,
                )
                learned = await learn(
                    learning,
                    provider,
                    redactor(),
                    budget=DiscoveryBudget.create(20),
                    probes={"explicit": "Owned repeatable information probe"},
                    browser_action_timeout_seconds=3,
                    response_timeout_seconds=2,
                )
                assert [step.locator.value for step in learned.config.navigation_steps] == [
                    "#explore",
                    "#next",
                ]
                assert len(learned.config.transitions) == 1
                await learning.close()
                await learning_browser.close()
                requests_before_replay = len(provider.requests)
                fresh_browser = await pw.chromium.launch()
                page = await fresh_browser.new_page()
                await page.route("**/*", route)
                await page.goto(URL)
                fresh = NativeChatBroker(URL, limits, page=page, contract=learned.config)
                await fresh.start()
                guide = await replay(
                    fresh,
                    learned,
                    budget=DiscoveryBudget.create(10),
                    browser_action_timeout_seconds=3,
                    response_timeout_seconds=2,
                )
                assert visits == ["/", "/services", "/", "/services"]
                assert submissions == ["Owned repeatable information probe"] * 2
                assert len(provider.requests) == requests_before_replay
                assert guide.validation.status == "verified"
            finally:
                if learning is not None:
                    await learning.close()
                if fresh is not None:
                    await fresh.close()
                await learning_browser.close()
                if fresh_browser is not None:
                    await fresh_browser.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("target", ["webchat", "whatsapp"])
def test_title_named_homepage_icon_with_badge_precedes_dense_generic_menus(target: str) -> None:
    async def exercise() -> None:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            broker = None
            visits: list[str] = []
            try:
                page = await browser.new_page()
                menus = "".join(
                    f'<button id="menu-{index}" onclick="fetch(\'/wrong-menu\')">'
                    f"Menu {index}</button>"
                    for index in range(200)
                )
                destination = (
                    f'<a id="continue" href="https://web.whatsapp.com/send?phone={PHONE}">'
                    "Continue</a>"
                    if target == "whatsapp"
                    else '<section role="dialog" aria-label="Chat assistant">'
                    '<textarea aria-label="Message"></textarea></section>'
                )
                body = (
                    f"<body>{menus}"
                    '<button title="WhatsApp Claro" onclick="'
                    "document.querySelector('#choices').hidden=false\">"
                    '<span>1</span><img alt="WhatsApp Claro" src="data:,"></button>'
                    f'<div id="choices" hidden>{destination}</div></body>'
                )

                async def route(request: Route) -> None:
                    visits.append(request.request.url)
                    assert "/wrong-menu" not in request.request.url
                    await request.fulfill(
                        content_type="text/html",
                        body=body if request.request.url == URL else "PRIVATE ACCOUNT CONTENT",
                    )

                await page.route("**/*", route)
                await page.goto(URL)
                broker = NativeChatBroker(
                    URL,
                    page=page,
                    chat_hint="WhatsApp Claro",
                    website_recipient=PHONE if target == "whatsapp" else None,
                )
                await broker.start()
                provider = WebsiteProvider()
                await explore_website(
                    broker,
                    provider,
                    redactor(),
                    target=target,
                    budget=DiscoveryBudget.create(15),
                    browser_action_timeout_seconds=5,
                )
                first = provider.requests[0]["inventory"]["catalog"]["candidates"][0]
                assert first["description"] == "WhatsApp Claro"
                assert (
                    broker.website_navigator.steps[0].locator.value
                    == 'button[title="WhatsApp Claro"]'
                )
                assert (await broker.observe_website())["terminal"] == target
                assert visits == (
                    [URL, f"https://web.whatsapp.com/send?phone={PHONE}"]
                    if target == "whatsapp"
                    else [URL]
                )
            finally:
                if broker is not None:
                    await broker.close()
                await browser.close()

    asyncio.run(exercise())

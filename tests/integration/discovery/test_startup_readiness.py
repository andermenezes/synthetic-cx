"""Startup waits for usable owned controls without interacting with the page."""

import asyncio
import time

import pytest
from playwright.async_api import Page, async_playwright

from syntheticcx.adapters.channel.broker import PREOWNED_SCAN_SELECTOR, NativeChatBroker
from syntheticcx.adapters.channel.readiness import wait_for_interaction
from syntheticcx.config.models import ExecutionLimits
from syntheticcx.errors import ChannelError

URL = "http://localhost:8765/"


async def load(page: Page, markup: str) -> None:
    await page.route("**/*", lambda route: route.fulfill(body=markup, content_type="text/html"))
    await page.goto(URL)


@pytest.mark.parametrize("shadow", [False, True])
def test_structural_panel_with_contenteditable_composer_is_ready(shadow):
    async def exercise():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                page = await browser.new_page()
                await load(page, '<button id="launcher">Open chat</button>')
                await page.evaluate(
                    """shadow => {
                    const target = shadow ? document.body.appendChild(
                        document.createElement('div')).attachShadow({mode:'open'}) : document.body;
                    target.innerHTML += '<div id="panel">' +
                        '<button title="Close chat">Close</button>' +
                        '<div role="textbox" contenteditable="true"></div>' +
                        '<button title="Send message">Send</button></div>';
                }""",
                    shadow,
                )
                opener = await page.locator("#launcher").element_handle()
                assert opener is not None
                try:
                    await wait_for_interaction(page, URL, opener, time.monotonic() + 1)
                finally:
                    await opener.dispose()
            finally:
                await browser.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("shadow", [False, True])
def test_delayed_root_is_not_ready_until_composer_appears(shadow: bool) -> None:
    async def exercise() -> None:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                page = await browser.new_page()
                await load(page, '<button id="launcher" aria-controls="chat">Chat</button>')
                await page.evaluate(
                    """shadow => {
                      window.clicks = 0;
                      document.addEventListener('click', () => window.clicks++);
                      const host = shadow ? document.body.appendChild(
                        document.createElement('div')) : null;
                      const target = shadow ? host.attachShadow({mode:'open'}) : document.body;
                      setTimeout(() => target.innerHTML +=
                        '<section id="chat" role="dialog" aria-label="Chat">' +
                        '<button type="button">Close</button><div role="status">Loading</div>' +
                        '</section>', 100);
                      setTimeout(() => target.querySelector('#chat').innerHTML +=
                        '<input id="composer" aria-label="Message">', 350);
                    }""",
                    shadow,
                )
                opener = await page.locator("#launcher").element_handle()
                started = time.monotonic()
                try:
                    await wait_for_interaction(page, URL, opener, started + 3)
                finally:
                    await opener.dispose()
                assert time.monotonic() - started >= 0.3
                assert await page.locator("#composer").count() == 1
                assert await page.evaluate("window.clicks") == 0
            finally:
                await browser.close()

    asyncio.run(exercise())


def test_unmarked_panel_is_pinned_without_requiring_idle_composer_layout() -> None:
    async def exercise() -> None:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await load(
                page,
                """<button type="button" id="launcher">Chat</button>
                <div id="panel" style="display:none">
                <button type="button" aria-label="Close chat">Close</button>
                <div id="messages">Welcome</div><textarea></textarea>
                <button type="button" aria-label="Send message">Send</button></div>
                <script>launcher.onclick=()=>panel.style.display='block';</script>""",
            )
            broker = NativeChatBroker(URL, page=page)
            try:
                await broker.start()
                observation = await broker.observe()
                opener = next(
                    c for c in observation["catalog"]["candidates"] if c["kind"] == "opener"
                )
                await broker.open(opener["id"], revision=observation["revision"])
                await broker.wait_until_ready()
                observation = await broker.observe()
                roots = [c for c in observation["catalog"]["candidates"] if c["kind"] == "scope"]
                assert len(roots) == 1
                await broker.bind_scope(roots[0]["id"], revision=observation["revision"])
                assert broker.owned is not None and broker.owned.proof == "chat_panel"
                await page.locator("textarea").evaluate("node => node.remove()")
                await broker.owned.check()
                await page.get_by_role("button", name="Close chat").evaluate(
                    "node => node.remove()"
                )
                with pytest.raises(ChannelError, match="panel proof changed"):
                    await broker.owned.check()
            finally:
                await broker.close()
                await browser.close()

    asyncio.run(exercise())


def test_header_buttons_and_spinner_do_not_establish_readiness() -> None:
    async def exercise() -> None:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                page = await browser.new_page()
                await load(
                    page,
                    '<section id="chat" role="dialog" aria-label="Chat">'
                    '<button type="button">Close</button><button type="button">Plans</button>'
                    '<div role="status">Loading</div></section>',
                )
                started = time.monotonic()
                with pytest.raises(ChannelError, match="startup deadline"):
                    await wait_for_interaction(page, URL, None, started + 0.2)
                assert time.monotonic() - started < 0.8
            finally:
                await browser.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "controls",
    [
        "<input disabled>",
        '<input aria-disabled="true">',
        "<input readonly>",
        '<input aria-label="Credit card">',
        "<input><textarea></textarea>",
    ],
)
def test_unavailable_or_ambiguous_composer_does_not_establish_readiness(controls: str) -> None:
    async def exercise() -> None:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                page = await browser.new_page()
                await load(
                    page,
                    '<section id="chat" role="dialog" aria-label="Chat">' + controls + "</section>",
                )
                with pytest.raises(ChannelError, match="startup deadline"):
                    await wait_for_interaction(page, URL, None, time.monotonic() + 0.15)
            finally:
                await browser.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("actor", "label", "ready"),
    [("assistant", "Plans", True), ("assistant", "Close", False), ("", "Plans", False)],
)
def test_only_assistant_owned_informational_menu_is_ready(
    actor: str, label: str, ready: bool
) -> None:
    async def exercise() -> None:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                page = await browser.new_page()
                await load(
                    page,
                    '<section id="chat" role="dialog" aria-label="Chat">'
                    f'<article data-actor="{actor}">Choose an information topic'
                    f'<button type="button">{label}</button></article></section>',
                )
                if ready:
                    await wait_for_interaction(page, URL, None, time.monotonic() + 1)
                else:
                    with pytest.raises(ChannelError, match="startup deadline"):
                        await wait_for_interaction(page, URL, None, time.monotonic() + 0.15)
            finally:
                await browser.close()

    asyncio.run(exercise())


def test_parent_origin_change_is_fatal_without_waiting_for_deadline() -> None:
    async def exercise() -> None:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                page = await browser.new_page()
                await load(page, '<section id="chat" role="dialog" aria-label="Chat"></section>')
                waiting = asyncio.create_task(
                    wait_for_interaction(page, URL, None, time.monotonic() + 2)
                )
                await asyncio.sleep(0.1)
                await page.goto("http://localhost:9999/")
                with pytest.raises(ChannelError, match="origin changed"):
                    await waiting
            finally:
                await browser.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("delay_ms", [0, 600])
def test_text_chat_hint_scans_large_page_narrowly_and_authorizes_named_launcher(
    delay_ms: int,
) -> None:
    async def exercise() -> None:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            markup = """<!doctype html><body><script>
              for (let index=0; index<25005; index++) {
                const node=document.createElement('div'); node.textContent='content';
                document.body.append(node);
              }
              const button=document.createElement('a');
              button.setAttribute('role','button');
              button.setAttribute('aria-label','Abrir chat');
              button.title='Pergunte à Aura, sou cliente Vivo e tenho dúvidas';
              button.style.cssText='display:block;width:120px;height:40px';
              button.onclick=()=>document.body.insertAdjacentHTML('beforeend',
                '<section id="chat" role="dialog" aria-label="Aura chat"></section>');
              setTimeout(()=>document.body.append(button), DELAY_MS);
            </script></body>"""
            markup = markup.replace("DELAY_MS", str(delay_ms))
            await page.route(
                "**/*",
                lambda route: route.fulfill(body=markup, content_type="text/html; charset=utf-8"),
            )
            await page.goto(URL)
            broker = NativeChatBroker(
                URL,
                ExecutionLimits(
                    max_turns=2,
                    journey_timeout_seconds=10,
                    greeting_grace_seconds=0,
                    browser_action_timeout_seconds=2,
                    response_timeout_seconds=2,
                    model_timeout_seconds=2,
                    evaluator_timeout_seconds=2,
                    persistence_timeout_seconds=2,
                    close_timeout_seconds=2,
                ),
                page=page,
                chat_hint="Pergunte à Aura",
            )
            try:
                await broker.start()
                assert broker.owned is None and broker._open_node is None
                observation = await broker.observe()
                candidates = observation["catalog"]["candidates"]
                opener = next((item for item in candidates if item["kind"] == "opener"), None)
                assert opener is not None, observation
                assert await page.locator("body").locator(PREOWNED_SCAN_SELECTOR).count() == 1
                assert "Abrir chat" in opener["description"]
                await page.locator("a[role=button]").evaluate(
                    "node => node.removeAttribute('aria-label')"
                )
                await broker.open(opener["id"], revision=observation["revision"])
                assert await page.get_by_role("dialog", name="Aura chat").count() == 1
            finally:
                await broker.close()
                await browser.close()

    asyncio.run(exercise())

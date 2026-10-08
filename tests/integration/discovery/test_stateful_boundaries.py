"""Real-browser bounds and independent ownership are not model declarations."""

import asyncio
import json

import pytest
from playwright.async_api import async_playwright

from syntheticcx.adapters.channel.broker import NativeChatBroker
from syntheticcx.errors import ChannelError

URL = "http://localhost:8765/"


@pytest.mark.parametrize("shadow", [False, True])
def test_tree_local_hidden_names_do_not_expose_editor_values(shadow):
    async def exercise():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                page = await browser.new_page()
                await page.route("**/*", lambda route: route.fulfill(body="<body></body>"))
                await page.goto(URL)
                markup = (
                    '<span id="name" hidden>Support chat</span>'
                    '<section role="dialog" aria-labelledby="name">'
                    '<article class="assistant">Information</article>'
                    "<textarea>PRIVATE_EDITOR_CANARY</textarea></section>"
                )
                if shadow:
                    await page.evaluate(
                        "html => {const host=document.createElement('div'); "
                        "document.body.append(host); "
                        "host.attachShadow({mode:'open'}).innerHTML=html;}",
                        markup,
                    )
                else:
                    await page.set_content(markup)
                broker = NativeChatBroker(URL, page=page)
                try:
                    await broker.start()
                    observed = await broker.observe()
                    assert "PRIVATE_EDITOR_CANARY" not in json.dumps(observed)
                    scope = next(
                        item
                        for item in observed["catalog"]["candidates"]
                        if item["kind"] == "scope"
                    )
                    assert scope["description"] == "Support chat"
                    await broker.bind_scope(observed["revision"], scope["id"])
                    assert "PRIVATE_EDITOR_CANARY" not in json.dumps(await broker.observe())
                finally:
                    await broker.close()
            finally:
                await browser.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("proof", ["chat_embed", "opener_linked"])
def test_loss_of_positive_root_proof_invalidates_pinned_scope(proof):
    async def exercise():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                page = await browser.new_page()
                await page.route(
                    "**/*",
                    lambda route: route.fulfill(
                        body='<button type="button" id="open" aria-controls="chat">Chat</button>'
                        '<section id="chat" data-chat-root>'
                        '<article class="assistant">Information</article></section>'
                    ),
                )
                await page.goto(URL)
                if proof == "opener_linked":
                    await page.locator("#chat").evaluate(
                        "node=>node.removeAttribute('data-chat-root')"
                    )
                broker = NativeChatBroker(URL, page=page)
                try:
                    await broker.start()
                    observed = await broker.observe()
                    if proof == "opener_linked":
                        opener = next(
                            item
                            for item in observed["catalog"]["candidates"]
                            if item["kind"] == "opener"
                        )
                        await broker.open(observed["revision"], opener["id"])
                        observed = await broker.observe()
                    scope = next(
                        item
                        for item in observed["catalog"]["candidates"]
                        if item["kind"] == "scope" and item["description"] == "Information"
                    )
                    await broker.bind_scope(observed["revision"], scope["id"])
                    assert broker.owned.proof == proof
                    await page.evaluate(
                        "proof => {if(proof==='chat_embed') "
                        "document.querySelector('#chat').removeAttribute('data-chat-root'); "
                        "else document.querySelector('#open').removeAttribute('aria-controls');}",
                        proof,
                    )
                    with pytest.raises(ChannelError, match="proof changed|relationship changed"):
                        await broker.observe()
                finally:
                    await broker.close()
            finally:
                await browser.close()

    asyncio.run(exercise())


def test_model_catalog_is_bounded_and_reports_truncation():
    async def exercise():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                page = await browser.new_page()
                await page.route("**/*", lambda route: route.fulfill(body="<body></body>"))
                await page.goto(URL)
                await page.set_content(
                    '<section role="dialog" aria-label="Chat">'
                    + "".join(
                        f'<article class="assistant">Reply {index}</article>'
                        for index in range(170)
                    )
                    + "</section>"
                )
                broker = NativeChatBroker(URL, page=page)
                try:
                    await broker.start()
                    observation = await broker.observe()
                    scope = next(
                        item
                        for item in observation["catalog"]["candidates"]
                        if item["kind"] == "scope"
                    )
                    await broker.bind_scope(observation["revision"], scope["id"])
                    catalog = (await broker.observe())["catalog"]
                    assert catalog["truncated"] is True
                    assert len(catalog["candidates"]) == 160
                    assert all(len(item["description"]) <= 256 for item in catalog["candidates"])
                finally:
                    await broker.close()
            finally:
                await browser.close()

    asyncio.run(exercise())


def test_page_overridden_origin_cannot_admit_a_foreign_frame():
    async def exercise():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                page = await browser.new_page()
                await page.route(
                    "**/*",
                    lambda route: route.fulfill(
                        body=(
                            '<iframe id="foreign" src="http://foreign.invalid/widget"></iframe>'
                            if route.request.url == URL
                            else '<section role="dialog" aria-label="Foreign chat">'
                            '<article class="assistant">FOREIGN_PRIVATE_CANARY</article></section>'
                        ),
                        content_type="text/html",
                    ),
                )
                await page.goto(URL)
                frame = next(frame for frame in page.frames if frame != page.main_frame)
                await frame.evaluate(
                    "url => Object.defineProperty(window, 'origin', {value:new URL(url).origin})",
                    URL,
                )
                assert await frame.evaluate("origin") == "http://localhost:8765"
                broker = NativeChatBroker(URL, page=page)
                try:
                    await broker.start()
                    assert "FOREIGN_PRIVATE_CANARY" not in json.dumps(await broker.observe())
                finally:
                    await broker.close()
            finally:
                await browser.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "attributes,kind",
    [
        ('class="assistant customer"', None),
        ('class="customer-theme" data-actor="assistant"', "assistant"),
        ('class="assistant" data-actor="assistant" data-sender="customer"', None),
    ],
)
def test_actor_catalog_requires_unambiguous_owned_evidence(attributes, kind):
    async def exercise():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                page = await browser.new_page()
                await page.route(
                    "**/*",
                    lambda route: route.fulfill(
                        body=(
                            '<section role="dialog" aria-label="Chat">'
                            f"<article {attributes}>Current reply</article></section>"
                        ),
                        content_type="text/html",
                    ),
                )
                await page.goto(URL)
                broker = NativeChatBroker(URL, page=page)
                try:
                    await broker.start()
                    observed = await broker.observe()
                    scope = next(
                        item
                        for item in observed["catalog"]["candidates"]
                        if item["kind"] == "scope"
                    )
                    await broker.bind_scope(observed["revision"], scope["id"])
                    candidates = (await broker.observe())["catalog"]["candidates"]
                    actors = [
                        item for item in candidates if item["kind"] in {"assistant", "customer"}
                    ]
                    assert [item["kind"] for item in actors] == ([] if kind is None else [kind])
                finally:
                    await broker.close()
            finally:
                await browser.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "control",
    [
        '<input role="textbox" type="password" id="composer">',
        '<input type="text" id="composer" aria-label="Login code">',
        '<input type="text" id="composer" placeholder="Credit card number">',
        '<input type="text" id="composer" name="purchaseConfirmation">',
        '<textarea id="composer" readonly></textarea>',
    ],
)
def test_forbidden_entry_fields_never_become_discovery_composers(control):
    async def exercise():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                page = await browser.new_page()
                await page.route(
                    "**/*",
                    lambda route: route.fulfill(
                        body=(
                            '<section role="dialog" aria-label="Chat">'
                            '<article class="assistant">Current reply</article>'
                            + control
                            + "</section>"
                        ),
                        content_type="text/html",
                    ),
                )
                await page.goto(URL)
                broker = NativeChatBroker(URL, page=page)
                try:
                    await broker.start()
                    observed = await broker.observe()
                    scope = next(
                        item
                        for item in observed["catalog"]["candidates"]
                        if item["kind"] == "scope"
                    )
                    await broker.bind_scope(observed["revision"], scope["id"])
                    assert not any(
                        item["kind"] == "input"
                        for item in (await broker.observe())["catalog"]["candidates"]
                    )
                    assert await page.locator("#composer").input_value() == ""
                finally:
                    await broker.close()
            finally:
                await browser.close()

    asyncio.run(exercise())

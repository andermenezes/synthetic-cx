"""Independent root proof and the controller's JSON-only native catalog seam."""

import asyncio
import json

import pytest
from playwright.async_api import async_playwright

from syntheticcx.adapters.channel.broker import NativeChatBroker
from syntheticcx.adapters.channel.ownership import attest_root
from syntheticcx.errors import ChannelError


def test_catalog_exposes_only_safe_unique_send_for_click_only_composer() -> None:
    async def exercise() -> None:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.route(
                "**/*",
                lambda route: route.fulfill(
                    status=200,
                    content_type="text/html",
                    body="""
                <section role="dialog" aria-label="Chat" id="chat">
                <button type="button" id="settings">Settings</button>
                <button type="button" id="attach">Attach</button>
                <input id="input"><button id="send" type="button">Enviar</button>
                <button type="button" class="duplicate">Send</button>
                <button type="button" class="duplicate">Send</button>
                <form><button type="button" id="form-send">Send</button></form>
                </section><script>
                window.sent=0;
                document.querySelector('#send').onclick=()=>{window.sent++;};
                </script>""",
                ),
            )
            await page.goto("http://localhost:8765/")
            broker = NativeChatBroker("http://localhost:8765/", page=page)
            await broker.start()
            snapshot = await broker.observe()
            scope = next(
                item for item in snapshot["catalog"]["candidates"] if item["kind"] == "scope"
            )
            await broker.bind_scope(scope["id"], revision=snapshot["revision"])
            snapshot = await broker.observe()
            candidates = snapshot["catalog"]["candidates"]
            sends = [item for item in candidates if item["kind"] == "send"]
            assert len(sends) == 1 and sends[0]["description"] == "Enviar"
            composer = next(item for item in candidates if item["kind"] == "input")
            await broker.bind_state(
                revision=snapshot["revision"],
                input_id=composer["id"],
                send_id=sends[0]["id"],
                send_mode="click",
            )
            await broker.submit_probe(snapshot["revision"], composer["id"], "Olá")
            assert await page.evaluate("window.sent") == 1
            await broker.close()
            await browser.close()

    asyncio.run(exercise())


def test_catalog_binding_is_owned_and_menu_capabilities_are_distinct() -> None:
    async def exercise() -> None:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.route(
                "**/*",
                lambda route: route.fulfill(
                    status=200,
                    content_type="text/html",
                    body="""
              <section id="decoy"><input id="outside"><p>Plans</p></section>
              <section id="chat" role="dialog" aria-label="Chat assistant">
              <header><button type="button">Settings</button></header>
              <article class="assistant"><p>Earlier information</p>
              <button type="button">Hours</button></article>
              <article class="assistant"><p>Choose an information topic</p>
              <button type="button" class="option">Plans</button>
              <button type="button" class="option">Coverage</button></article></section>""",
                ),
            )
            await page.goto("http://localhost:8765/")
            broker = NativeChatBroker("http://localhost:8765/", page=page)
            await broker.start()
            snapshot = await broker.observe()
            json.dumps(snapshot)
            candidates = snapshot["catalog"]["candidates"]
            scope = next(
                item
                for item in candidates
                if item["kind"] == "scope" and item["description"] == "Chat assistant"
            )
            await broker.bind_scope(scope["id"], revision=snapshot["revision"])
            snapshot = await broker.observe()
            assert not any(item["kind"] == "input" for item in snapshot["catalog"]["candidates"])
            candidates = snapshot["catalog"]["candidates"]
            assistant = next(
                item
                for item in candidates
                if item["kind"] == "assistant"
                and item["description"] == "Choose an information topic"
            )
            await broker.bind_reader(assistant["id"])
            snapshot = await broker.observe()
            candidates = snapshot["catalog"]["candidates"]
            assert snapshot["available_options"] == []
            offered = [item for item in candidates if item["kind"] == "option"]
            assert {item["description"] for item in offered} == {"Plans", "Coverage"}
            # The first such candidate is the article, not its inner paragraph.
            option = next(item for item in candidates if item["kind"] == "option")
            state = await broker.bind_state("initial", option_id=option["id"])
            assert state.kind == "menu"
            assert len((await broker.observe())["available_options"]) == 2
            initial = await broker.observe_initial()
            snapshot = await broker.observe()
            assert len(snapshot["available_options"]) == 2
            assert snapshot["available_options"][0]["id"] == initial.messages[-1].options[0].id
            assert option["id"] not in {item["id"] for item in snapshot["available_options"]}
            with pytest.raises(ChannelError):
                await broker.select_option(option["id"])
            await broker.close()
            await browser.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "root",
    [
        '<section id="chat"><input><article>Choose information</article></section>',
        '<section id="chat" role="dialog" aria-label="Payment"><article>Plans</article></section>',
        '<section id="chat" role="dialog" aria-label="Chat"></section>'
        '<section id="chat"></section>',
    ],
)
def test_new_panel_or_ambiguous_root_is_not_chat_proof(root: str) -> None:
    async def exercise() -> None:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.route(
                "**/*", lambda route: route.fulfill(status=200, content_type="text/html", body=root)
            )
            await page.goto("http://localhost:8765/")
            broker = NativeChatBroker("http://localhost:8765/", page=page)
            await broker.start()
            snapshot = await broker.observe()
            assert not any(item["kind"] == "scope" for item in snapshot["catalog"]["candidates"])
            with pytest.raises(ChannelError):
                await attest_root(page, page.main_frame, page.locator("#chat"), broker.url)
            assert broker.owned is None
            await broker.close()
            await browser.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("shadow", [False, True])
def test_scope_catalog_excludes_unrelated_ids_and_rechecks_proof(shadow: bool) -> None:
    async def exercise() -> None:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            broker = None
            try:
                page = await browser.new_page()
                await page.route("**/*", lambda route: route.fulfill(body="<body></body>"))
                await page.goto("http://localhost:8765/")
                await page.evaluate(
                    """shadow => {
                  document.body.innerHTML = '<header id="faq">FAQ</header>' +
                    '<button id="launcher" type="button">Chat</button>';
                  const markup = '<div id="unproven">Unrelated page region</div>' +
                    '<section id="chat" role="dialog" aria-label="Chat assistant">' +
                    '<input id="composer"></section>';
                  if(shadow) {
                    const host = document.createElement('widget-panel');
                    document.body.append(host); host.attachShadow({mode:'open'}).innerHTML=markup;
                  } else document.body.insertAdjacentHTML('beforeend', markup);
                }""",
                    shadow,
                )
                broker = NativeChatBroker("http://localhost:8765/", page=page)
                await broker.start()
                snapshot = await broker.observe()
                scopes = [
                    item for item in snapshot["catalog"]["candidates"] if item["kind"] == "scope"
                ]
                assert len(scopes) == 1
                assert scopes[0]["description"] == "Chat assistant"
                assert not snapshot["ownership_bound"]
                # Catalog proof is not authority to bind a root whose proof has changed.
                await page.locator("#chat").evaluate("node => node.removeAttribute('role')")
                with pytest.raises(ChannelError, match="ownership proof is missing"):
                    await broker.bind_scope(snapshot["revision"], scopes[0]["id"])
                assert broker.owned is None
                assert not any(
                    item["kind"] == "scope"
                    for item in (await broker.observe())["catalog"]["candidates"]
                )
            finally:
                if broker is not None:
                    await broker.close()
                await browser.close()

    asyncio.run(exercise())


def test_empty_owned_composer_bootstraps_unique_actual_assistant_node() -> None:
    async def exercise() -> None:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.route(
                "**/*",
                lambda route: route.fulfill(
                    status=200,
                    content_type="text/html",
                    body="""
              <p class="assistant">Unowned decoy</p>
              <section id="chat" role="dialog" aria-label="Chat assistant">
              <input id="composer"></section>
              <script>document.querySelector('#composer').addEventListener('keydown', event => {
                if (event.key === 'Enter') document.querySelector('#chat')
                  .insertAdjacentHTML('beforeend',
                  '<article class="assistant">Here are the coverage details.</article>');
              });</script>""",
                ),
            )
            await page.goto("http://localhost:8765/")
            broker = NativeChatBroker("http://localhost:8765/", page=page)
            await broker.start()
            snapshot = await broker.observe()
            scope = next(
                item
                for item in snapshot["catalog"]["candidates"]
                if item["kind"] == "scope" and item["description"] == "Chat assistant"
            )
            await broker.bind_scope(snapshot["revision"], scope["id"])
            snapshot = await broker.observe()
            composer = next(
                item for item in snapshot["catalog"]["candidates"] if item["kind"] == "input"
            )
            initial = await broker.bind_state(
                snapshot["revision"], composer["id"], None, None, "press_enter"
            )
            assert not (await broker.observe())["reader_bound"]
            await broker.submit_probe(
                snapshot["revision"], composer["id"], "Tell me about coverage"
            )
            response = await broker.receive()
            assert [message.content for message in response.messages] == [
                "Here are the coverage details."
            ]
            snapshot = await broker.observe()
            assert snapshot["reader_bound"]
            composer = snapshot["composer_id"]
            final = await broker.bind_state(
                snapshot["revision"], composer, None, None, "press_enter"
            )
            assert final == initial
            config = broker.export_config(
                [initial],
                [
                    {
                        "id": "t1",
                        "from_state": initial.id,
                        "to_state": final.id,
                        "action_kind": "probe",
                        "probe_id": "first",
                    }
                ],
                [{"transition_id": "t1"}],
            )
            assert config.initial_state_id == initial.id
            assert config.assistant_message_locator.value == "article.assistant"
            await broker.close()
            await browser.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "markup",
    [
        '<button id="opaque" type="submit">Chat</button>',
        '<button id="opaque">Chat</button>',
        '<form><button id="opaque" type="button">Chat</button></form>',
        '<button id="opaque" type="button" aria-label="Buy">Chat</button>',
        '<button id="opaque" type="button" hidden>Chat</button>',
        '<div id="opaque" role="button">Chat</div>',
        '<button id="opaque" type="button">Chat</button>'
        '<button id="opaque" type="button">Chat</button>',
    ],
)
def test_exact_id_hint_rejects_unsafe_current_native_node(markup: str) -> None:
    async def exercise() -> None:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.route(
                "**/*",
                lambda route: route.fulfill(status=200, content_type="text/html", body=markup),
            )
            await page.goto("http://localhost:8765/")
            await page.evaluate(
                "window.clicks=0; document.addEventListener('click', () => clicks++)"
            )
            broker = NativeChatBroker("http://localhost:8765/", page=page, chat_hint="id=opaque")
            with pytest.raises(ChannelError):
                await broker.start()
            assert await page.evaluate("clicks") == 0
            await broker.close()
            await browser.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("explicit_id", [False, True])
@pytest.mark.parametrize(
    ("placement", "attribute", "value"),
    [
        ("light", attribute, value)
        for attribute, value in [
            ("data-action", "purchase"),
            ("data-url", "/checkout"),
            ("data-href", "/checkout"),
            ("href", "/checkout"),
            ("target", "_blank"),
            ("download", "receipt"),
            ("formaction", "/checkout"),
            ("form", "purchase"),
            ("action", "/checkout"),
            ("src", "/checkout"),
            ("disabled", ""),
            ("inert", ""),
            ("aria-disabled", "true"),
        ]
    ]
    + [(placement, "data-action", "purchase") for placement in ("shadow", "slot")],
)
def test_opener_rejects_unsafe_ancestor_before_click(
    explicit_id: bool, placement: str, attribute: str, value: str
) -> None:
    async def exercise() -> None:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            broker = None
            try:
                page = await browser.new_page()
                await page.route("**/*", lambda route: route.fulfill(body="<body></body>"))
                await page.goto("http://localhost:8765/")
                await page.evaluate(
                    """placement => {
                  const button = '<button id="launcher" type="button">Chat</button>';
                  const host = document.createElement('launcher-widget');
                  document.body.append(host);
                  if (placement === 'light') host.innerHTML = button;
                  else {
                    const root = host.attachShadow({mode:'open'});
                    if (placement === 'shadow') root.innerHTML = button;
                    else {
                      root.innerHTML = '<div id="slot-parent"><slot></slot></div>';
                      host.innerHTML = button;
                    }
                  }
                  window.clicks = 0;
                  document.addEventListener('click', () => clicks++);
                }""",
                    placement,
                )
                broker = NativeChatBroker(
                    "http://localhost:8765/",
                    page=page,
                    chat_hint="id=launcher" if explicit_id else None,
                )
                opener = None
                if not explicit_id:
                    await broker.start()
                    snapshot = await broker.observe()
                    opener = next(
                        item
                        for item in snapshot["catalog"]["candidates"]
                        if item["kind"] == "opener"
                    )
                # Change ancestry after cataloging to exercise dispatch-time checks.
                await page.evaluate(
                    """args => {
                  const host = document.querySelector('launcher-widget');
                  const ancestor = args.placement === 'slot' ?
                    host.shadowRoot.querySelector('#slot-parent') : host;
                  ancestor.setAttribute(args.attribute, args.value);
                }""",
                    {"placement": placement, "attribute": attribute, "value": value},
                )
                with pytest.raises(ChannelError, match="safe chat control"):
                    if explicit_id:
                        await broker.start()
                    else:
                        await broker.open(snapshot["revision"], opener["id"])
                assert await page.evaluate("clicks") == 0
            finally:
                if broker is not None:
                    await broker.close()
                await browser.close()

    asyncio.run(exercise())


def test_exact_id_hint_opens_only_observed_shadow_button_once() -> None:
    async def exercise() -> None:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.route(
                "**/*",
                lambda route: route.fulfill(
                    status=200, content_type="text/html", body="<body></body>"
                ),
            )
            await page.goto("http://localhost:8765/")
            await page.evaluate("""() => {
              const host = document.createElement('launcher-widget'); document.body.append(host);
              const root = host.attachShadow({mode:'open'});
              root.innerHTML = '<button id="opaque" type="button">' +
                '<svg width="20" height="20"></svg></button>';
              window.clicks = 0;
              root.querySelector('button').addEventListener('click', () => {
                clicks++;
                const panel = document.createElement('section'); panel.id = 'chat';
                panel.setAttribute('role', 'dialog');
                panel.setAttribute('aria-label', 'Chat assistant');
                panel.innerHTML = '<input id="composer">'; root.append(panel);
              });
            }""")
            broker = NativeChatBroker("http://localhost:8765/", page=page, chat_hint="id=opaque")
            await broker.start()
            assert await page.evaluate("clicks") == 1
            snapshot = await broker.observe()
            scope = next(
                item
                for item in snapshot["catalog"]["candidates"]
                if item["kind"] == "scope" and item["description"] == "Chat assistant"
            )
            await broker.bind_scope(snapshot["revision"], scope["id"])
            assert broker.owned.proof == "chat_dialog"
            assert await page.evaluate("clicks") == 1
            await broker.close()
            await browser.close()

    asyncio.run(exercise())

"""Generic nested open-shadow discovery and calibration, without external probes."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml
from playwright.async_api import async_playwright

from syntheticcx.adapters.channel.cursor import Bubble, MessageCursor
from syntheticcx.adapters.discovery.native import (
    CalibrationObservationUnavailable,
    NativeCalibrationBrowser,
)
from syntheticcx.adapters.discovery.playwright_mcp import _CHAT_INSPECTION, _ID_HINT_INSPECTION
from syntheticcx.config.models import ChannelConfig, ExecutionLimits
from syntheticcx.errors import ChannelError


@pytest.mark.parametrize("actor", ["customer", "assistant"])
def test_invalid_message_selector_identifies_actor_without_leaking_selector(actor):
    async def exercise():
        sample = yaml.safe_load(Path("tests/fixtures/discovery-sample.yaml").read_text())
        data = sample["channel"]
        data.update(
            {
                "url": "http://127.0.0.1:8000/",
                "customer_message_locator": {"strategy": "css", "value": "p.customer"},
                "assistant_message_locator": {"strategy": "css", "value": "p.assistant"},
            }
        )
        data[f"{actor}_message_locator"] = {"strategy": "css", "value": "[PRIVATE_SELECTOR"}
        browser = NativeCalibrationBrowser(
            ChannelConfig.model_validate(data), ExecutionLimits.model_validate(sample["execution"])
        )
        async with async_playwright() as p:
            chromium = await p.chromium.launch()
            try:
                page = await chromium.new_page()
                await page.route(
                    "http://127.0.0.1:8000/",
                    lambda route: route.fulfill(
                        body='<p class="customer">Hello</p><p class="assistant">Reply</p>'
                    ),
                )
                await page.goto("http://127.0.0.1:8000/")
                browser._page = browser._scope = page
                with pytest.raises(
                    ChannelError, match=f"{actor} message locator has invalid"
                ) as error:
                    await browser.message_evidence(browser._config)
                assert "PRIVATE_SELECTOR" not in str(error.value)
            finally:
                await chromium.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "variant,safe",
    [
        ("nested", True),
        ("light_duplicate", False),
        ("sibling_duplicate", False),
        ("unsafe_label", False),
        ("safe_label", True),
        ("missing_local_label", False),
        ("closed", False),
        ("scan_limit", False),
    ],
)
def test_shadow_launcher_lookup_is_unique_bounded_and_tree_scoped(variant, safe):
    async def exercise():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            try:
                page = await browser.new_page()
                await page.set_content('<div id="host"></div><div id="second"></div>')
                await page.evaluate(
                    """variant => {
                  const first = document.getElementById('host').attachShadow({mode:
                    variant === 'closed' ? 'closed' : 'open'});
                  first.innerHTML = '<section id="nested"></section>';
                  const tree = first.querySelector('section').attachShadow({mode:'open'});
                  tree.innerHTML = '<button id="chat" type="button">Chat</button>';
                  const button = tree.querySelector('button');
                  if (variant === 'light_duplicate') {
                    const other = button.cloneNode(true); document.body.append(other);
                  }
                  if (variant === 'sibling_duplicate') {
                    document.getElementById('second').attachShadow({mode:'open'})
                      .append(button.cloneNode(true));
                  }
                  if (variant.includes('label')) {
                    button.setAttribute('aria-labelledby', 'label');
                    document.body.insertAdjacentHTML('beforeend', '<span id="label">Chat</span>');
                    if (variant !== 'missing_local_label') {
                      const label=document.createElement('span'); label.id='label';
                      label.textContent=variant === 'unsafe_label' ? 'Buy' : 'Chat';
                      tree.append(label);
                    }
                  }
                  if (variant === 'scan_limit') for (let i=0;i<20001;i++)
                    document.body.append(document.createElement('i'));
                }""",
                    variant,
                )
                result = await page.evaluate(
                    _ID_HINT_INSPECTION.replace("__SELECTOR__", json.dumps('[id="chat"]'))
                )
                assert result["safe"] is safe
                if variant == "scan_limit":
                    assert not result["complete"]
                elif variant != "closed":
                    assert result["complete"]
            finally:
                await browser.close()

    asyncio.run(exercise())


def test_shadow_inventory_prioritizes_dialog_editor_and_omits_values():
    async def exercise():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            try:
                page = await browser.new_page()
                await page.set_content('<input id="page-search"><div id="host"></div>')
                await page.evaluate("""() => {
                  const root=document.getElementById('host').attachShadow({mode:'open'});
                  root.innerHTML='<div role="dialog"><textarea id="composer" ' +
                    'placeholder="Message">private-value</textarea>' +
                    '<button id="send" type="button">Send</button></div>';
                }""")
                result = await page.evaluate(_CHAT_INSPECTION)
                assert result["complete"] and result["editor_ready"]
                assert result["elements"][0]["attributes"]["id"] == "composer"
                assert result["elements"][0]["in_dialog"]
                assert "private-value" not in json.dumps(result)
                assert any(el["attributes"].get("id") == "send" for el in result["elements"])
            finally:
                await browser.close()

    asyncio.run(exercise())


_SHADOW_CHAT = """<!doctype html>
<button id="open" type="button">Open chat</button><div id="outer"></div>
<script>
const outer=document.getElementById('outer').attachShadow({mode:'open'});
outer.innerHTML='<section id="inner"></section>';
const root=outer.getElementById('inner').attachShadow({mode:'open'});
root.innerHTML='<div id="conversation" role="dialog" hidden><div id="messages"></div>' +
  '<textarea id="composer" placeholder="Message"></textarea>' +
  '<button id="send" type="button">Send</button></div>';
document.getElementById('open').onclick=()=>root.getElementById('conversation').hidden=false;
window.submissions=0;
root.getElementById('send').onclick=()=>{
  window.submissions++;
  const message=root.getElementById('composer').value;
  const customer=document.createElement('p'); customer.dataset.actor='customer';
  customer.dataset.messageId='customer-1'; customer.textContent=message;
  const assistant=document.createElement('p'); assistant.dataset.actor='assistant';
  assistant.dataset.messageId='assistant-1'; assistant.textContent='Hello from nested shadows';
  root.getElementById('messages').append(customer,assistant);
  root.getElementById('composer').remove();
};
</script>"""


@pytest.mark.parametrize("node_identity", [False, True])
@pytest.mark.parametrize("anonymous_scope", [False, True])
def test_native_calibration_retains_nested_scope_after_composer_removal(
    node_identity, anonymous_scope
):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            document = _SHADOW_CHAT
            if anonymous_scope:
                document = document.replace(
                    "window.submissions=0;",
                    "document.getElementById('outer').removeAttribute('id');window.submissions=0;",
                )
            self.wfile.write(document.encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    async def exercise():
        sample = yaml.safe_load(Path("tests/fixtures/discovery-sample.yaml").read_text())
        data = sample["channel"]
        data.update(
            {
                "url": f"http://127.0.0.1:{server.server_port}/",
                "open_locator": {"strategy": "css", "value": "#open"},
                "input_locator": {"strategy": "placeholder", "value": "Message"},
                "send_strategy": {
                    "type": "click",
                    "locator": {"strategy": "css", "value": "#send"},
                },
                "customer_message_locator": {"strategy": "css", "value": '[data-actor="customer"]'},
                "assistant_message_locator": {
                    "strategy": "css",
                    "value": '[data-actor="assistant"]',
                },
                "message_identity": {"strategy": "attribute", "attribute_name": "data-message-id"},
            }
        )
        config = ChannelConfig.model_validate(data)
        if node_identity:
            data["message_identity"] = {"strategy": "dom_node"}
            config = ChannelConfig.model_validate(data)
        browser = NativeCalibrationBrowser(
            config, ExecutionLimits.model_validate(sample["execution"])
        )
        try:
            await browser.start()
            baseline = await browser.inventory()
            assert not baseline["entries"]
            await browser.send("Olá")
            observed = await browser.inventory()
            assert any(
                entry["text"] == "Hello from nested shadows" for entry in observed["entries"]
            )
            customers, assistants = await browser.message_evidence(config)
            assert (
                customers[0]["text"] == "Olá"
                and assistants[0]["text"] == "Hello from nested shadows"
            )
            assert await browser._page.evaluate("window.submissions") == 1
            await browser._page.evaluate("""() => {
              window.savedHost=document.body.querySelector('div'); window.savedHost.remove();
            }""")
            with pytest.raises(CalibrationObservationUnavailable):
                await browser.inventory()
            await browser._page.evaluate("document.body.append(window.savedHost)")
            if anonymous_scope:
                with pytest.raises(CalibrationObservationUnavailable):
                    await browser.inventory()
                await browser._page.evaluate(
                    "document.body.insertBefore(window.savedHost, document.querySelector('script'))"
                )
            restored = await browser.inventory()
            assert any(
                entry["text"] == "Hello from nested shadows" for entry in restored["entries"]
            )
            if anonymous_scope:
                await browser._page.evaluate("""() => {
                  const old=document.body.querySelector('div'); window.originalHost=old;
                  const other=document.createElement('div'); old.replaceWith(other);
                  const tree=other.attachShadow({mode:'open'});
                  tree.innerHTML='<section id="inner"></section>';
                  tree.getElementById('inner').attachShadow({mode:'open'}).innerHTML=
                    '<div id="conversation" role="dialog"><p>PRIVATE_OTHER_WIDGET</p></div>';
                }""")
                with pytest.raises(CalibrationObservationUnavailable):
                    await browser.inventory()
                await browser._page.evaluate(
                    "document.body.querySelector('div').replaceWith(window.originalHost)"
                )
                await browser.inventory()
            if node_identity:
                await browser._page.evaluate("""() => {
                  const root=document.body.querySelector('div').shadowRoot
                    .getElementById('inner').shadowRoot;
                  const bubble=root.querySelector('[data-actor=assistant]');
                  bubble.replaceWith(bubble.cloneNode(true));
                }""")
                _, replaced = await browser.message_evidence(config)
                assert replaced[0]["identity"] != assistants[0]["identity"]
                cursor = MessageCursor(config.message_identity)
                old = Bubble("assistant", 0, assistants[0]["text"], assistants[0]["identity"])
                cursor.observe("assistant", [old], baseline=True)
                new = Bubble("assistant", 0, replaced[0]["text"], replaced[0]["identity"])
                with pytest.raises(ChannelError, match="observed message disappeared"):
                    cursor.observe("assistant", [new])
        finally:
            await browser.close()

    try:
        asyncio.run(exercise())
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(
    "case,expected", [("private", "Hello"), ("slot", "Before Actual reply After")]
)
def test_message_text_is_rendered_safe_and_slot_aware(case, expected):
    from syntheticcx.adapters.discovery.native import _MESSAGE_EVIDENCE

    async def exercise():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            try:
                page = await browser.new_page()
                await page.set_content('<div id="message">Hello</div>')
                if case == "private":
                    await page.locator("#message").evaluate("""node => {
                      node.innerHTML+='<!--not a message--><script type="application/json">' +
                        '{"token":"PRIVATE_SECRET"}</script><span hidden>PRIVATE_HIDDEN</span>' +
                        '<span style="visibility:hidden">PRIVATE_INVISIBLE</span>';
                    }""")
                else:
                    await page.locator("#message").evaluate("""node => {
                      node.innerHTML='<section><span>Actual reply</span></section>';
                      node.firstElementChild.attachShadow({mode:'open'}).innerHTML=
                        'Before <slot>Fallback reply</slot> After';
                    }""")
                result = await page.locator("#message").evaluate_all(
                    _MESSAGE_EVIDENCE, {"url": page.url, "attribute": "id", "identities": []}
                )
                assert result["entries"][0]["text"] == expected
                assert "PRIVATE" not in json.dumps(result)
            finally:
                await browser.close()

    asyncio.run(exercise())


def test_inventory_when_dialog_is_shadow_host():
    from syntheticcx.adapters.discovery.native import _INVENTORY

    async def exercise():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            try:
                page = await browser.new_page()
                await page.set_content('<div id="dialog" role="dialog"></div>')
                await page.locator("#dialog").evaluate("""node => {
                  node.attachShadow({mode:'open'}).innerHTML=
                    '<textarea id="composer"></textarea><p>Hello</p>';
                }""")
                registry = await page.evaluate_handle(
                    "() => ({document, nodes: new WeakMap(), next: 0, prefix: 'test-generation'})"
                )
                result = await page.locator("#composer").evaluate_all(
                    _INVENTORY, {"url": page.url, "registry": registry}
                )
                await registry.dispose()
                assert result.get("status") != "missing_scope"
                assert result["entries"][0]["text"] == "Hello"
            finally:
                await browser.close()

    asyncio.run(exercise())


def test_shadow_inventory_associates_same_origin_iframe_controls():
    async def exercise():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            try:
                page = await browser.new_page()
                await page.set_content('<div id="host"></div>')
                await page.evaluate("""() => {
                  const root=document.getElementById('host').attachShadow({mode:'open'});
                  const frame=document.createElement('iframe'); frame.id='chat-frame';
                  frame.srcdoc='<textarea id="composer" placeholder="Message"></textarea>';
                  root.append(frame);
                }""")
                await (
                    page.frame_locator("#chat-frame").locator("#composer").wait_for(state="visible")
                )
                inventory = await page.evaluate(_CHAT_INSPECTION)
                frame = inventory["frames"][0]
                assert frame["frame"]["attributes"]["id"] == "chat-frame"
                assert frame["complete"] and frame["editor_ready"]
                assert frame["elements"][0]["attributes"]["id"] == "composer"
            finally:
                await browser.close()

    asyncio.run(exercise())

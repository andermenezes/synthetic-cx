"""Typed calibration of nested-shadow menus is observation-only and submits once."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from syntheticcx.adapters.channel.playwright import PlaywrightChannel
from syntheticcx.adapters.discovery.native import NativeCalibrationBrowser
from syntheticcx.config.models import ChannelConfig, ExecutionLimits, PrivacyConfig
from syntheticcx.discovery import calibrate, verify
from syntheticcx.discovery.catalog import css_string, node_locators
from syntheticcx.discovery.explore import DiscoveryCandidate
from syntheticcx.discovery.proposals import READING_SCHEMA
from syntheticcx.privacy import Redactor

DOCUMENT = """<!doctype html><div id="outer"></div><script>
const outer=document.getElementById('outer').attachShadow({mode:'open'});
outer.innerHTML='<section id="inner"></section>';
const root=outer.getElementById('inner').attachShadow({mode:'open'});
root.innerHTML='<div id="conversation" role="dialog"><div id="transcript">' +
  '<article data-actor="assistant"><p>Hello</p></article></div>' +
  '<textarea placeholder="Message"></textarea><button id="send" type="button">Send</button></div>';
window.submissions=0;window.optionClicks=0;
root.getElementById('send').onclick=()=>{
  window.submissions++;
  const customer=document.createElement('article');customer.dataset.actor='customer';
  customer.textContent=root.querySelector('textarea').value;
  const assistant=document.createElement('article');assistant.dataset.actor='assistant';
  assistant.className='flex:row w-[40px]';
  assistant.innerHTML=__PROSE__ + '<div role="group" aria-label="Information menu">' +
    '<button type="button" data-testid="reply-option">Plans</button>' +
    '<button type="button" data-testid="reply-option">Coverage</button></div>';
  assistant.querySelectorAll('button').forEach(button=>button.onclick=()=>{
    window.optionClicks++;
    assistant.querySelectorAll('button').forEach(node=>node.remove());
    const followup=document.createElement('article');followup.dataset.actor='assistant';
    followup.textContent='Plan information';
    root.getElementById('transcript').append(followup);
  });
  root.getElementById('transcript').append(customer,assistant);
};
</script>"""


@pytest.mark.parametrize("prose", [False, True])
@pytest.mark.parametrize("remove_composer", [False, True])
@pytest.mark.parametrize("echo", [False, True])
def test_calibration_reads_options_and_verification_selects_one_guarded_option(
    monkeypatch, prose, remove_composer, echo
):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            document = DOCUMENT.replace(
                "__PROSE__",
                json.dumps("<p>Hello. Choose plan information</p>" if prose else ""),
            )
            if remove_composer:
                document = document.replace(
                    "root.getElementById('transcript').append(customer,assistant);",
                    "root.getElementById('transcript').append(customer,assistant);"
                    "root.querySelector('textarea').remove();",
                )
            if not echo:
                document = document.replace("append(customer,assistant)", "append(assistant)")
            self.wfile.write(document.encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    counts = []
    verification_counts = []
    requests = []
    progress = []

    class Browser(NativeCalibrationBrowser):
        async def close(self):
            if self._page is not None:
                counts.append(
                    await self._page.evaluate("() => [window.submissions, window.optionClicks]")
                )
            await super().close()

    class Provider:
        async def generate(self, request):
            requests.append(request)
            evidence = json.loads(request.messages[-1].content)
            catalog = evidence["catalog"]

            def pick(kind, fragment=""):
                return next(
                    item["id"]
                    for item in catalog["candidates"]
                    if item["kind"] == kind and fragment in item["description"]
                )

            return type(
                "Response",
                (),
                {
                    "content": json.dumps(
                        {
                            "revision": catalog["revision"],
                            "customer_id": pick(
                                "customer", "Observed article" if echo else "empty selection"
                            ),
                            "assistant_id": pick("assistant", "Observed article"),
                            "identity_id": pick("identity"),
                            "completion_id": pick("completion"),
                            "scope_id": pick("scope"),
                            "option_id": pick("option"),
                        }
                    )
                },
            )()

    class VerificationBrowser(PlaywrightChannel):
        async def close(self):
            if self._page is not None:
                verification_counts.append(
                    await self._page.evaluate("() => [window.submissions, window.optionClicks]")
                )
            await super().close()

    async def exercise():
        sample = yaml.safe_load(Path("tests/fixtures/discovery-sample.yaml").read_text())
        data = sample["channel"]
        data.update(
            url=f"http://127.0.0.1:{server.server_port}/",
            open_locator=None,
            input_locator={"strategy": "placeholder", "value": "Message"},
            send_strategy={"type": "click", "locator": {"strategy": "css", "value": "#send"}},
        )
        candidate = DiscoveryCandidate(channel=ChannelConfig.model_validate(data), open_steps=0)
        limits = ExecutionLimits.model_validate(sample["execution"]).model_copy(
            update={"greeting_grace_seconds": 0}
        )
        result = await calibrate.calibrate_candidate(
            candidate,
            limits,
            "Hello",
            Provider(),
            Redactor(PrivacyConfig()),
            max_attempts=1,
            on_progress=progress.append,
        )
        assert result.channel.reply_options is not None
        assert result.channel.message_identity.strategy == "dom_node"
        if not echo:
            assert result.channel.customer_message_locator.value == ":not(*)"
        guide = await verify.verify_candidate(result, limits, "Hello")
        assert guide.validation.status == "verified"
        assert "select_option" in guide.validation.observed_behaviors

    monkeypatch.setattr(calibrate, "NativeCalibrationBrowser", Browser)
    monkeypatch.setattr(verify, "PlaywrightChannel", VerificationBrowser)
    try:
        asyncio.run(exercise())
    finally:
        server.shutdown()
        server.server_close()
    assert counts == [[1, 0]]
    assert verification_counts == [[1, 1]]
    assert any("deferred to fresh verification" in line for line in progress) == remove_composer
    assert len(requests) == 1 and requests[0].response_schema == READING_SCHEMA
    assert "channel_json" not in requests[0].messages[0].content.replace("or channel_json", "")


def test_owned_special_attribute_candidates_match_real_chromium():
    from playwright.async_api import async_playwright

    async def exercise():
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            try:
                page = await browser.new_page()
                await page.set_content("<div></div>")
                attributes = {"class": "flex:row w-[40px]", "aria-label": 'quote" slash\\\nline'}
                await page.locator("div").evaluate(
                    "(node, attrs) => {for (const [key,value] of Object.entries(attrs)) "
                    "node.setAttribute(key,value)}",
                    attributes,
                )
                for locator in node_locators({"tag": "div", "attributes": attributes}):
                    if locator.strategy == "css":
                        assert await page.locator(locator.value).count() == 1
                assert (
                    await page.locator(
                        f"[aria-label={css_string(attributes['aria-label'])}]"
                    ).count()
                    == 1
                )
            finally:
                await browser.close()

    asyncio.run(exercise())

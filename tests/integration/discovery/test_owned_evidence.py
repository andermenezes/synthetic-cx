"""Calibration registries and selection ownership are private, read-only boundaries."""

import asyncio
from pathlib import Path

import pytest
import yaml
from playwright.async_api import async_playwright

from syntheticcx.adapters.discovery.native import (
    _MESSAGE_EVIDENCE,
    CalibrationObservationUnavailable,
    NativeCalibrationBrowser,
)
from syntheticcx.config.models import ChannelConfig, ExecutionLimits
from syntheticcx.discovery.calibrate import _reading_catalog
from syntheticcx.errors import ChannelError


def config():
    sample = yaml.safe_load(Path("tests/fixtures/discovery-sample.yaml").read_text())
    return ChannelConfig.model_validate(
        {
            **sample["channel"],
            "url": "http://127.0.0.1:8000/",
            "open_locator": None,
            "input_locator": {"strategy": "css", "value": "#composer"},
            "customer_message_locator": {"strategy": "css", "value": '[data-actor="customer"]'},
            "assistant_message_locator": {"strategy": "css", "value": '[data-actor="assistant"]'},
            "message_identity": {"strategy": "dom_node"},
        }
    ), ExecutionLimits.model_validate(sample["execution"])


@pytest.mark.parametrize("scope_kind", ["dialog", "shadow_dialog", "shadow_root"])
def test_private_node_registry_cannot_be_overwritten_and_scope_rejects_faq(scope_kind):
    async def exercise():
        channel, limits = config()
        browser = NativeCalibrationBrowser(channel, limits)
        async with async_playwright() as playwright:
            chromium = await playwright.chromium.launch()
            try:
                page = await chromium.new_page()
                await page.route(
                    str(channel.url), lambda route: route.fulfill(body="<body></body>")
                )
                await page.goto(str(channel.url))
                browser._page = browser._scope = page
                await page.evaluate(
                    """kind => {
                  document.body.innerHTML='<p id="faq" data-secret="PRIVATE_FAQ">PRIVATE_FAQ</p>' +
                    '<div id="host"></div>';
                  const root=kind !== 'dialog' ?
                    document.getElementById('host').attachShadow({mode:'open'}) :
                    document.getElementById('host');
                  root.innerHTML='<div id="chat" ' +
                    (kind === 'shadow_root' ? '' : 'role="dialog"') +
                    '><textarea id="composer"></textarea>' +
                    '<p data-actor="assistant">Repeated greeting</p></div>';
                }""",
                    scope_kind,
                )
                baseline = await browser.inventory()
                old = next(
                    entry["node_id"]
                    for entry in baseline["entries"]
                    if entry["text"] == "Repeated greeting"
                )
                assert not await page.evaluate(
                    "() => Object.hasOwn(document, Symbol.for('syntheticcx.discovery.nodes'))"
                )
                await page.evaluate("""() => {
                  document[Symbol.for('syntheticcx.discovery.nodes')] = {
                    nodes: {has: () => true, get: () => 'FORGED_ID'}, next: 77
                  };
                  const host=document.getElementById('host');
                  const root=host.shadowRoot || host;
                  const node=root.querySelector('p');
                  node.parentElement.append(node.cloneNode(true));
                }""")
                after = await browser.inventory()
                identities = [
                    entry["node_id"]
                    for entry in after["entries"]
                    if entry["text"] == "Repeated greeting"
                ]
                assert old in identities and len(set(identities)) == 2
                assert "FORGED_ID" not in identities
                _, evidence = await browser.message_evidence(channel)
                assert {entry["node_id"] for entry in evidence} == set(identities)
                assert (
                    await page.evaluate(
                        "() => document[Symbol.for('syntheticcx.discovery.nodes')].next"
                    )
                    == 77
                )

                external = channel.model_copy(
                    update={
                        "assistant_message_locator": channel.assistant_message_locator.model_copy(
                            update={"value": "#faq"}
                        )
                    }
                )
                with pytest.raises(ChannelError, match="owned observation scope") as error:
                    await browser.message_evidence(external)
                assert "PRIVATE_FAQ" not in str(error.value)
                # The exporting helper itself returns only an owned value-free status.
                rejected = await page.locator("#faq").evaluate_all(
                    _MESSAGE_EVIDENCE,
                    {
                        "url": str(channel.url),
                        "hint": browser._observation_hint,
                        "scope_nodes": browser._scope_nodes,
                        "registry": browser._node_registry,
                    },
                )
                assert rejected["status"] == "outside_scope" and rejected["entries"] == []
                assert "PRIVATE_FAQ" not in str(rejected)
            finally:
                await browser.close()
                assert browser._node_registry is None
                await chromium.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("anonymous", [False, True])
def test_document_reload_does_not_alias_node_ids_or_guess_anonymous_scope(anonymous):
    async def exercise():
        channel, limits = config()
        browser = NativeCalibrationBrowser(channel, limits)
        document = (
            '<div role="dialog" __ID__><textarea id="composer"></textarea><p>Greeting</p></div>'
        )
        document = document.replace("__ID__", "" if anonymous else 'id="chat"')
        async with async_playwright() as playwright:
            chromium = await playwright.chromium.launch()
            try:
                page = await chromium.new_page()
                await page.route(str(channel.url), lambda route: route.fulfill(body=document))
                await page.goto(str(channel.url))
                browser._page = browser._scope = page
                old = await browser.inventory()
                await page.reload()
                with pytest.raises(CalibrationObservationUnavailable):
                    await browser.inventory()
                if anonymous:
                    with pytest.raises(CalibrationObservationUnavailable, match="anonymous"):
                        await browser.inventory()
                else:
                    fresh = await browser.inventory()
                    assert not set(old["node_ids"]) & set(fresh["node_ids"])
            finally:
                await browser.close()
                await chromium.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("shadow", [False, True])
@pytest.mark.parametrize("decorated", [False, True])
def test_reader_catalog_excludes_shared_actor_classes_and_transcript_ancestors(shadow, decorated):
    async def exercise():
        channel, limits = config()
        browser = NativeCalibrationBrowser(channel, limits)
        async with async_playwright() as playwright:
            chromium = await playwright.chromium.launch()
            try:
                page = await chromium.new_page()
                await page.route(
                    str(channel.url), lambda route: route.fulfill(body='<div id="host"></div>')
                )
                await page.goto(str(channel.url))
                browser._page = browser._scope = page
                await page.evaluate(
                    """shadow => {
                  const host = document.getElementById('host');
                  const root = shadow ? host.attachShadow({mode:'open'}) : host;
                  root.innerHTML = '<div id="chat" role="dialog">' +
                    '<div id="transcript"><div class="bubble flex rounded">' +
                    '<p class="greeting">Greeting</p></div></div>' +
                    '<textarea id="composer"></textarea></div>';
                }""",
                    shadow,
                )
                baseline = await browser.inventory()
                await page.locator("#transcript").evaluate(
                    """(node, decorated) => {
                  node.insertAdjacentHTML('beforeend',
                    '<div class="bubble flex rounded right"><p class="message-text">Hello</p>' +
                    (decorated ? '<span>12:34</span>' : '') + '</div>' +
                    '<div class="bubble flex rounded left"><p class="message-text">' +
                    'Plan information</p></div>');
                }""",
                    decorated,
                )
                after = await browser.inventory()
                catalog = await _reading_catalog(browser, baseline, after, "Hello")
                assistant_locators = [
                    item.value for item in catalog.items.values() if item.kind == "assistant"
                ]
                customer_locators = [
                    item.value for item in catalog.items.values() if item.kind == "customer"
                ]
                assert assistant_locators and customer_locators
                assert any("left" in locator.value for locator in assistant_locators)
                for locator in assistant_locators:
                    nodes = await page.locator(locator.value).element_handles()
                    assert nodes
                    for node in nodes:
                        assert not await node.evaluate("node => node.textContent.includes('Hello')")
                assert not any(
                    locator.value
                    in {
                        "div",
                        "p",
                        'div[id="transcript"]',
                        'div[class~="bubble"]',
                        'div[class~="flex"]',
                    }
                    for locator in assistant_locators
                )
            finally:
                await browser.close()
                await chromium.close()

    asyncio.run(exercise())

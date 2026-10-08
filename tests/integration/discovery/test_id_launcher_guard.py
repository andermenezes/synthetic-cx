"""Read-only launcher checks against real Chromium, with no external target."""

import asyncio
import json

import pytest
from playwright.async_api import async_playwright

from syntheticcx.adapters.discovery.playwright_mcp import _ID_HINT_INSPECTION


@pytest.mark.parametrize(
    "markup,safe",
    [
        ('<button id="chat" type="button"><svg width="24" height="24"></svg></button>', True),
        ('<button id="chat" type="submit">Chat</button>', False),
        ('<button id="chat">Chat</button>', False),
        ('<form><button id="chat" type="button">Chat</button></form>', False),
        ('<button id="chat" type="button" disabled>Chat</button>', False),
        ('<button id="chat" type="button" hidden>Chat</button>', False),
        ('<button id="chat" type="button" aria-expanded="true">Chat</button>', False),
        ('<button id="chat" type="button">Pagar</button>', False),
        ('<button id="chat" type="button" aria-label="Buy">Icon</button>', False),
        (
            '<span id="label">Buy</span>'
            '<button id="chat" type="button" aria-labelledby="label"></button>',
            False,
        ),
        (
            '<span id="label">Chat</span>'
            '<button id="chat" type="button" aria-labelledby="label"></button>',
            True,
        ),
        ('<button id="chat" type="button" aria-labelledby="missing">Chat</button>', False),
        ('<a id="chat" href="/pay">Chat</a>', False),
        ('<div id="chat" role="button">Chat</div>', False),
        (
            '<button id="chat" type="button">Chat</button>'
            '<button id="chat" type="button">Chat</button>',
            False,
        ),
        ('<button type="button">Chat</button>', False),
    ],
)
def test_id_hint_dom_guard(markup, safe):
    async def exercise():
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            try:
                page = await browser.new_page()
                await page.set_content(markup)
                code = _ID_HINT_INSPECTION.replace("__SELECTOR__", json.dumps('[id="chat"]'))
                result = await page.evaluate(code)
                assert result["safe"] is safe
                assert set(result) == {"origin", "safe", "count", "complete"}
            finally:
                await browser.close()

    asyncio.run(exercise())

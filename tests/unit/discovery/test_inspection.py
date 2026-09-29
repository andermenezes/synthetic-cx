"""Fixed DOM inventory stays bounded and origin-checked."""

from __future__ import annotations

import asyncio
import json

import pytest

from syntheticcx.adapters.discovery.playwright_mcp import (
    _CHAT_INSPECTION,
    ManagedPlaywrightMCP,
)
from syntheticcx.errors import ChannelError


class InventoryBrowser(ManagedPlaywrightMCP):
    def __init__(self, result: dict[str, object]) -> None:
        super().__init__("https://example.com/")
        self.result = result
        self.snapshots = 0

    async def snapshot(self) -> str:
        self.snapshots += 1
        return "### Page\n- Page URL: https://example.com/"

    async def _call(
        self, name: str, arguments: dict[str, str], *, check_origin: bool = True
    ) -> str:
        assert name == "browser_evaluate"
        assert arguments == {"function": _CHAT_INSPECTION}
        assert not check_origin
        return "### Result\n" + json.dumps(self.result) + "\n### Ran Playwright code"


def test_read_only_inventory_checks_origin_before_and_after_evaluation() -> None:
    browser = InventoryBrowser(
        {"origin": "https://example.com", "elements": [{"tag": "textarea"}], "frames": []}
    )
    assert "textarea" in asyncio.run(browser.inspect_chat())
    assert browser.snapshots == 2
    assert "textContent" not in _CHAT_INSPECTION
    assert "node.value" not in _CHAT_INSPECTION


def test_cross_origin_inspection_is_rejected() -> None:
    browser = InventoryBrowser({"origin": "https://other.example", "elements": [], "frames": []})
    with pytest.raises(ChannelError, match="selected target origin"):
        asyncio.run(browser.inspect_chat())
    assert browser.snapshots == 1

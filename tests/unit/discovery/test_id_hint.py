"""Exact ID hints stay separate from model-generated click targets/code."""

import asyncio
import json

import pytest

from syntheticcx.adapters.discovery.playwright_mcp import ManagedPlaywrightMCP
from syntheticcx.discovery.hints import launcher_id
from syntheticcx.errors import ChannelError, ConfigurationError


@pytest.mark.parametrize(
    "hint,expected",
    [
        (None, None),
        ("Open chat", None),
        ("id=ovc-btn-chat-widget", "ovc-btn-chat-widget"),
        ("id=chat:launcher", "chat:launcher"),
    ],
)
def test_hint_syntax(hint, expected):
    assert launcher_id(hint) == expected


@pytest.mark.parametrize(
    "hint", ["id=", 'id=x";alert(1)', "id=#chat", "id=a b", "id=" + "a" * 129, "id=a\n"]
)
def test_invalid_id_hint_is_rejected(hint):
    with pytest.raises(ConfigurationError):
        launcher_id(hint)


class IdBrowser(ManagedPlaywrightMCP):
    def __init__(self, *, result=None):
        super().__init__("https://example.com/faq", chat_hint="id=ovc-btn-chat-widget")
        self._snapshot = "### Page\n- Page URL: https://example.com/faq\n- button [ref=e1]"
        self.calls = []
        self.result = result or {
            "origin": "https://example.com",
            "safe": True,
            "count": 1,
            "complete": True,
        }

    async def _call(self, name, arguments, *, check_origin=True):
        self.calls.append((name, arguments))
        if name == "browser_evaluate":
            assert not check_origin
            return "### Result\n" + json.dumps(self.result) + "\n### Ran Playwright code"
        assert name == "browser_click" and check_origin
        return self._snapshot

    async def snapshot(self):
        self._snapshot += '\n- textbox "Message" [ref=e2]'
        return self._snapshot


def test_id_opener_uses_owned_code_and_returns_exact_native_locator():
    browser = IdBrowser()
    snapshot, locator = asyncio.run(browser.open_id_hint(browser._snapshot))
    assert "textbox" in snapshot
    assert locator.strategy == "css" and locator.value == '[id="ovc-btn-chat-widget"]'
    assert len(browser.calls) == 2
    code = browser.calls[0][1]["function"]
    assert "nodes.length === 1" in code
    assert "getClientRects" in code and ":disabled" in code and "location.origin" in code
    assert "force:" not in code and "dispatchEvent" not in code
    assert "__SELECTOR__" not in code
    target = browser.calls[1][1]["target"]
    assert '[id="ovc-btn-chat-widget"]' in target and 'button[type="button"]' in target
    assert ":not(form button)" in target


@pytest.mark.parametrize(
    "result",
    [
        {"origin": "https://other.example", "safe": True},
        {"origin": "https://example.com", "safe": False},
    ],
)
def test_id_opener_rejects_invalid_acknowledgement(result):
    browser = IdBrowser(result=result)
    with pytest.raises(ChannelError):
        asyncio.run(browser.open_id_hint(browser._snapshot))
    assert len(browser.calls) == 1


def test_id_opener_rejects_stale_snapshot_without_browser_action():
    browser = IdBrowser()
    with pytest.raises(ChannelError, match="stale"):
        asyncio.run(browser.open_id_hint("stale"))
    assert browser.calls == []


def test_id_opener_waits_only_for_missing_control_then_clicks_once(monkeypatch):
    browser = IdBrowser()
    browser.result = {"origin": "https://example.com", "safe": False, "count": 0, "complete": True}
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)
        browser.result = {
            "origin": "https://example.com",
            "safe": True,
            "count": 1,
            "complete": True,
        }

    monkeypatch.setattr(asyncio, "sleep", sleep)
    asyncio.run(browser.open_id_hint(browser._snapshot))
    assert sleeps == [1]
    assert [name for name, _ in browser.calls] == [
        "browser_evaluate",
        "browser_evaluate",
        "browser_click",
    ]


def test_id_opener_rejects_wrong_initial_origin_without_inspection():
    browser = IdBrowser()
    browser._snapshot = "### Page\n- Page URL: https://other.example/"
    with pytest.raises(ChannelError, match="selected target origin"):
        asyncio.run(browser.open_id_hint(browser._snapshot))
    assert browser.calls == []

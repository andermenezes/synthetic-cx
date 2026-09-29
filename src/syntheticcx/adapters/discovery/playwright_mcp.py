"""Managed, allowlisted Playwright MCP client for short discovery sessions."""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
from contextlib import AsyncExitStack
from typing import Protocol, cast
from urllib.parse import urlsplit

from pydantic import AnyHttpUrl, TypeAdapter, ValidationError

from syntheticcx.errors import ChannelError, ConfigurationError

SERVER_VERSION = "0.0.82"
ALLOWED_TOOLS = frozenset(
    {"browser_navigate", "browser_snapshot", "browser_hover", "browser_click", "browser_evaluate"}
)
_PAGE_URL = re.compile(r"^- Page URL: (\S+)$", re.MULTILINE)
_CHAT_WORDS = re.compile(r"\b(chat|support|help|contact|assistant|message)\b", re.I)
_UNSAFE_WORDS = re.compile(
    r"\b(pay|purchase|buy|order|submit|delete|confirm|login|log in|sign in)\b", re.I
)
_LABELED_CONTROL = re.compile(
    r'^\s*-\s*(button|link)\s+"([^"\n]{1,256})"[^\n]*\[ref=(e\d+)\]', re.I
)


class DiscoveryBrowser(Protocol):
    async def navigate(self, url: str) -> str: ...
    async def snapshot(self) -> str: ...
    async def hover(self, target: str, element: str, snapshot: str) -> str: ...
    async def click(self, target: str, element: str, snapshot: str) -> str: ...
    async def inspect_chat(self) -> str: ...


# Fixed, read-only DOM inventory. Never pass page/model text into executable code.
# Omit input values, text content, links, and arbitrary attributes.
_CHAT_INSPECTION = """() => {
  const fields = ['role', 'aria-label', 'placeholder', 'data-testid',
                  'data-message-id', 'data-actor', 'data-role'];
  const inventory = (doc) => Array.from(doc.querySelectorAll(
    'iframe, textarea, input:not([type=password]), [contenteditable=true], button, ' +
    '[role=textbox], [role=button], [data-message-id], [data-actor], [data-role]'
  )).slice(0, 50).map(node => ({
    tag: node.tagName.toLowerCase(),
    attributes: Object.fromEntries(fields.filter(key => node.hasAttribute(key))
      .map(key => [key, (node.getAttribute(key) || '').slice(0, 80)]))
  }));
  const frames = Array.from(document.querySelectorAll('iframe')).slice(0, 3)
    .map(frame => { try { return inventory(frame.contentDocument); } catch { return []; } });
  return { origin: location.origin, elements: inventory(document), frames };
}"""


def _origin(url: str) -> tuple[str, str, int | None]:
    try:
        parsed = TypeAdapter(AnyHttpUrl).validate_python(url)
    except ValidationError:
        raise ConfigurationError("discover URL", "target must be HTTP or HTTPS") from None
    parts = urlsplit(str(parsed))
    if parts.username is not None or parts.password is not None:
        raise ConfigurationError("discover URL", "target URL cannot contain credentials")
    return (parts.scheme, parts.hostname or "", parts.port)


def _check_page_origin(text: str, expected: tuple[str, str, int | None]) -> None:
    match = _PAGE_URL.search(text)
    try:
        actual = _origin(match.group(1)) if match is not None else None
    except ConfigurationError:
        actual = None
    if actual != expected:
        raise ChannelError("discover navigation", "browser left the selected target origin")


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _hint_matches(text: str, chat_hint: str | None) -> bool:
    return bool(chat_hint and _normalized(chat_hint) in _normalized(text))


def _check_hover(snapshot: str, target: str, element: str, chat_hint: str | None) -> None:
    """Only hover a current, non-sensitive button or link that looks chat-related."""
    matching_lines = [line for line in snapshot.splitlines() if f"[ref={target}]" in line]
    if (
        not re.fullmatch(r"e\d+", target)
        or len(matching_lines) != 1
        or _UNSAFE_WORDS.search(matching_lines[0])
        or not any(f"{kind} " in matching_lines[0].lower() for kind in ("button", "link"))
        or not (_CHAT_WORDS.search(element) or _hint_matches(element, chat_hint))
    ):
        raise ChannelError("discover hover", "unsafe or ambiguous chat control")


def _check_click(snapshot: str, target: str, element: str, chat_hint: str | None = None) -> None:
    """Only an identified chat-opening control may be clicked."""
    matching_lines = [line for line in snapshot.splitlines() if f"[ref={target}]" in line]
    if (
        not re.fullmatch(r"e\d+", target)
        or not (_CHAT_WORDS.search(element) or _hint_matches(element, chat_hint))
        or _UNSAFE_WORDS.search(element)
        or len(matching_lines) != 1
        or not (
            _CHAT_WORDS.search(matching_lines[0]) or _hint_matches(matching_lines[0], chat_hint)
        )
        or _UNSAFE_WORDS.search(matching_lines[0])
    ):
        raise ChannelError("discover click", "unsafe or ambiguous chat control")
    line = matching_lines[0].lower()
    if not any(f"{kind} " in line for kind in ("button", "link")):
        raise ChannelError("discover click", "chat control must be a button or link")


def _unique_safe_launcher(snapshot: str, chat_hint: str | None) -> tuple[str, str] | None:
    """Select only a unique, explicitly labeled control passing the regular click guard."""
    candidates: list[tuple[str, str]] = []
    for line in snapshot.splitlines():
        match = _LABELED_CONTROL.match(line)
        if match is None:
            continue
        label, ref = match.group(2), match.group(3)
        try:
            _check_click(snapshot, ref, label, chat_hint)
        except ChannelError:
            continue
        candidates.append((ref, label))
    return candidates[0] if len(candidates) == 1 else None


class ManagedPlaywrightMCP:
    """Start a pinned MCP subprocess and contain its output and browser state."""

    def __init__(
        self,
        url: str,
        *,
        timeout_seconds: float = 15.0,
        headed: bool = False,
        chat_hint: str | None = None,
    ) -> None:
        self._origin = _origin(url)
        self._timeout = timeout_seconds
        self._headed = headed
        self._stack: AsyncExitStack | None = None
        self._session: object | None = None
        self._snapshot = ""
        self._chat_hint = chat_hint

    async def __aenter__(self) -> ManagedPlaywrightMCP:
        # Import the optional SDK only when discovery is invoked.
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            from playwright.async_api import async_playwright
        except ImportError:
            raise ConfigurationError(
                "discover setup", "install the discovery extra and Node.js"
            ) from None

        stack = AsyncExitStack()
        try:
            output = stack.enter_context(tempfile.TemporaryDirectory(prefix="syntheticcx-mcp-"))
            async with async_playwright() as playwright:
                executable = playwright.chromium.executable_path
            params = StdioServerParameters(
                command="npx",
                # Never forward model credentials or ambient provider settings
                # to the Node/browser subprocess.
                env={
                    key: os.environ[key] for key in ("PATH", "HOME", "TMPDIR") if key in os.environ
                },
                args=[
                    "--yes",
                    f"@playwright/mcp@{SERVER_VERSION}",
                    "--isolated",
                    "--no-webmcp",
                    "--image-responses",
                    "omit",
                    "--codegen",
                    "none",
                    "--output-dir",
                    output,
                    "--executable-path",
                    executable,
                    "--timeout-action",
                    str(int(self._timeout * 1000)),
                    "--timeout-navigation",
                    str(int(self._timeout * 1000)),
                ],
            )
            if not self._headed:
                params.args.insert(2, "--headless")
            async with asyncio.timeout(self._timeout):
                read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                available = {tool.name for tool in (await session.list_tools()).tools}
                if not ALLOWED_TOOLS.issubset(available):
                    raise ChannelError("discover setup", "required browser tools are unavailable")
            self._stack = stack
            self._session = session
            return self
        except BaseException:
            await stack.aclose()
            raise

    async def __aexit__(self, *_exc: object) -> None:
        stack, self._stack = self._stack, None
        self._session = None
        if stack is not None:
            await stack.aclose()

    async def _call(
        self, name: str, arguments: dict[str, str], *, check_origin: bool = True
    ) -> str:
        if name not in ALLOWED_TOOLS or self._session is None:
            raise ChannelError("discover tool", "tool is unavailable")
        # Session is kept opaque to all callers. The SDK type remains inside this adapter.
        try:
            from mcp import ClientSession
            from mcp.types import TextContent

            if not isinstance(self._session, ClientSession):
                raise ValueError("invalid session")
            async with asyncio.timeout(self._timeout):
                result = await self._session.call_tool(name, arguments=arguments)
            if (
                result.isError
                or len(result.content) != 1
                or not isinstance(result.content[0], TextContent)
            ):
                raise ValueError("tool failed")
            text = result.content[0].text
            if not text or len(text) > 100_000 or text.startswith("### Error"):
                raise ValueError("invalid tool output")
            if check_origin:
                _check_page_origin(text, self._origin)
            return cast(str, text)
        except asyncio.CancelledError:
            raise
        except ChannelError:
            raise
        except Exception:
            raise ChannelError("discover tool", "browser tool failed") from None

    async def navigate(self, url: str) -> str:
        if _origin(url) != self._origin:
            raise ChannelError("discover navigation", "browser left the selected target origin")
        await self._call("browser_navigate", {"url": url})
        return await self.snapshot()

    async def snapshot(self) -> str:
        text = await self._call("browser_snapshot", {})
        self._snapshot = text
        return text

    async def inspect_chat(self) -> str:
        """Collect bounded selector evidence only after a safe chat-opening click."""
        await self.snapshot()  # Confirm the target origin before running fixed JS.
        text = await self._call(
            "browser_evaluate", {"function": _CHAT_INSPECTION}, check_origin=False
        )
        try:
            result = json.loads(text.split("### Result\n", 1)[1].split("\n### ", 1)[0])
            if not isinstance(result, dict) or not isinstance(result.get("origin"), str):
                raise ValueError("invalid inspection result")
            if _origin(result["origin"]) != self._origin:
                raise ChannelError("discover navigation", "browser left the selected target origin")
            inventory = json.dumps(
                {"elements": result["elements"], "frames": result["frames"]},
                ensure_ascii=False,
            )
            if len(inventory) > 12_000:
                raise ValueError("inspection result too large")
        except ChannelError:
            raise
        except (ValueError, IndexError, KeyError, TypeError, ConfigurationError):
            raise ChannelError("discover inspection", "chat inspection failed") from None
        await self.snapshot()  # Also reject navigation during/after evaluation.
        return inventory

    async def hover(self, target: str, element: str, snapshot: str) -> str:
        if snapshot != self._snapshot:
            raise ChannelError("discover hover", "chat control snapshot is stale")
        _check_hover(snapshot, target, element, self._chat_hint)
        await self._call("browser_hover", {"target": target, "element": element})
        return await self.snapshot()

    async def click(self, target: str, element: str, snapshot: str) -> str:
        if snapshot != self._snapshot:
            raise ChannelError("discover click", "chat control snapshot is stale")
        _check_click(snapshot, target, element, self._chat_hint)
        await self._call("browser_click", {"target": target, "element": element})
        return await self.snapshot()

"""Independent, native chat-root attestation and immutable scope pins."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from playwright.async_api import ElementHandle, Frame, Locator, Page

from syntheticcx.adapters.channel.reading import READING_HELPERS
from syntheticcx.errors import ChannelError


def origin(url: str) -> tuple[str, str | None, int]:
    parsed = urlsplit(url)
    return parsed.scheme, parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)


CHAT_NAME = re.compile(
    r"\b(chat|webchat|chatbot|assistant|assistente|atendimento|conversation|conversa)\b", re.I
)

CHAT_PANEL_HELPER = """
const namedChatControl = n => n.matches('button,[role=button]') &&
  !n.hasAttribute('href') && /\\b(chat|webchat|chatbot|conversation|conversa)\\b/i.test(
    [n.getAttribute('aria-label') || '', n.getAttribute('title') || ''].join(' '));
const structuralChatPanel = root => {
  const scan = walkDOM(root);
  if (!scan.complete || !renderedElement(root)) return false;
  const composers = scan.nodes.filter(n => safeComposerElement(n) && renderedElement(n));
  if (composers.length !== 1) return false;
  const controls = scan.nodes.filter(n => n.matches('button,[role=button]') &&
    renderedElement(n) && !n.hasAttribute('href'));
  const namedChat = controls.filter(namedChatControl);
  const sends = controls.filter(n => /\\b(send|enviar)\\b/i.test(
    [n.getAttribute('aria-label') || '', n.getAttribute('title') || ''].join(' ')));
  if (!namedChat.length || !sends.length) return false;
  for (let p = composedParent(composers[0]); p && p !== root; p = composedParent(p)) {
    if (namedChat.some(n => composedContains(p,n)) && sends.some(n => composedContains(p,n)))
      return false;
  }
  return !root.matches('html,body,form,nav,main');
};
"""


class UnprovenChatRoot(ChannelError):
    """Read-only candidate rejection; origin and pinned-scope failures stay fatal."""


async def frame_origin_matches(frame: Frame, url: str) -> bool:
    """A page-overridable window.origin is never sufficient origin evidence."""
    expected = origin(url)
    ancestors: list[Frame] = []
    current: Frame | None = frame
    while current is not None:
        if len(ancestors) >= 32:
            return False
        if current.url not in {"about:blank", "about:srcdoc"} and origin(current.url) != expected:
            return False
        ancestors.append(current)
        current = current.parent_frame
    for child in ancestors:
        if child.parent_frame is None:
            continue
        element = await child.frame_element()
        try:
            sandbox = await element.get_attribute("sandbox")
            if sandbox is not None and "allow-same-origin" not in sandbox.split():
                return False
        finally:
            await element.dispose()
    reference = (
        "globalThis.origin" if frame.url in {"about:blank", "about:srcdoc"} else "location.origin"
    )
    return bool(await frame.evaluate(f"url => {reference} === new URL(url).origin", url))


@dataclass
class OwnedScope:
    page: Page
    frame: Frame
    locator: Locator
    root: ElementHandle
    url: str
    proof: str
    opener: ElementHandle | None = None
    embedded_frame: ElementHandle | None = None

    async def check(self) -> None:
        if origin(self.page.url) != origin(self.url):
            raise ChannelError("channel.ownership", "parent target origin changed")
        if not await frame_origin_matches(self.frame, self.url):
            raise ChannelError("channel.ownership", "selected frame origin changed")
        if await self.locator.count() != 1:
            # Navigation can race the count after the initial origin check.
            # Preserve the fatal origin diagnosis rather than misreporting drift.
            if origin(self.page.url) != origin(self.url):
                raise ChannelError("channel.ownership", "parent target origin changed")
            if not await frame_origin_matches(self.frame, self.url):
                raise ChannelError("channel.ownership", "selected frame origin changed")
            raise ChannelError("channel.ownership", "chat root is missing or ambiguous")
        current = await self.locator.element_handle()
        try:
            if not await self.root.evaluate(
                "(root, node) => root === node && root.isConnected", current
            ):
                raise ChannelError("channel.ownership", "pinned chat root changed")
        finally:
            if current is not None:
                await current.dispose()
        if self.proof == "chat_dialog":
            evidence = await self.root.evaluate(
                "node => {"
                + READING_HELPERS
                + """
              const refs = (node.getAttribute('aria-labelledby') || '')
                .trim().split(/\\s+/).filter(Boolean);
              const scan = walkDOM(node.getRootNode(), false);
              const labels = refs.map(id => scan.nodes.filter(el => el.id === id));
              return {dialog: node.matches('dialog,[role=dialog],[role=alertdialog]'),
                valid: scan.complete && refs.length <= 8 &&
                  labels.every(entries => entries.length === 1),
                name: node.getAttribute('aria-label') ||
                  labels.flat().map(labelText).join(' ')};
            }"""
            )
            if (
                not evidence["dialog"]
                or not evidence["valid"]
                or not CHAT_NAME.search(evidence["name"])
            ):
                raise ChannelError("channel.ownership", "positive chat dialog proof changed")
        elif self.proof == "chat_embed":
            if self.embedded_frame is None:
                valid = await self.root.evaluate(
                    "node => {"
                    + READING_HELPERS
                    + CHAT_PANEL_HELPER
                    + "return node.hasAttribute('data-chat-root') || "
                    "node.hasAttribute('data-chat-widget') || /^chat[-_]/i.test(node.tagName); }"
                )
            else:
                current_frame = await self.frame.frame_element()
                try:
                    metadata = await self.embedded_frame.evaluate(
                        "(node, current) => ({valid: node === current && node.isConnected, "
                        "name: node.getAttribute('aria-label') || "
                        "node.getAttribute('title') || ''})",
                        current_frame,
                    )
                    valid = metadata["valid"] and bool(CHAT_NAME.search(metadata["name"]))
                finally:
                    await current_frame.dispose()
            if not valid:
                raise ChannelError("channel.ownership", "positive chat embed proof changed")
        elif self.proof == "chat_panel":
            valid = await self.root.evaluate(
                "node => {"
                + READING_HELPERS
                + CHAT_PANEL_HELPER
                + "const scan = walkDOM(node); return scan.complete && "
                "scan.nodes.some(namedChatControl);}"
            )
            if not valid:
                raise ChannelError("channel.ownership", "positive chat panel proof changed")
        elif self.proof == "opener_linked":
            if self.opener is None or not await self.opener.evaluate(
                "(node, root) => node.isConnected && root.id && "
                "(node.getAttribute('aria-controls') || '').split(/\\s+/).includes(root.id)",
                self.root,
            ):
                raise ChannelError("channel.ownership", "positive opener relationship changed")

    async def contains(self, nodes: list[ElementHandle]) -> bool:
        await self.check()
        return bool(
            await self.root.evaluate(
                "(root, nodes) => {"
                + READING_HELPERS
                + "return nodes.every(node => node !== root && node.isConnected && "
                + "composedContains(root, node));}",
                nodes,
            )
        )


async def attest_root(
    page: Page,
    frame: Frame,
    locator: Locator,
    url: str,
    *,
    opener: ElementHandle | None = None,
    expected_proof: str | None = None,
) -> OwnedScope:
    """A new panel, arbitrary container, or body is never ownership evidence."""
    if origin(page.url) != origin(url) or not await frame_origin_matches(frame, url):
        raise ChannelError("channel.ownership", "target origin is not attested")
    if await locator.count() != 1:
        raise UnprovenChatRoot("channel.ownership", "chat root must be unique")
    root = await locator.element_handle()
    if root is None:
        raise UnprovenChatRoot("channel.ownership", "chat root is absent")
    embedded_frame = None
    try:
        metadata = await root.evaluate(
            "node => {"
            + READING_HELPERS
            + CHAT_PANEL_HELPER
            + r"""
          const refs = (node.getAttribute('aria-labelledby') || '')
            .trim().split(/\s+/).filter(Boolean);
          const scan = walkDOM(node.getRootNode(), false);
          const labels = refs.map(id => scan.nodes.filter(el => el.id === id));
          return {
          valid: node.isConnected && !node.matches('body,html,form,nav,main') &&
            !!node.getClientRects().length,
          dialog: node.matches('dialog,[role=dialog],[role=alertdialog]'),
          name: node.getAttribute('aria-label') ||
            labels.flat().map(labelText).join(' '),
          validLabels: scan.complete && refs.length <= 8 &&
            labels.every(entries => entries.length === 1),
          embed: node.getAttribute('data-chat-root') !== null ||
            node.getAttribute('data-chat-widget') !== null ||
            /^chat[-_]/i.test(node.tagName),
           structural: structuralChatPanel(node), id: node.id
        }; }"""
        )
        proof = None
        if (
            metadata["valid"]
            and metadata["dialog"]
            and metadata["validLabels"]
            and CHAT_NAME.search(metadata["name"])
        ):
            proof = "chat_dialog"
        elif metadata["valid"] and metadata["embed"]:
            proof = "chat_embed"
        elif metadata["valid"] and opener is not None and metadata["structural"]:
            proof = "chat_panel"
        elif metadata["valid"] and frame != page.main_frame:
            embedded_frame = await frame.frame_element()
            name = await embedded_frame.evaluate(
                "node => node.getAttribute('aria-label') || node.getAttribute('title') || ''"
            )
            if CHAT_NAME.search(name):
                proof = "chat_embed"
        elif metadata["valid"] and opener is not None and metadata["id"]:
            linked = await opener.evaluate(
                "(node, id) => node.isConnected && (node.getAttribute('aria-controls') || '')"
                ".split(/\\s+/).includes(id)",
                metadata["id"],
            )
            if linked:
                proof = "opener_linked"
        if proof is None or (expected_proof is not None and proof != expected_proof):
            raise UnprovenChatRoot(
                "channel.ownership", "independent positive chat ownership proof is missing"
            )
        owned = OwnedScope(
            page, frame, locator, root, url, proof, opener=opener, embedded_frame=embedded_frame
        )
        await owned.check()
        return owned
    except BaseException:
        await root.dispose()
        if embedded_frame is not None:
            await embedded_frame.dispose()
        raise

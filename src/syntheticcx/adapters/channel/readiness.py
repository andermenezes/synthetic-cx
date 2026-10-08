"""Read-only startup readiness inside independently attested chat roots."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from playwright.async_api import ElementHandle, Locator, Page
from playwright.async_api import Error as PlaywrightError

from syntheticcx.adapters.channel.options import eligibility
from syntheticcx.adapters.channel.ownership import (
    CHAT_NAME,
    OwnedScope,
    UnprovenChatRoot,
    attest_root,
    frame_origin_matches,
    origin,
)
from syntheticcx.adapters.channel.playwright import locate as native_locate
from syntheticcx.adapters.channel.reading import READING_HELPERS
from syntheticcx.config.models import LocatorConfig
from syntheticcx.discovery.catalog import css_string
from syntheticcx.errors import ChannelError


async def interaction_ready(
    owned: OwnedScope,
    assistant_locator: LocatorConfig | None = None,
    locate: Callable[..., Locator] = native_locate,
) -> bool:
    """An available composer or positively owned informational menu is ready."""
    await owned.check()
    inventory = await owned.locator.locator("*").element_handles()
    selected = inventory if assistant_locator is None else []
    try:
        if assistant_locator is not None:
            selected = await locate(owned.locator, assistant_locator).element_handles()
        if len(inventory) > 2000 or (assistant_locator is not None and len(selected) > 120):
            raise ChannelError("channel.start", "startup inspection limit exceeded")
        evidence: dict[str, Any] = await owned.root.evaluate(
            "(root, args) => {"
            + READING_HELPERS
            + """
          const ancestors = node => {
            const result = [];
            for (let p = node; p; p = p.assignedSlot || composedParent(p)) result.push(p);
            return result;
          };
          const enabled = node => !node.matches(':disabled') && !ancestors(node).some(p =>
            p.matches('[disabled],[inert],fieldset[disabled]') ||
            (p.getAttribute('aria-disabled') || '').toLowerCase() === 'true');
          const visible = node => renderedElement(node) && ancestors(node).every(p => {
            const style = getComputedStyle(p);
            return style.display !== 'none' && style.visibility === 'visible' &&
              style.contentVisibility !== 'hidden' && style.opacity !== '0' &&
              (p.getAttribute('aria-hidden') || '').toLowerCase() !== 'true';
          });
          const composers = args.inventory.filter(node => safeComposerElement(node) &&
            visible(node) && enabled(node));
          const forbidden = 'html,body,form,button,a,input,textarea,select,dialog,nav,main,' +
            'script,style,template,noscript,[role=dialog],[role=log],[role=feed],' +
            '[role=status],[role=button],[role=textbox]';
          const wrappers = args.selected.filter(node => node !== root && node.isConnected &&
            composedContains(root,node) && visible(node) && observedActor(node) === 'assistant' &&
            !node.matches(forbidden) && walkDOM(node).complete &&
            !walkDOM(node).nodes.some(child => child.matches(
              'input,textarea,select,[role=textbox],[contenteditable=true]')));
          if (wrappers.length > 120)
            return {composers: composers.length, menus: [], oversized: true};
          if (wrappers.some(node => wrappers.some(other => node !== other &&
            (composedContains(node,other) || composedContains(other,node)))))
            return {composers: composers.length, menus: []};
          const navigation = new Set(['href','target','download','formaction','form','src',
            'action','data-href','data-url','data-action']);
          const menus = wrappers.map(owner => {
            const controls = walkDOM(owner).nodes.filter(node =>
              node.matches('button,[role=button]'));
            if (controls.length > 64) return {oversized: true};
            return {prose: proseText(owner), options: controls.map(node => {
              const lineage = ancestors(node);
              const unsafe = lineage.some(p => p.matches('a,form,input,select,textarea,' +
                '[role=link],[contenteditable]:not([contenteditable=false])') ||
                [...p.attributes].some(attribute => navigation.has(attribute.name.toLowerCase())));
              return {label: composedText(node,1001), visible: visible(node),
                enabled: enabled(node),
                safe: !unsafe && node.isConnected && composedContains(root,node) &&
                  wrappers.filter(p => composedContains(p,node)).length === 1 &&
                  (node.tagName === 'BUTTON' ?
                    !['submit', 'reset'].includes(
                      (node.getAttribute('type') || '').toLowerCase()) :
                    node.getAttribute('role') === 'button')};
            })};
          });
          return {composers: composers.length, menus};
        }""",
            {"inventory": inventory, "selected": selected},
        )
        await owned.check()
        if evidence["composers"] == 1:
            return True
        if evidence.get("oversized") or any(
            menu.get("oversized") for menu in evidence.get("menus", [])
        ):
            return False
        if evidence["composers"]:
            return False
        for menu in evidence["menus"]:
            options = menu["options"]
            labels = [option["label"] for option in options]
            if any(label is None for label in labels):
                raise ChannelError("channel.start", "incomplete startup option label scan")
            if options and all(
                eligibility(menu["prose"], option["label"], option, labels) is None
                for option in options
            ):
                return True
        return False
    finally:
        for node in inventory:
            await node.dispose()
        if assistant_locator is not None:
            for node in selected:
                await node.dispose()


async def _page_ready(page: Page, url: str, open_node: ElementHandle | None) -> bool:
    if origin(page.url) != origin(url):
        raise ChannelError("channel.ownership", "parent target origin changed")
    linked = (await open_node.get_attribute("aria-controls") or "").split() if open_node else []
    for frame in page.frames[:20]:
        if not await frame_origin_matches(frame, url):
            if frame == page.main_frame:
                raise ChannelError("channel.ownership", "target document origin changed")
            continue
        chat_frame = False
        if frame != page.main_frame:
            frame_node = await frame.frame_element()
            try:
                metadata = await frame_node.evaluate(
                    "node => ({selector: node.id ? '#' + CSS.escape(node.id) : null, "
                    "name: node.getAttribute('aria-label') || node.getAttribute('title') || ''})"
                )
                if (
                    not metadata["selector"]
                    or await page.locator(metadata["selector"]).count() != 1
                ):
                    continue
                chat_frame = bool(CHAT_NAME.search(metadata["name"]))
            finally:
                await frame_node.dispose()
        inventory: dict[str, Any] = (
            await frame.locator("body")
            .locator(
                "*"
                if chat_frame
                else "dialog,[role=dialog],[role=alertdialog],[data-chat-root],[data-chat-widget],"
                "div:has(textarea),section:has(textarea),div:has(input),section:has(input),"
                "div:has([role=textbox]),section:has([role=textbox]),"
                "div:has([contenteditable=true]),section:has([contenteditable=true])"
                + "".join(", [id=" + css_string(identifier) + "]" for identifier in linked)
            )
            .evaluate_all(
                "(nodes, args) => {"
                + READING_HELPERS
                + """
          if (nodes.length > 2000) return {oversized: true};
          const excluded = 'html,body,form,nav,main,button,input,textarea,select,a,iframe,' +
            'script,style,template,noscript';
           return {selectors: nodes.filter(node => renderedElement(node) &&
            (node.hasAttribute('data-chat-root') || node.hasAttribute('data-chat-widget') ||
              /^chat[-_]/i.test(node.tagName) ||
              node.matches('dialog,[role=dialog],[role=alertdialog]') ||
                args.linked.includes(node.id) ||
                node.querySelector('textarea,input,[role=textbox],[contenteditable=true]') ||
               (args.chatFrame && !node.matches(excluded))))
            .map(node => node.id ? '#' + CSS.escape(node.id) : node.tagName.toLowerCase() +
              [...node.classList].map(value => '.' + CSS.escape(value)).join(''))};
        }""",
                {"linked": linked, "chatFrame": chat_frame},
            )
        )
        if inventory.get("oversized"):
            raise ChannelError("channel.start", "startup inspection limit exceeded")
        for selector in dict.fromkeys(inventory["selectors"]):
            candidate = None
            try:
                candidate = await attest_root(
                    page, frame, frame.locator(selector), url, opener=open_node
                )
                if await interaction_ready(candidate):
                    return True
            except UnprovenChatRoot:
                if origin(page.url) != origin(url) or not await frame_origin_matches(frame, url):
                    raise ChannelError(
                        "channel.ownership", "startup target origin changed"
                    ) from None
            finally:
                if candidate is not None:
                    await candidate.root.dispose()
                    if candidate.embedded_frame is not None:
                        await candidate.embedded_frame.dispose()
    if origin(page.url) != origin(url):
        raise ChannelError("channel.ownership", "parent target origin changed")
    return False


async def wait_for_interaction(
    page: Page, url: str, open_node: ElementHandle | None, deadline: float
) -> None:
    """Wait within the caller's existing absolute deadline; never dispatch an action."""
    try:
        async with asyncio.timeout_at(deadline):
            while True:
                try:
                    if await _page_ready(page, url, open_node):
                        return
                except PlaywrightError:
                    # A navigation can destroy the execution context before its
                    # final URL is committed. Reobserve read-only within the same
                    # deadline so origin drift is classified on the new document.
                    if page.is_closed():
                        raise
                await asyncio.sleep(0.05)
    except TimeoutError:
        raise ChannelError(
            "channel.start",
            "owned interaction controls did not become ready before startup deadline",
        ) from None
    except PlaywrightError:
        if origin(page.url) != origin(url):
            raise ChannelError("channel.ownership", "startup target origin changed") from None
        raise ChannelError(
            "channel.start", "startup DOM inspection context is unavailable"
        ) from None

"""Native, recipient-bound website exploration and exact typed path replay.

The planner sees current-revision opaque controls, never executable selectors.
WhatsApp account DOM is never inspected, including while waiting for effects.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import secrets
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlsplit

from playwright.async_api import ElementHandle, Page
from playwright.async_api import Error as PlaywrightError

from syntheticcx.adapters.channel.playwright import locate
from syntheticcx.config.models import ExecutionLimits, LocatorConfig, WebsiteNavigationStep
from syntheticcx.discovery.policy import portability_flow
from syntheticcx.errors import ChannelError


def origin(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return ""
    return f"{parsed.scheme}://{parsed.hostname}" + (f":{parsed.port}" if parsed.port else "")


def whatsapp_destination(value: str) -> tuple[str, str] | None:
    """Only explicit, unique HTTPS recipient handoffs qualify as evidence."""
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or parsed.username
            or parsed.password
            or parsed.port not in {None, 443}
        ):
            return None
        if parsed.hostname == "wa.me":
            phone = parsed.path.strip("/")
            kind = "landing"
        elif (
            parsed.hostname
            in {"api.whatsapp.com", "www.whatsapp.com", "whatsapp.com", "web.whatsapp.com"}
            and parsed.path.rstrip("/") == "/send"
        ):
            phones = parse_qs(parsed.query).get("phone", [])
            if len(phones) != 1:
                return None
            phone = phones[0]
            kind = "web" if parsed.hostname == "web.whatsapp.com" else "landing"
        else:
            return None
        if not re.fullmatch(r"\+?[0-9]{6,15}", phone):
            return None
        return kind, phone.lstrip("+")
    except ValueError:
        return None


# Both the chosen label and the ORIGINAL text/attributes participate in safety.
_CONTROL = r"""node => {
  if (node.ownerDocument.location.hostname === 'web.whatsapp.com')
    return {safe: false};
  const refs = (node.getAttribute('aria-labelledby') || '').trim().split(/\s+/).filter(Boolean);
  const labelled = refs.map(id =>
    node.getRootNode().getElementById?.(id)?.innerText || '').join(' ');
  const icon = node.querySelector('img[alt],[aria-label],svg title');
  const title = node.getAttribute('title') || '';
  const text = (node.innerText || '').trim();
  const aria = node.getAttribute('aria-label') || '';
  const meaningful = /[\p{L}]/u.test(text);
  const label = (aria || labelled || (meaningful ? text : '') || title ||
    icon?.getAttribute('alt') || icon?.getAttribute('aria-label') ||
    icon?.textContent || text).trim();
  const unsafe = new RegExp(
    "\\b(pay|payment|purchase|buy|order|submit|delete|confirm|login|consent|agree|" +
    "accept|upload|log in|sign in|sign up|register|checkout|subscribe|comprar|pagar|" +
    "contratar|assinar|enviar|excluir|confirmar|entrar|aceitar|cadastro|acessar|" +
    "minha conta|my account)\\b", "i");
  const link = node.tagName === 'A' && node.hasAttribute('href');
  const button = (node.tagName === 'BUTTON' && node.type !== 'reset' && !node.form) ||
    (node.getAttribute('role') === 'button' && node.tagName !== 'INPUT');
  const ancestors = [];
  for (let p = node; p; p = p.assignedSlot || p.parentElement || p.getRootNode()?.host)
    ancestors.push(p);
  const blocked = ancestors.some(p => p.matches('form,[inert],[disabled],[aria-disabled=true]') ||
    ['download','formaction','form','action','data-action','data-url','data-href']
      .some(a => p.hasAttribute(a)) || (p !== node && p.matches('a,[role=link]')));
  const rect = node.getBoundingClientRect();
  const inViewport = rect.bottom > 0 && rect.right > 0 &&
    rect.top < innerHeight && rect.left < innerWidth;
  const x = Math.max(0, Math.min(innerWidth - 1,
    (Math.max(0,rect.left) + Math.min(innerWidth,rect.right))/2));
  const y = Math.max(0, Math.min(innerHeight - 1,
    (Math.max(0,rect.top) + Math.min(innerHeight,rect.bottom))/2));
  const top = inViewport ? node.ownerDocument.elementFromPoint(x,y) : null;
  const receives = inViewport && (top === node || node.contains(top) ||
    top?.getRootNode()?.host === node);
  const context = node.closest(
    'dialog,[role=dialog],[role=alertdialog],li,section,article,nav') || node.parentElement;
  const names = [label,text,title,aria,labelled,node.id,
    icon?.getAttribute('alt') || '', icon?.getAttribute('aria-label') || '',
    icon?.textContent || ''];
  const sendsMessage = button && !link && names.some(value =>
    /^(send|send message|enviar|enviar mensagem)$/i.test(value.trim().replace(/\s+/g, ' ')));
  return {safe: node.isConnected && !!node.getClientRects().length &&
    getComputedStyle(node).visibility !== 'hidden' && !node.matches(':disabled') && !blocked &&
    !unsafe.test(names.join(' ')) && !sendsMessage &&
    (button || link),
    label: label.slice(0,256), href: link ? node.href : '',
    target: link ? (node.target || '').toLowerCase() : '',
    role: node.getAttribute('role') || (link ? 'link' : 'button'),
    selector: node.id ? '#' + CSS.escape(node.id) :
      title ? node.tagName.toLowerCase() + '[title=' + JSON.stringify(title) + ']' : null,
    context: ancestors.slice(1,7).map(p => p.getAttribute('aria-label') ||
      [...p.children].filter(n => /^(H[1-6]|LEGEND)$/.test(n.tagName))
        .map(n => n.innerText || '').join(' ')).filter(Boolean)
      .concat((context?.innerText || '').slice(0,700)).join(' > ').slice(0,1400),
    actionable: receives,
    blockedBy: !inViewport ? (rect.top >= innerHeight ? 'below_viewport' : 'outside_viewport') :
      receives ? '' : ((top?.closest(
        'dialog,[role=dialog],[role=alertdialog]') || top)?.innerText || '').slice(0,400)};
}"""

# Visible overlays and viewport content precede the document's introductory prose.
_PAGE_STATE = r"""node => {
  if (node.ownerDocument.location.hostname === 'web.whatsapp.com') return null;
  const visible = n => !!n.getClientRects().length && getComputedStyle(n).visibility !== 'hidden';
  const overlays = [...node.querySelectorAll(
    'dialog,[role=dialog],[role=alertdialog],[aria-modal=true],[role=menu]')]
    .filter(visible).map(n => (n.innerText || '').slice(0,2400));
  const controls = [...node.querySelectorAll(
    'a[href],button,[role=button],textarea,[role=textbox]')]
    .filter(visible).map(n => ({
      text: (n.innerText || n.getAttribute('aria-label') || n.title || '').slice(0,256),
      expanded: n.getAttribute('aria-expanded'), role: n.getAttribute('role')}));
  const viewport = [...node.querySelectorAll('h1,h2,h3,p,li,section')].filter(n => {
    const r=n.getBoundingClientRect(); return visible(n) && r.bottom>0 && r.top<innerHeight;
  }).slice(-30).map(n => (n.innerText || '').slice(0,600)).join('\n').slice(0,4000);
  const root = node.ownerDocument.scrollingElement;
  return {overlays, viewport, text: (node.innerText || '').slice(0,6000), controls,
    scroll: Math.round(root.scrollTop), height: root.scrollHeight, viewportHeight: innerHeight};
}"""

_SENSITIVE = re.compile(r"https?://\S+|[+]?[0-9][0-9\s().-]{5,}[0-9]")


def _public(value: str) -> str:
    return _SENSITIVE.sub("[redacted]", " ".join(value.split()))


@dataclass
class WebsiteControl:
    node: ElementHandle
    locator: LocatorConfig
    label: str
    href: str
    context: str
    actions: list[str]
    blocked_by: str
    target: str = ""


class WebsiteNavigator:
    """Bounded state-aware discovery; every dispatch reserves its revision first."""

    def __init__(
        self, page: Page, url: str, limits: ExecutionLimits, *, recipient: str | None = None
    ) -> None:
        self.page = page
        self.url = url
        self.limits = limits
        self.recipient = recipient
        self.steps: list[WebsiteNavigationStep] = []
        self.last_node: ElementHandle | None = None
        self.revision = ""
        self.feedback = "initial_page"
        self.attested = False
        self.discovery_only = False
        self._controls: dict[str, WebsiteControl] = {}
        self._pages = {page}
        self._initial_pages = set(page.context.pages)
        self._used: set[tuple[str, str, str]] = set()
        self._offset = 0
        self._state_key = ""
        self._available: list[str] = []
        self._history: list[dict[str, Any]] = []
        self._back: list[tuple[Page, str]] = []
        self._attempts = 0
        self._uncertain = False
        self._observations = 0

    async def _clear(self) -> None:
        controls, self._controls = self._controls, {}
        for item in controls.values():
            if item.node is not self.last_node:
                await item.node.dispose()

    def _destination_allowed(self, href: str) -> bool:
        destination = whatsapp_destination(href)
        if destination is not None:
            return self.recipient is not None and destination[1] == self.recipient
        parsed = urlsplit(href)
        return bool(
            origin(href) == origin(self.url) and not parsed.username and not parsed.password
        )

    async def terminal(self) -> bool:
        """Validate all owned/new URLs before reading any website DOM."""
        current = [
            p
            for p in self.page.context.pages
            if not p.is_closed() and (p in self._pages or p not in self._initial_pages)
        ]
        web = [p for p in current if urlsplit(p.url).hostname == "web.whatsapp.com"]
        if len(web) > 1:
            raise ChannelError("channel.ownership", "multiple WhatsApp Web conversations opened")
        for page in current:
            if page.url == "about:blank":
                continue
            destination = whatsapp_destination(page.url)
            if destination is not None:
                if self.recipient is None or destination[1] != self.recipient:
                    raise ChannelError(
                        "channel.ownership", "handoff recipient changed or unsupported channel"
                    )
                self.attested = True
            elif origin(page.url) != origin(self.url):
                parsed = urlsplit(page.url)
                if (
                    parsed.scheme != "https"
                    or parsed.hostname != "web.whatsapp.com"
                    or parsed.username is not None
                    or parsed.password is not None
                    or parsed.port not in {None, 443}
                    or parsed.path not in {"/", "/send", "/send/"}
                ):
                    raise ChannelError(
                        "channel.ownership", "website reached an unrelated destination"
                    )
                explicit = parse_qs(parsed.query).get("phone", [])
                if explicit and (len(explicit) != 1 or explicit[0].lstrip("+") != self.recipient):
                    raise ChannelError("channel.ownership", "handoff recipient changed")
                if not self.attested:
                    raise ChannelError(
                        "channel.ownership", "WhatsApp destination lacks recipient evidence"
                    )
        if web:
            if not self.attested:
                raise ChannelError(
                    "channel.ownership", "WhatsApp destination lacks recipient evidence"
                )
            self.page = web[0]
            self._pages.add(self.page)
            return True
        new = [p for p in current if p not in self._pages and p.url != "about:blank"]
        if new:
            self.page = new[-1]
            self._pages.update(new)
        return False

    async def _state(self) -> dict[str, Any] | None:
        deadline = time.monotonic() + self.limits.browser_action_timeout_seconds * 0.9
        while True:
            # A URL can commit between capturing it and evaluating the body.
            # Re-attest before retrying; never inspect the newly committed account
            # document or turn a genuine origin/recipient violation into a retry.
            if await self.terminal():
                return None
            expected = self.page.url
            if not self._destination_allowed(expected):
                raise ChannelError(
                    "channel.ownership", "website document is outside the allowed boundary"
                )
            state = await self.page.locator("body").evaluate(
                "(node, expected) => node.ownerDocument.location.href === expected"
                f" ? ({_PAGE_STATE})(node) : null",
                expected,
            )
            if isinstance(state, dict):
                return state
            if await self.terminal():
                return None
            if time.monotonic() >= deadline:
                raise ChannelError(
                    "channel.ownership", "website document did not settle during inspection"
                )
            await asyncio.sleep(0.025)

    def _key(self, state: dict[str, Any]) -> str:
        return hashlib.sha256(
            (self.page.url + json.dumps(state, sort_keys=True)).encode()
        ).hexdigest()[:20]

    async def _locator(self, node: ElementHandle, data: dict[str, Any]) -> LocatorConfig | None:
        selectors: list[LocatorConfig] = []
        selector = data["selector"]
        if selector and not _SENSITIVE.search(selector):
            selectors.append(LocatorConfig(strategy="css", value=selector))
        if data["label"] and not _SENSITIVE.search(data["label"]):
            selectors.append(LocatorConfig(strategy="role", value=data["role"], name=data["label"]))
        if selector and not _SENSITIVE.search(selector):
            selectors.append(LocatorConfig(strategy="css", value=selector + ":visible"))
        for config in selectors:
            found = locate(self.page, config)
            if await found.count() == 1 and await found.evaluate(
                "(current, observed) => current === observed", node
            ):
                return config
        return None

    async def observe(
        self, *, initial_launcher_id: str | None = None, chat_hint: str | None = None
    ) -> dict[str, Any]:
        self._observations += 1
        if self._observations > self.limits.website_observation_limit:
            raise ChannelError("channel.observe", "website observation limit exceeded")
        await self._clear()
        self.revision = secrets.token_urlsafe(18)
        state = await self._state()
        if state is None:
            return {
                "revision": self.revision,
                "terminal": "whatsapp",
                "available_actions": ["observe", "stop"],
                "catalog": {"candidates": [], "truncated": False},
            }
        self._state_key = self._key(state)
        if initial_launcher_id is not None:
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_:-]{0,127}", initial_launcher_id):
                raise ChannelError("discover hint", "invalid exact launcher ID")
            handles = await self.page.locator(f'[id="{initial_launcher_id}"]').element_handles()
        else:
            handles = await self.page.locator("button,[role=button]").element_handles()
            handles.extend(await self.page.locator("a[href]:not([role=button])").element_handles())
        if len(handles) > 2000:
            for node in handles:
                await node.dispose()
            raise ChannelError("channel.observe", "website inspection limit exceeded")
        metadata = await self.page.evaluate(f"nodes => nodes.map({_CONTROL})", handles)
        hint = " ".join((chat_hint or "").casefold().split())
        # Hidden/unsafe responsive copies must not consume a catalog page and
        # make previously observed overlay choices disappear on every observe.
        entries = []
        for node, data in zip(handles, metadata, strict=True):
            if not data["safe"] or (data["href"] and not self._destination_allowed(data["href"])):
                await node.dispose()
            else:
                entries.append((node, data))
        entries.sort(
            key=lambda entry: (
                not (hint and hint in " ".join(entry[1].get("label", "").casefold().split())),
                not bool(whatsapp_destination(entry[1].get("href", ""))),
                not entry[1]["actionable"],
            )
        )
        start = self._offset if self._offset < len(entries) else 0
        has_more = start + 160 < len(entries)
        self._offset = start + 160 if has_more else 0
        on_handoff_page = (
            origin(self.page.url) != origin(self.url)
            and whatsapp_destination(self.page.url) is not None
        )
        for index, (node, data) in enumerate(entries):
            keep = False
            try:
                if not start <= index < start + 160:
                    continue
                locator = await self._locator(node, data)
                if locator is None:
                    continue
                control_actions = (
                    ("follow_link",)
                    if on_handoff_page and whatsapp_destination(data["href"]) is not None
                    else ("open", "hover", "follow_link")
                )
                actions = [
                    action
                    for action in control_actions
                    if data["actionable"]
                    and (action != "follow_link" or data["href"])
                    and (self._state_key, locator.model_dump_json(), action) not in self._used
                ]
                self._controls[secrets.token_urlsafe(12)] = WebsiteControl(
                    node,
                    locator,
                    data["label"],
                    data["href"],
                    data["context"],
                    actions,
                    data["blockedBy"],
                    target=data["target"],
                )
                keep = True
            finally:
                if not keep:
                    await node.dispose()
        self._available = ["observe", "stop", "wait"]
        self._available.extend(
            action
            for action in ("open", "hover", "follow_link")
            if any(action in item.actions for item in self._controls.values())
        )
        if (
            state["scroll"] + state["viewportHeight"] < state["height"] - 2
            and (self._state_key, "", "scroll_down") not in self._used
        ):
            self._available.append("scroll_down")
        if state["scroll"] > 2 and (self._state_key, "", "scroll_up") not in self._used:
            self._available.append("scroll_up")
        if self._back:
            self._available.append("back")
        if self._uncertain or self._attempts >= self.limits.website_action_limit:
            self._available = ["observe", "stop"]
        context = "\n".join([*state["overlays"], state["viewport"], state["text"]])[:10000]
        return {
            "revision": self.revision,
            "page_context": _public(context),
            "page": {
                "origin": origin(self.page.url),
                "state": self._state_key,
                "scroll_position": state["scroll"],
                "back_depth": len(self._back),
            },
            "feedback": self.feedback,
            "path_length": len(self.steps),
            "terminal": None,
            "available_actions": self._available,
            "history": list(self._history[-128:]),
            "catalog_page": start // 160,
            "catalog": {
                "truncated": len(entries) > 160,
                "has_more": has_more,
                "candidates": [
                    {
                        "id": key,
                        "kind": "opener",
                        "description": _public(item.label),
                        "context": _public(item.context),
                        "actions": item.actions,
                        "blocked_by": _public(item.blocked_by),
                        "tried_actions": [
                            action
                            for action in ("open", "hover", "follow_link")
                            if (self._state_key, item.locator.model_dump_json(), action)
                            in self._used
                        ],
                        "evidence": "recipient_handoff"
                        if whatsapp_destination(item.href)
                        else "same_origin_link"
                        if item.href
                        else "native_button",
                    }
                    for key, item in self._controls.items()
                ],
            },
        }

    async def act(self, revision: str, action: str, target_id: str | None = None) -> None:
        if revision != self.revision or not self.revision:
            raise ChannelError("channel.open", "website proposal is missing or stale")
        if action not in self._available or action in {"observe", "stop"}:
            raise ChannelError("channel.open", "website action is unavailable")
        if (action in {"open", "hover", "follow_link"}) != (target_id is not None):
            raise ChannelError("channel.open", "website action target is invalid")
        item = self._controls.get(target_id) if target_id is not None else None
        if target_id is not None and (item is None or action not in item.actions):
            raise ChannelError("channel.open", "website control action is unavailable")
        await self._dispatch(action, item)

    async def _checked(self, item: WebsiteControl) -> None:
        found = locate(self.page, item.locator)
        if await found.count() != 1 or not await found.evaluate(
            "(node, expected) => node === expected", item.node
        ):
            self.feedback = "control_changed_before_dispatch"
            self.revision = ""
            raise ChannelError("website.readiness", "website control changed before dispatch")
        data = await item.node.evaluate(_CONTROL)
        if self.discovery_only and portability_flow(
            data["label"], data.get("context", ""), data.get("href", "")
        ):
            raise ChannelError(
                "channel.open", "number-portability flows are forbidden in discovery"
            )
        if (
            not data["safe"]
            or data["label"] != item.label
            or data["href"] != item.href
            or data["target"] != item.target
            or (data["href"] and not self._destination_allowed(data["href"]))
        ):
            raise ChannelError("channel.open", "website control is unsafe or destination changed")
        if not data["actionable"]:
            self.feedback = "control_blocked_before_dispatch"
            self.revision = ""
            raise ChannelError(
                "website.readiness", "website control became blocked before dispatch"
            )

    async def _dispatch(self, action: str, item: WebsiteControl | None = None) -> None:
        if self._uncertain or self._attempts >= self.limits.website_action_limit:
            raise ChannelError("channel.open", "website navigation limit or uncertain dispatch")
        if await self.terminal():
            raise ChannelError("channel.open", "website handoff already completed")
        if item is not None:
            await self._checked(item)
        if action == "back" and not self._back:
            raise ChannelError("channel.open", "website back path is unavailable")
        page, before_url = self.page, self.page.url
        before = await self._state()
        if before is None:
            raise ChannelError("channel.open", "website handoff already completed")
        state_key = self._key(before)
        self._attempts += 1
        self.revision = ""
        self._uncertain = True
        self._offset = 0
        if item is not None:
            self._used.add((state_key, item.locator.model_dump_json(), action))
        elif action in {"scroll_down", "scroll_up"}:
            self._used.add((state_key, "", action))
        entry = {
            "action": action,
            "state": state_key,
            "description": _public(item.label) if item else "",
            "result": "uncertain",
        }
        self._history.append(entry)
        timeout = self.limits.browser_action_timeout_seconds * 1000
        if action in {"open", "hover", "follow_link"}:
            assert item is not None
            if action in {"open", "follow_link"} and whatsapp_destination(item.href) is not None:
                self.attested = True
            if action == "follow_link":
                if not item.href:
                    raise ChannelError("channel.open", "website control is not a link")
                await page.goto(item.href, wait_until="domcontentloaded", timeout=timeout)
            elif action == "open":
                if item.target == "_blank":
                    # Subscribe before dispatch: automatic scroll-into-view is not
                    # completion evidence for a link that advertises a new tab.
                    async with page.expect_popup(timeout=timeout) as pending_popup:
                        await item.node.click(no_wait_after=True, timeout=timeout)
                    popup = await pending_popup.value
                    await popup.wait_for_url(
                        lambda url: url != "about:blank",
                        wait_until="commit",
                        timeout=timeout,
                    )
                    # URL-only attestation precedes every read of the new page.
                    await self.terminal()
                else:
                    await item.node.click(no_wait_after=True, timeout=timeout)
            else:
                await item.node.hover(timeout=timeout)
        elif action in {"scroll_down", "scroll_up"}:
            await page.mouse.move(1, 1)
            await page.mouse.wheel(
                0, (1 if action == "scroll_down" else -1) * before["viewportHeight"] * 0.8
            )
        elif action == "back":
            previous, previous_url = self._back[-1]
            if previous.is_closed() or not self._destination_allowed(previous_url):
                raise ChannelError("channel.ownership", "observed back destination is unavailable")
            if previous is not page:
                await page.close()
                self.page = previous
                await previous.bring_to_front()
            if self.page.url != previous_url:
                await self.page.goto(previous_url, wait_until="domcontentloaded", timeout=timeout)
            self._back.pop()
        elif action != "wait":
            raise ChannelError("channel.open", "unsupported website action")
        step = WebsiteNavigationStep.model_validate(
            {
                "action": "click" if action == "open" else action,
                "locator": item.locator if item else None,
            }
        )
        self.steps.append(step)
        if action in {"open", "follow_link"} and item is not None:
            previous_node, self.last_node = self.last_node, item.node
            if previous_node is not None and previous_node is not item.node:
                await previous_node.dispose()
        deadline = time.monotonic() + min(2.0, self.limits.browser_action_timeout_seconds * 0.8)
        while True:
            if await self.terminal():
                self.feedback = "recipient_attested_whatsapp_handoff"
                break
            try:
                if self.page is not page or self.page.url != before_url:
                    await self.page.wait_for_load_state("domcontentloaded", timeout=timeout)
                    self.feedback = "navigation_completed"
                    break
                after = await self._state()
                if after is None:
                    self.feedback = "recipient_attested_whatsapp_handoff"
                    break
                if self.page is not page or self.page.url != before_url:
                    continue
                changed = after != before
                if action in {"open", "hover"} and after["scroll"] != before["scroll"]:
                    # Clicking an off-screen control can scroll before an unknown
                    # JS popup/navigation commits. Scroll alone must not end the
                    # result wait; actual content/control changes still can.
                    changed = any(
                        after[key] != before[key]
                        for key in ("text", "controls", "overlays", "height")
                    )
                if changed:
                    self.feedback = "visible_page_changed"
                    break
            except PlaywrightError:
                pass
            if time.monotonic() >= deadline:
                self.feedback = "no_visible_change_yet"
                break
            await asyncio.sleep(0.05)
        if action != "back" and (self.page is not page or self.page.url != before_url):
            self._back.append((page, before_url))
        entry["result"] = self.feedback
        if (
            action in {"open", "hover"}
            and item is not None
            and self.feedback == "visible_page_changed"
            and self.page is page
            and self.page.url == before_url
        ):
            # Successful same-page reveal controls must reopen after backtracking.
            # No-effect controls and navigated branches remain exhausted per state.
            self._used.discard((state_key, item.locator.model_dump_json(), action))
        self._uncertain = False

    async def replay(self, steps: list[WebsiteNavigationStep]) -> None:
        for step in steps:
            if await self.terminal():
                raise ChannelError(
                    "channel.open", "website replay ended before exact path completed"
                )
            item = None
            if step.locator is not None:
                found = locate(self.page, step.locator)
                deadline = time.monotonic() + self.limits.browser_action_timeout_seconds * 0.9
                while True:
                    if await self.terminal():
                        raise ChannelError("channel.open", "website replay completed unexpectedly")
                    count = await found.count()
                    if count > 1:
                        raise ChannelError("channel.open", "website replay control is ambiguous")
                    if count == 1 and await found.is_visible():
                        node = await found.element_handle()
                        if node is not None:
                            try:
                                data = await node.evaluate(_CONTROL)
                                if not data["safe"] or (
                                    data["href"] and not self._destination_allowed(data["href"])
                                ):
                                    raise ChannelError(
                                        "channel.open", "website replay control is unsafe"
                                    )
                                if data["actionable"]:
                                    item = WebsiteControl(
                                        node,
                                        step.locator,
                                        data["label"],
                                        data["href"],
                                        "",
                                        [],
                                        "",
                                        target=data["target"],
                                    )
                            finally:
                                if item is None:
                                    await node.dispose()
                            if item is not None:
                                break
                    if time.monotonic() >= deadline:
                        raise ChannelError(
                            "website.readiness", "website replay control is not ready"
                        )
                    await asyncio.sleep(0.025)
            try:
                await self._dispatch("open" if step.action == "click" else step.action, item)
            finally:
                if item is not None and item.node is not self.last_node:
                    await item.node.dispose()
        await self.terminal()

    async def close(self) -> None:
        await self._clear()
        if self.last_node is not None:
            await self.last_node.dispose()
            self.last_node = None

"""Dedicated-profile WhatsApp Web adapter with recipient-attested native actions.

Selectors describe a conservative supported surface, not live-verified compatibility.
Identity loss, unsupported menus, and unowned dialogs fail closed. No account-wide
screenshots, account navigation, login automation, or messaging retries are used.
"""

from __future__ import annotations

import asyncio
import fcntl
import os
import re
import time
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any, Concatenate, ParamSpec, TypeVar
from urllib.parse import urlsplit
from uuid import uuid4

from playwright.async_api import (
    BrowserContext,
    ElementHandle,
    JSHandle,
    Page,
    Playwright,
    Request,
    async_playwright,
)

from syntheticcx.adapters.channel.broker import NativeChatBroker
from syntheticcx.adapters.channel.options import eligibility
from syntheticcx.adapters.channel.playwright import normalize_echo
from syntheticcx.adapters.channel.website import whatsapp_destination
from syntheticcx.config.models import ExecutionLimits, WhatsAppChannelConfig
from syntheticcx.discovery.policy import portability_flow
from syntheticcx.domain.models import (
    CompletedResponse,
    Message,
    ReplyOption,
    SelectionReceipt,
    SendReceipt,
)
from syntheticcx.domain.protocols import ObservedControls
from syntheticcx.errors import ChannelError, ConfigurationError, ModelError

_MESSAGE_ID = re.compile(r"^(true|false)_([0-9]{6,15})@(c\.us|s\.whatsapp\.net)_(.+)$")
_COMPOSER = 'footer [role="textbox"][contenteditable="true"]'
_CONTACT_OPENER = 'header [data-testid="conversation-info-header"][role="button"]'
_BUTTONS = (
    'button:not([data-testid="quoted-message"] *),'
    '[role="button"]:not([data-testid="quoted-message"] *)'
)

# Current Web builds expose opaque stanza IDs, including system notifications.
# These IDs are never recipient evidence: they require the live contact proof.
_IDENTITY_JS = r"""
const identity = node => {
  const id = node.getAttribute('data-id') || '';
  const legacy = /^(true|false)_([0-9]{6,15})@(c\.us|s\.whatsapp\.net)_(.+)$/.exec(id);
  if (legacy) return {recipient: legacy[2], actor: legacy[1] === 'true' ? 'customer' : 'assistant'};
  if (!/^[A-F0-9]{16,64}$/.test(id) ||
      node.getAttribute('data-testid') !== 'conv-msg-' + id) return null;
  const system = node.querySelectorAll('[data-testid="system_message"]').length;
  const messages = node.querySelectorAll('[data-testid="msg-container"]').length;
  if (system === 1 && messages === 0) return {system: true};
  if (system > 1 || messages > 1 || system && messages) return null;
  // Older rows retain their stanza ID but deliberately unmount their contents.
  // Unlike an in-progress render, this shell will not hydrate just by waiting.
  if (!messages && node.querySelector(':scope > [data-virtualized="true"]') &&
      !node.textContent.trim() && !node.querySelector('button,[role="button"],[role="textbox"]'))
    return {virtualized: true};
  // React inserts the opaque stanza shell before its direction/contents.
  // This is not recipient evidence and must not revoke an independent proof.
  if (messages === 0) return {pending: true};
  // Consecutive bubbles omit their tail. The current Web surface groups each
  // bubble in a column aligned to the outgoing/incoming edge independently.
  const container = node.querySelector('[data-testid="msg-container"]');
  const group = container.parentElement;
  const layout = node.contains(group) ? getComputedStyle(group) : null;
  const column = layout && layout.display === 'flex' && layout.flexDirection === 'column';
  const outgoing = node.matches('.message-out') || !!node.querySelector(
    '.message-out,[data-testid="tail-out"],[data-testid="fail-container"]') ||
    column && layout.alignItems === 'flex-end';
  const incoming = node.matches('.message-in') || !!node.querySelector(
    '.message-in,[data-testid="tail-in"]') || column && layout.alignItems === 'flex-start';
  if (outgoing && incoming) return null;
  if (!outgoing && !incoming) return {pending: true};
  return {actor: outgoing ? 'customer' : 'assistant'};
};
"""


def resolve_recipient(config: WhatsAppChannelConfig, recipient: str | None = None) -> str:
    """Resolve an explicit recipient before environment fallback; never persist it."""
    value = (
        recipient
        if recipient is not None
        else os.environ.get(config.recipient_environment_variable, "")
    )
    if not re.fullmatch(r"\+?[0-9]{6,15}", value):
        raise ConfigurationError("channel.config", "a valid international recipient is required")
    return value.lstrip("+")


def validate_browser_profile(path: Path) -> Path:
    """Require an existing dedicated user-data root, never a personal browser profile."""
    try:
        resolved = path.expanduser().resolve(strict=True)
        personal = (
            Path.home() / "Library/Application Support/Google/Chrome",
            Path.home() / "Library/Application Support/Chromium",
            Path.home() / "Library/Application Support/Microsoft Edge",
            Path.home() / "Library/Application Support/BraveSoftware/Brave-Browser",
            Path.home() / ".config/google-chrome",
            Path.home() / ".config/chromium",
            Path.home() / ".config/microsoft-edge",
            Path.home() / ".config/BraveSoftware/Brave-Browser",
        )
        if (
            not resolved.is_dir()
            or resolved.name == "Default"
            or re.fullmatch(r"Profile [0-9]+", resolved.name)
        ):
            raise ValueError
        if any(resolved == root or root in resolved.parents for root in personal):
            raise ValueError
        return resolved
    except (OSError, RuntimeError, ValueError):
        raise ConfigurationError(
            "channel.config", "an existing dedicated browser profile is required"
        ) from None


def _origin(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return ""
    port = parsed.port
    return f"{parsed.scheme}://{parsed.hostname}" + (f":{port}" if port else "")


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _bounded(
    operation: str,
    *,
    response: bool = False,
) -> Callable[
    [Callable[Concatenate[WhatsAppWebChannel, _P], Awaitable[_R]]],
    Callable[Concatenate[WhatsAppWebChannel, _P], Coroutine[Any, Any, _R]],
]:
    """Serialize public operations and prevent SDK exception content escaping."""

    def decorate(
        function: Callable[Concatenate[WhatsAppWebChannel, _P], Awaitable[_R]],
    ) -> Callable[Concatenate[WhatsAppWebChannel, _P], Coroutine[Any, Any, _R]]:
        @wraps(function)
        async def wrapped(self: WhatsAppWebChannel, /, *args: _P.args, **kwargs: _P.kwargs) -> _R:
            try:
                async with self._lock:
                    timeout = (
                        self._limits.response_timeout_seconds
                        if response
                        else self._limits.browser_action_timeout_seconds
                    )
                    async with asyncio.timeout(timeout):
                        return await function(self, *args, **kwargs)
            except asyncio.CancelledError:
                if self._pending or self._uncertain:
                    self._uncertain = True
                raise
            except ChannelError:
                raise
            except Exception:
                raise ChannelError(
                    operation, "WhatsApp operation failed or exceeded its deadline"
                ) from None

        return wrapped

    return decorate


@dataclass
class _Option:
    public: ReplyOption
    owner: str
    node: ElementHandle
    label: str
    prose: str
    dialog: ElementHandle | None = None


class WhatsAppWebChannel:
    """One exclusively owned persistent browser and one attested conversation."""

    def __init__(
        self,
        config: WhatsAppChannelConfig,
        limits: ExecutionLimits,
        *,
        browser_profile: Path,
        recipient: str | None = None,
        headed: bool = False,
        action_authorizations: list[dict[str, Any]] | None = None,
        website_explorer: Callable[[NativeChatBroker], Awaitable[None]] | None = None,
        discovery_only: bool = False,
    ) -> None:
        self._config = config
        self.resolved_config = config
        self._limits = limits
        self._profile = validate_browser_profile(browser_profile)
        self._recipient = resolve_recipient(config, recipient)
        self._headed = headed
        self._authorizations = action_authorizations or []
        self._website_explorer = website_explorer
        self._discovery_only = discovery_only
        self._playwright: Playwright | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._root: ElementHandle | None = None
        self._profile_lock: int | None = None
        self._lock = asyncio.Lock()
        self._closing: asyncio.Task[None] | None = None
        self._started = False
        self._has_submitted = False
        self._dispatch_guard: JSHandle | None = None
        self._contact_proof: JSHandle | None = None
        self._native_list_proof: JSHandle | None = None
        self._pending = False
        self._uncertain = False
        self._response_bubble_attached = False
        self._seen: set[str] = set()
        self._emitted: dict[str, tuple[str, str]] = {}
        self._last_tail: str | None = None
        self._rows: list[dict[str, str]] = []
        self._sequence = 0
        self._options: dict[str, _Option] = {}
        self._option_revision: tuple[str, str, str] | None = None
        self._response_deadline: float | None = None

    @property
    def current_url(self) -> str | None:
        return _origin(self._page.url) if self._page is not None else None

    @property
    def response_bubble_attached(self) -> bool:
        return self._response_bubble_attached

    def _acquire_profile(self) -> None:
        descriptor: int | None = None
        try:
            descriptor = os.open(
                self._profile / ".syntheticcx-whatsapp.lock",
                os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
                0o600,
            )
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._profile_lock = descriptor
        except OSError:
            if descriptor is not None:
                os.close(descriptor)
            raise ChannelError(
                "channel.state", "dedicated browser profile is unavailable or already in use"
            ) from None

    async def start(self) -> None:
        try:
            async with self._lock:
                if self._started or self._closing is not None:
                    raise ChannelError("channel.state", "channel cannot be started again")
                # Discovery is bounded by its owner; runtime/replay may execute
                # multiple individually bounded steps before binding the chat.
                timeout = (
                    None
                    if self._website_explorer
                    else self._limits.browser_action_timeout_seconds
                    * (len(self._config.navigation_steps) + 1)
                )
                async with asyncio.timeout(timeout):
                    self._acquire_profile()
                    self._playwright = await async_playwright().start()
                    self._context = await self._playwright.chromium.launch_persistent_context(
                        str(self._profile),
                        headless=not self._headed,
                        accept_downloads=False,
                    )
                    self._context.set_default_timeout(
                        self._limits.browser_action_timeout_seconds * 1000
                    )
                    self._page = await self._context.new_page()
                    await self._page.goto(str(self._config.url), wait_until="domcontentloaded")
                    await self._handoff()
                    await self._bind()
                    self._started = True
        except BaseException as exc:
            await self.close()
            if isinstance(
                exc, (asyncio.CancelledError, ChannelError, ConfigurationError, ModelError)
            ):
                raise
            raise ChannelError(
                "channel.state", "WhatsApp startup failed or exceeded its deadline"
            ) from None

    async def _handoff(self) -> None:
        assert self._context is not None and self._page is not None
        website = self._page
        broker = NativeChatBroker(
            str(self._config.url),
            self._limits,
            page=website,
            website_recipient=self._recipient,
        )
        broker.discovery_only = self._discovery_only
        initial_pages = set(self._context.pages)
        handoff_urls: dict[Page, str] = {}

        def remember_handoff(request: Request) -> None:
            # Capture the observed main-frame destination before the app clears
            # its phone query via history.replaceState. Never construct a URL
            # from the configured number or borrow a restored account tab.
            if not request.is_navigation_request():
                return
            destination = whatsapp_destination(request.url)
            if destination is None or destination[0] != "web":
                return
            try:
                frame = request.frame
                page = frame.page
                if frame != page.main_frame or (page is not website and page in initial_pages):
                    return
                handoff_urls[page] = request.url
            except Exception:
                # Requests without an available frame cannot attest a handoff.
                return

        self._context.on("request", remember_handoff)
        try:
            await broker.start()
            navigator = broker.website_navigator
            if self._website_explorer is not None:
                await self._website_explorer(broker)
                if not navigator.steps:
                    raise ChannelError("channel.ownership", "website entry path is unavailable")
                self.resolved_config = self._config.model_copy(
                    update={"navigation_steps": list(navigator.steps)}
                )
            else:
                if not self._config.navigation_steps:
                    raise ChannelError(
                        "channel.ownership", "an exact verified website entry path is required"
                    )
                await navigator.replay(self._config.navigation_steps)
            if not await navigator.terminal():
                raise ChannelError(
                    "channel.ownership",
                    "exact website path did not reach recipient-attested WhatsApp",
                )
            # A website-created handoff target can render the account DOM while
            # dropping native input (observed with the live Claro popup chain).
            # Transfer the EXACT attested URL to a fresh context-owned tab before
            # binding or touching a draft. Do not construct a recipient URL or
            # treat this URL attestation as proof of the displayed conversation.
            handoff = navigator.page
            handoff_url = handoff_urls.get(handoff, handoff.url)
            if whatsapp_destination(handoff_url) != ("web", self._recipient):
                raise ChannelError(
                    "channel.ownership", "observed WhatsApp handoff URL lacks matching recipient"
                )
            self._page = await self._context.new_page()
            await handoff.close()
            await self._page.goto(handoff_url, wait_until="domcontentloaded")
        finally:
            self._context.remove_listener("request", remember_handoff)
            await broker.close()
            if self._page is not website:
                for page in self._context.pages:
                    if page is not self._page and not page.is_closed():
                        await page.close()

    async def _bind(self) -> None:
        assert self._page is not None
        stage = "conversation"
        try:
            async with asyncio.timeout(self._limits.browser_action_timeout_seconds):
                root = self._page.locator("#main")
                await root.wait_for(state="visible", timeout=0)
                stage = "recipient evidence"
                # Hydration may expose numeric metadata or the exact contact-info
                # opener. Opaque IDs also need a freshly opened live contact proof;
                # conflicting or unsupported evidence is never replaced by it.
                evidence = await self._page.wait_for_function(
                    r"""() => {
                      if (location.protocol !== 'https:' ||
                          location.hostname !== 'web.whatsapp.com' ||
                          (location.port && location.port !== '443')) return true;
                      const roots = document.querySelectorAll('#main');
                      if (roots.length !== 1) return roots.length > 1;
                      const root = roots[0];
                      if ([...root.querySelectorAll('header [title],header [aria-label]')]
                        .some(n => [
                          n.getAttribute('title') || '', n.getAttribute('aria-label') || ''
                        ].some(t => /^\+?[0-9][0-9 ().-]{5,}$/.test(t))) ||
                        root.querySelector('[data-id]') !== null) return true;
                      const openers = root.querySelectorAll(
                        'header [data-testid="conversation-info-header"][role="button"]'
                      );
                      if (openers.length !== 1) return openers.length > 1;
                      const titles = openers[0].querySelectorAll(
                        '[data-testid="conversation-info-header-chat-title"]'
                      );
                      if (titles.length !== 1) return titles.length > 1;
                      return !!titles[0].textContent.trim() && [openers[0], titles[0]].every(n =>
                        n.getClientRects().length > 0 &&
                        getComputedStyle(n).visibility !== 'hidden' &&
                        !n.closest('[hidden],[aria-hidden="true"],[aria-disabled="true"],[disabled]'));
                    }""",
                    timeout=0,
                )
                await evidence.dispose()
                if await root.count() != 1:
                    raise ChannelError("channel.ownership", "WhatsApp conversation is ambiguous")
                self._root = await root.element_handle(timeout=0)
                needs_proof = await self._root.evaluate(
                    "(root, recipient) => {"
                    + _IDENTITY_JS
                    + r"""
                      const nodes = [...root.querySelectorAll('[data-id]')];
                      const identities = nodes.map(identity);
                      const phones = [...root.querySelectorAll(
                        'header [title],header [aria-label]')]
                        .flatMap(n => [n.getAttribute('title') || '',
                          n.getAttribute('aria-label') || ''])
                        .filter(t => /^\+?[0-9][0-9 ().-]{5,}$/.test(t))
                        .map(t => t.replace(/[^0-9]/g, ''));
                      if (identities.some(i => !i || i.recipient && i.recipient !== recipient) ||
                          phones.some(p => p !== recipient)) return false;
                      return identities.some(i => !i.recipient) ||
                        (!identities.length && !phones.length);
                    }""",
                    self._recipient,
                )
                if needs_proof:
                    stage = "contact recipient proof"
                    await self._open_contact_proof()
                    stage = "contact info close"
                    await self._close_contact_proof()
                stage = "message history"
                await self._snapshot(baseline=True)
                await self._install_dispatch_guard()
                stage = "text composer"
                while True:
                    composer = await self._composer(allow_pending=True)
                    if composer is not None:
                        try:
                            await self._snapshot(baseline=True)
                            # Website draft text is never an authorized message.
                            await self._clear_composer(composer)
                        finally:
                            await composer.dispose()
                        return
                    await asyncio.sleep(0.025)
                    await self._snapshot(baseline=True)
        except TimeoutError:
            raise ChannelError(
                "channel.state", f"WhatsApp {stage} did not become ready before startup deadline"
            ) from None

    async def _open_contact_proof(self) -> None:
        """Attest the phone and bind this chat through its owned contact-info close."""
        assert self._page is not None and self._root is not None
        openers = await self._root.query_selector_all(_CONTACT_OPENER)
        try:
            if (
                len(openers) != 1
                or not await openers[0].is_visible()
                or not await openers[0].is_enabled()
            ):
                raise ChannelError("channel.ownership", "contact recipient evidence is ambiguous")
            self._contact_proof = await self._root.evaluate_handle(
                "(root, args) => {"
                + _IDENTITY_JS
                + r"""
                  const opener = args.opener;
                  const header = opener.closest('header');
                  const titleSelector = '[data-testid="conversation-info-header-chat-title"]';
                  const titles = [...opener.querySelectorAll(titleSelector)];
                   const title = titles.length === 1 ? titles[0] : null;
                   const panelSelector = '[data-testid="conversation-panel-messages"]';
                   const panels = [...root.querySelectorAll(panelSelector)];
                   const panel = panels.length === 1 ? panels[0] : null;
                  const drawerSelector = '[data-testid="chat-info-drawer"]';
                  const headingSelector = '[data-testid="section-about-and-phone-number"]';
                  const fieldSelector =
                    '[data-testid="container_with_separator"] [data-testid="selectable-text"]';
                   let rejected = false, clicked = false;
                   let closing = false, closed = false, closeControl = null;
                   let anchor = null, anchorId = null;
                  const invalid = () => { rejected = true; return 'invalid'; };
                  let drawer = null, section = null, phone = null, phoneText = null;
                  const fingerprint = () => JSON.stringify([
                    // The adjacent typing/online status is deliberately not identity.
                    title.textContent, title.getAttribute('title'), title.getAttribute('aria-label')
                  ]);
                  const headerContent = title ? fingerprint() : null;
                  const visible = node => node.isConnected && node.getClientRects().length > 0 &&
                    getComputedStyle(node).visibility !== 'hidden' &&
                    !node.closest('[hidden],[aria-hidden="true"]');
                  const base = () => {
                    if (location.protocol !== 'https:' ||
                        location.hostname !== 'web.whatsapp.com' ||
                        (location.port && location.port !== '443') || !root.isConnected ||
                        document.querySelectorAll('#main').length !== 1 ||
                        document.querySelector('#main') !== root || !header || !title ||
                        !header.isConnected || !root.contains(header) ||
                        opener.closest('header') !== header ||
                        !opener.isConnected || !header.contains(opener) ||
                        root.querySelectorAll(
                          'header [data-testid="conversation-info-header"][role="button"]'
                        ).length !== 1 || opener.querySelectorAll(titleSelector).length !== 1 ||
                        opener.querySelector(titleSelector) !== title ||
                        fingerprint() !== headerContent ||
                         !visible(header) || !visible(opener) || !visible(title) ||
                         !panel || !panel.isConnected || !root.contains(panel) ||
                         root.querySelectorAll(panelSelector).length !== 1 ||
                         root.querySelector(panelSelector) !== panel) return false;
                    const phones = [...root.querySelectorAll('header [title],header [aria-label]')]
                      .flatMap(n => [
                        n.getAttribute('title') || '', n.getAttribute('aria-label') || ''
                      ])
                      .filter(t => /^\+?[0-9][0-9 ().-]{5,}$/.test(t))
                      .map(t => t.replace(/[^0-9]/g, ''));
                     const nodes = [...root.querySelectorAll('[data-id]')];
                     if (closed && !anchor && nodes.length) {
                       anchor = nodes[nodes.length - 1]; anchorId = anchor.dataset.id;
                     }
                     if (anchor && (!anchor.isConnected || !panel.contains(anchor) ||
                         anchor.dataset.id !== anchorId)) return false;
                     const ids = nodes.map(n => n.dataset.id);
                     if (ids.length > 500 || new Set(ids).size !== ids.length) return false;
                     for (const node of nodes) {
                       const value = identity(node);
                       if (!value) return false;
                       if (value.recipient) phones.push(value.recipient);
                    }
                    return phones.every(value => value === args.recipient);
                  };
                  const inspect = () => {
                    if (rejected || !base()) return invalid();
                     const drawers = [...document.querySelectorAll(drawerSelector)];
                     if (closed) return drawers.length ? invalid() : 'ready';
                     if (closing && !drawers.length) { closed = true; return 'ready'; }
                    if (!clicked) {
                      if (drawers.length) return invalid();
                      return 'pending';
                    }
                    if (!drawers.length && !drawer) return 'pending';
                    if (drawers.length !== 1 || (drawer && drawers[0] !== drawer))
                      return invalid();
                    const currentDrawer = drawers[0];
                    const headings = [...currentDrawer.querySelectorAll(headingSelector)];
                    if (!headings.length && !section) return 'pending';
                    if (headings.length !== 1) return invalid();
                    const currentSection = headings[0].closest('section');
                    if (!currentSection || !currentDrawer.contains(currentSection) ||
                        (section && currentSection !== section)) return invalid();
                    // Scope to the heading's section, not duplicate top-card phone
                    // text or unrelated registration numbers elsewhere in the drawer.
                    const fields = [...currentSection.querySelectorAll(fieldSelector)];
                    if (!fields.length && !phone) return 'pending';
                    if (fields.length !== 1) return invalid();
                    const currentPhone = fields[0];
                    const text = currentPhone.textContent.trim();
                    if (!text && !phone) return 'pending';
                    if (!/^\+?[0-9][0-9 ().-]{5,}$/.test(text) ||
                        text.replace(/[^0-9]/g, '') !== args.recipient ||
                        (phone && (currentPhone !== phone || text !== phoneText)))
                      return invalid();
                     if (!closing && (!visible(currentDrawer) || !visible(currentPhone)))
                      return phone ? invalid() : 'pending';
                    // Pin the complete proof atomically, not a transient loading
                    // drawer/section that React may replace before the phone is ready.
                    drawer = currentDrawer;
                    section = currentSection;
                    phone = currentPhone;
                     phoneText = text;
                     if (!anchor) {
                       const nodes = [...root.querySelectorAll('[data-id]')];
                       anchor = nodes[nodes.length - 1] || null;
                       anchorId = anchor?.dataset.id || null;
                     }
                    return 'ready';
                  };
                  const consume = records => {
                    for (const record of records) {
                       const identities = [root, header, opener, title, panel, anchor,
                         drawer, section, phone]
                         .filter(Boolean);
                       if (record.type === 'childList') {
                         const removed = [...record.removedNodes];
                         const contains = (n, node) => node && (n === node || n.contains(node));
                         if (removed.some(n => [root, header, opener, title, panel, anchor]
                             .some(node => contains(n, node)))) rejected = true;
                         const removesDrawer = removed.some(n => contains(n, drawer));
                         if ((!closing || !removesDrawer) && removed.some(n =>
                             [drawer, section, phone].some(node => contains(n, node))))
                           rejected = true;
                       }
                      if ((record.type === 'childList' || record.type === 'characterData') &&
                          ((title && title.contains(record.target)) ||
                            (phone && phone.contains(record.target)))) rejected = true;
                       if (record.type === 'attributes' &&
                           ((record.target === anchor && record.attributeName === 'data-id') ||
                            (title && title.contains(record.target) &&
                            ['title', 'aria-label', 'data-testid', 'role']
                              .includes(record.attributeName)) ||
                             identities.includes(record.target) &&
                             ['data-testid', 'hidden', 'aria-hidden']
                               .includes(record.attributeName) &&
                             !(closing && record.target === drawer &&
                               ['hidden', 'aria-hidden'].includes(record.attributeName))))
                        rejected = true;
                      if (drawer && record.type === 'childList' &&
                          [...record.addedNodes].some(n => n.nodeType === 1 &&
                            (n.matches(drawerSelector) || n.querySelector(drawerSelector))))
                        rejected = true;
                      // About/status rendering is not phone identity. New phone
                      // fields still latch ambiguity, even if removed in this turn.
                      if (phone && record.type === 'childList' &&
                          section.contains(record.target) &&
                          [...record.addedNodes].some(n => n.nodeType === 1 &&
                            (n.matches(fieldSelector) || n.querySelector(fieldSelector) ||
                              record.target.closest('[data-testid="container_with_separator"]') &&
                              (n.matches('[data-testid="selectable-text"]') ||
                                n.querySelector('[data-testid="selectable-text"]')))))
                        rejected = true;
                    }
                  };
                  const observer = new MutationObserver(records => {
                    consume(records);
                    inspect();
                  });
                  observer.observe(document, {
                    subtree: true, childList: true, characterData: true, attributes: true,
                    attributeFilter: ['title', 'aria-label', 'data-testid', 'role', 'data-id',
                      'hidden', 'aria-hidden', 'class', 'style']
                  });
                  const capture = () => {
                    consume(observer.takeRecords());
                    return inspect();
                  };
                  const openingGuard = event => {
                    if (clicked || !opener.contains(event.target)) return;
                    if (capture() !== 'pending') {
                      event.preventDefault();
                      event.stopImmediatePropagation();
                      return;
                    }
                    clicked = true;
                    window.removeEventListener('click', openingGuard, true);
                  };
                   window.addEventListener('click', openingGuard, true);
                   // An empty chat has no history anchor yet. Do not allow a
                   // same-named empty sidebar conversation to inherit its proof.
                   for (const type of ['pointerdown', 'keydown'])
                     window.addEventListener(type, event => {
                       if (closed && !anchor && !root.contains(event.target)) rejected = true;
                     }, true);
                   const closingGuard = event => {
                     if (!closeControl || closing || closed ||
                         !closeControl.contains(event.target)) return;
                     if (!event.isTrusted || capture() !== 'ready') {
                       event.preventDefault(); event.stopImmediatePropagation(); return;
                     }
                     closing = true;
                     window.removeEventListener('click', closingGuard, true);
                   };
                   return {capture, valid: () => capture() === 'ready',
                     isClosed: () => capture() === 'ready' && closed,
                     armClose: control => {
                       if (capture() !== 'ready' || closing || closed) return false;
                       const controls = [...drawer.querySelectorAll(
                         'button[aria-label],[role="button"][aria-label]')]
                         .filter(n => /^(close|fechar)$/i.test(
                           n.getAttribute('aria-label').trim()));
                       if (controls.length !== 1 || controls[0] !== control || !visible(control) ||
                           control.closest('[disabled],[aria-disabled="true"]')) return false;
                       closeControl = control;
                       window.addEventListener('click', closingGuard, true);
                       return true;
                     }};
                }""",
                {"opener": openers[0], "recipient": self._recipient},
            )
            if await self._contact_proof.evaluate("proof => proof.capture()") != "pending":
                raise ChannelError("channel.ownership", "contact recipient evidence is unavailable")
            await openers[0].click(no_wait_after=True)
            ready = await self._page.wait_for_function(
                "proof => proof.capture() !== 'pending'",
                arg=self._contact_proof,
                timeout=0,
            )
            await ready.dispose()
            if not await self._contact_proof.evaluate("proof => proof.valid()"):
                raise ChannelError("channel.ownership", "contact recipient evidence is unavailable")
        finally:
            for opener in openers:
                await opener.dispose()

    async def _close_contact_proof(self) -> None:
        """Close only the verified drawer; retain a continuously guarded chat binding."""
        assert self._page is not None and self._contact_proof is not None
        controls = await self._page.locator(
            '[data-testid="chat-info-drawer"] button[aria-label],'
            '[data-testid="chat-info-drawer"] [role="button"][aria-label]'
        ).element_handles()
        try:
            closers = [
                node
                for node in controls
                if (await node.get_attribute("aria-label") or "").strip().casefold()
                in {"close", "fechar"}
            ]
            if len(closers) != 1 or not await self._contact_proof.evaluate(
                "(proof, control) => proof.armClose(control)", closers[0]
            ):
                raise ChannelError(
                    "channel.ownership", "verified contact info cannot be safely closed"
                )
            await closers[0].click(no_wait_after=True)
            closed = await self._page.wait_for_function(
                "proof => proof.isClosed() || !proof.valid()",
                arg=self._contact_proof,
                timeout=0,
            )
            await closed.dispose()
            if not await self._contact_proof.evaluate("proof => proof.isClosed()"):
                raise ChannelError(
                    "channel.ownership", "recipient proof changed while closing info"
                )
        finally:
            for node in controls:
                await node.dispose()

    async def _clear_composer(self, composer: ElementHandle) -> None:
        """Clear the editor through native keys, including its framework-owned draft state."""
        await self._guard()
        await composer.press("ControlOrMeta+A")
        await self._guard()
        await composer.press("Backspace")
        await self._guard()
        if (await composer.inner_text()).strip():
            raise ChannelError("channel.controls", "existing draft could not be cleared safely")

    async def _install_dispatch_guard(self) -> None:
        assert self._root is not None
        self._dispatch_guard = await self._root.evaluate_handle(
            "(root, args) => {"
            + _IDENTITY_JS
            + r"""
              let rejected = false;
              let ownedDraft = null;
              const clearDraft = () => {
                if (ownedDraft) ownedDraft.textContent = '';
              };
              const valid = () => {
                if (!root.isConnected || document.querySelectorAll('#main').length !== 1 ||
                    document.querySelector('#main') !== root) return false;
                 const nodes = [...root.querySelectorAll('[data-id]')];
                 const ids = nodes.map(n => n.dataset.id);
                 if (ids.length > 500 || new Set(ids).size !== ids.length) return false;
                const phones = new Set([...root.querySelectorAll(
                  'header [title],header [aria-label]')]
                  .flatMap(n => [n.getAttribute('title') || '', n.getAttribute('aria-label') || ''])
                  .filter(t => /^\+?[0-9][0-9 ().-]{5,}$/.test(t))
                  .map(t => t.replace(/[^0-9]/g, '')));
                 for (const node of nodes) {
                   const value = identity(node);
                   if (!value || !value.recipient && !args.proof) return false;
                   if (value.recipient) phones.add(value.recipient);
                }
                if (args.proof) {
                  if (!args.proof.valid()) return false;
                  phones.add(args.recipient);
                }
                return phones.size === 1 && phones.has(args.recipient);
              };
              const guard = event => {
                if (valid() && !rejected) {
                  if (['beforeinput', 'input'].includes(event.type) &&
                      root.contains(event.target) &&
                      event.target.matches('[contenteditable=true]')) ownedDraft = event.target;
                  return;
                }
                rejected = true;
                event.preventDefault();
                event.stopImmediatePropagation();
                clearDraft();
                // Do not allow an input event to save a private draft in another chat.
                if (event.type === 'input' && root.contains(event.target) &&
                    event.target.matches('[contenteditable=true]')) event.target.textContent = '';
              };
              for (const kind of ['beforeinput', 'input', 'keydown', 'click'])
                window.addEventListener(kind, guard, true);
              return {valid: () => {
                if (valid() && !rejected) return true;
                rejected = true;
                clearDraft();
                return false;
              }};
            }""",
            {"recipient": self._recipient, "proof": self._contact_proof},
        )

    async def _guard(self) -> None:
        if self._page is None or self._root is None or self._page.is_closed():
            raise ChannelError("channel.ownership", "bound conversation is unavailable")
        parsed = urlsplit(self._page.url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "web.whatsapp.com"
            or parsed.port not in {None, 443}
        ):
            raise ChannelError("channel.ownership", "bound conversation destination changed")
        valid = await self._root.evaluate(
            """node => node.isConnected && document.querySelectorAll('#main').length===1 &&
              document.querySelector('#main')===node"""
        )
        if not valid:
            raise ChannelError("channel.ownership", "bound conversation identity changed")
        if self._dispatch_guard is not None and not await self._dispatch_guard.evaluate(
            "guard => guard.valid()"
        ):
            raise ChannelError("channel.ownership", "recipient-bound dispatch was revoked")
        if self._contact_proof is not None and not await self._contact_proof.evaluate(
            "proof => proof.valid()"
        ):
            raise ChannelError("channel.ownership", "contact recipient proof was revoked")

    async def _read_snapshot(self) -> dict[str, Any]:
        assert self._root is not None
        result: dict[str, Any] = await self._root.evaluate(
            "(root,args) => {"
            + _IDENTITY_JS
            + """
           const renderedText = node => {
             const read = n => {
               if (n.nodeType === Node.TEXT_NODE) return n.nodeValue;
               if (n.nodeType !== Node.ELEMENT_NODE) return '';
               if (n.matches('[hidden],[aria-hidden="true"]')) return '';
               if (n.tagName === 'IMG') {
                 const plain = n.getAttribute('data-plain-text') || '';
                 return n.matches('img.emoji') && plain.length <= 32 &&
                   plain === n.getAttribute('alt') ? plain : '';
               }
               if (n.tagName === 'BR') return '\\n';
               const text = [...n.childNodes].map(read).join('');
               return /^(DIV|P|LI)$/.test(n.tagName) ? text + '\\n' : text;
             };
             return read(node).trim();
           };
           const all=[...root.querySelectorAll('[data-id]')];
           if(all.some(n => !identity(n))) return {invalidIdentity:true};
           if(all.some(n => !identity(n).recipient) && !args.proof) return {invalidProof:true};
           if(all.some(n => identity(n).pending)) return {pending:true};
           const virtualized=all.filter(n => identity(n).virtualized).map(n => n.dataset.id);
           const excluded=all.filter(n => identity(n).system || identity(n).virtualized)
             .map(n => n.dataset.id);
           const nodes=all.filter(n => !identity(n).system && !identity(n).virtualized);
           if(nodes.length>500) return {overflow:true};
           const rows=nodes.map(n => ({raw:n.dataset.id,actor:identity(n).actor,
             failed:n.querySelector('[data-testid="fail-container"]') ? 'true' : 'false',
             text:[...n.querySelectorAll('.selectable-text')]
               .filter(s=>!s.parentElement.closest('.selectable-text') &&
                 !s.closest('[data-testid="quoted-message"]'))
               .map(renderedText).join('\\n').trim(),
             fallback:[...n.querySelectorAll('button,[role=button]')]
               .filter(s=>!s.closest('[data-testid="quoted-message"]'))
              .map(s=>s.innerText.trim()).filter(Boolean).join('\\n')}));
          const phones=[...root.querySelectorAll('header [title],header [aria-label]')]
            .flatMap(n=>[n.getAttribute('title')||'',n.getAttribute('aria-label')||''])
            .filter(t=>/^\\+?[0-9][0-9 ().-]{5,}$/.test(t)).map(t=>t.replace(/[^0-9]/g,''));
          if(args.proof) {
            if(!args.proof.valid()) return {invalidProof:true};
            phones.push(args.recipient);
          }
           return {rows,phones,virtualized,excluded};
        }""",
            {"recipient": self._recipient, "proof": self._contact_proof},
        )
        return result

    async def _snapshot(self, *, baseline: bool = False) -> list[dict[str, str]]:
        deadline = time.monotonic() + self._limits.browser_action_timeout_seconds
        while True:
            await self._guard()
            result = await self._read_snapshot()
            if not result.get("pending"):
                break
            if time.monotonic() >= deadline:
                raise ChannelError(
                    "channel.reader_evidence", "message direction did not become ready"
                )
            await asyncio.sleep(0.025)
        if result.get("invalidProof"):
            raise ChannelError("channel.ownership", "contact recipient proof was revoked")
        if result.get("invalidIdentity"):
            raise ChannelError(
                "channel.reader_evidence", "unsupported conversation message identity"
            )
        if result.get("overflow"):
            raise ChannelError("channel.reader", "conversation inspection limit exceeded")
        rows: list[dict[str, str]] = result["rows"]
        ids = [row["raw"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ChannelError("channel.reader_evidence", "message identities are ambiguous")
        phones = set(result["phones"])
        for row in rows:
            identity = _MESSAGE_ID.fullmatch(row["raw"])
            if identity is None:
                if self._contact_proof is None or row["actor"] not in {"customer", "assistant"}:
                    raise ChannelError(
                        "channel.reader_evidence", "unsupported conversation message identity"
                    )
            else:
                phones.add(identity[2])
            if len(row["text"]) > 32768:
                raise ChannelError("channel.reader", "message inspection limit exceeded")
        if phones != {self._recipient}:
            raise ChannelError(
                "channel.ownership", "conversation lacks unambiguous matching recipient evidence"
            )
        if not baseline and any(raw not in self._seen for raw in result["virtualized"]):
            raise ChannelError(
                "channel.reader_evidence",
                "unobserved message was virtualized before it could be read",
            )
        if not baseline and self._last_tail is not None:
            if self._last_tail not in ids:
                raise ChannelError("channel.reader_evidence", "message history identity was lost")
            boundary = ids.index(self._last_tail)
            if any(row["raw"] not in self._seen for row in rows[:boundary]):
                raise ChannelError(
                    "channel.reader_evidence", "historical messages cannot become fresh evidence"
                )
        if (
            not baseline
            and not self._uncertain
            and any(row["actor"] == "customer" and row["raw"] not in self._seen for row in rows)
        ):
            raise ChannelError(
                "channel.reader_evidence", "unowned outgoing message changed conversation evidence"
            )
        for row in rows:
            if row["raw"] in self._emitted and row["text"] != self._emitted[row["raw"]][1]:
                raise ChannelError(
                    "channel.reader_evidence", "acknowledged message content changed"
                )
        if baseline:
            # Retain the IDs, never invented actors/text, of pre-run placeholders
            # and notifications. Later hydration must not create fresh evidence.
            self._seen.update(result["excluded"])
            self._advance(rows)
        self._rows = rows
        return rows

    def _advance(self, rows: list[dict[str, str]]) -> None:
        self._seen.update(row["raw"] for row in rows)
        if rows:
            self._last_tail = rows[-1]["raw"]

    async def _composer(self, *, allow_pending: bool = False) -> ElementHandle | None:
        await self._guard()
        assert self._page is not None
        handles = await self._page.locator(f"#main {_COMPOSER}").element_handles()
        visible: list[ElementHandle] = []
        try:
            visible = [node for node in handles if await node.is_visible()]
            if len(visible) > 1:
                raise ChannelError("channel.controls", "composer is ambiguous")
            if visible:
                node = visible[0]
                if not await node.is_enabled() or not await node.is_editable():
                    if allow_pending:
                        return None
                    raise ChannelError("channel.controls", "composer is unavailable")
                handles.remove(node)
                return node
            return None
        finally:
            for node in handles:
                await node.dispose()

    async def _invalidate_options(self) -> None:
        old, self._options = self._options, {}
        self._option_revision = None
        for option in old.values():
            await option.node.dispose()
            if option.dialog is not None:
                await option.dialog.dispose()
        if self._native_list_proof is not None:
            await self._native_list_proof.dispose()
            self._native_list_proof = None

    async def _safe_option(
        self, node: ElementHandle, owner: str, dialog: ElementHandle | None = None
    ) -> bool:
        assert self._root is not None
        return bool(
            await node.evaluate(
                """(node,args) => {
          const owner=[...args.root.querySelectorAll('[data-id]')]
            .find(n=>n.dataset.id===args.owner);
           const local=owner && (owner.contains(node) ||
             (args.proof && args.proof.valid(owner,node,args.dialog)) ||
            (args.dialog && args.dialog.contains(node) &&
              (owner.contains(args.dialog) || [...owner.querySelectorAll('[aria-controls]')]
                .some(n=>n.getAttribute('aria-controls')===args.dialog.id))));
           return node.isConnected && local &&
             !node.closest('a[href],form,[data-testid="quoted-message"]') &&
            !node.matches('[type=submit],[type=file]') &&
            !node.querySelector('a[href],input[type=file]');
        }""",
                {
                    "root": self._root,
                    "owner": owner,
                    "dialog": dialog,
                    "proof": self._native_list_proof,
                },
            )
            and await node.is_visible()
            and await node.is_enabled()
        )

    async def _dialog_prose(self, dialog: ElementHandle) -> str:
        # Authorize the selected business choice and descriptive terms, not the
        # transport's Send button or unrelated unselected choices.
        return str(
            await dialog.evaluate("""node => {
          const copy = node.cloneNode(true);
           for (const control of copy.querySelectorAll(
               'button,[role=button],[role=radio],[role=option],'
               + '[data-testid="cell-frame-container"]')) control.remove();
          return copy.textContent.trim();
        }""")
        )

    async def _option_label(self, node: ElementHandle) -> str:
        if self._native_list_proof is not None and await node.get_attribute("role") == "radio":
            return str(
                await node.evaluate("""node => {
              const cell = node.closest('[data-testid="cell-frame-container"]');
              const titles = cell?.querySelectorAll('[data-testid="cell-frame-title"]');
              const label = (node.getAttribute('aria-label') || '').trim();
              return titles?.length === 1 && titles[0].innerText.trim() === label ? label : '';
            }""")
            )
        return str(await node.inner_text()).strip()

    async def _is_native_list_opener(self, node: ElementHandle) -> bool:
        return bool(
            await node.evaluate("""node => node.tagName === 'BUTTON' &&
          node.querySelectorAll('svg title').length === 1 &&
          node.querySelector('svg title').textContent === 'ic-format-list-bulleted' &&
          !!node.innerText.trim() && node.getAttribute('title') === node.innerText.trim()
        """)
        )

    async def _list_dialog(
        self, opener: ElementHandle, source: ElementHandle, owner: dict[str, str]
    ) -> ElementHandle:
        """Open one message-owned list without treating its opener as a submission."""
        assert self._page is not None
        controlled = await opener.get_attribute("aria-controls")
        native = not controlled and await self._is_native_list_opener(opener)
        if not controlled and not native:
            raise ChannelError("channel.options", "list dialog ownership is unsupported")
        label = (await opener.inner_text()).strip()
        existing = await self._page.locator('[role="dialog"],[role="listbox"]').element_handles()
        try:
            if native and existing:
                raise ChannelError("channel.options", "an unowned dialog already exists")
            matches = [
                item
                for item in existing
                if controlled
                and await item.get_attribute("id") == controlled
                and await item.is_visible()
            ]
            if len(matches) > 1:
                raise ChannelError("channel.options", "list dialog is ambiguous")
            if matches:
                retained = await matches[0].evaluate_handle("node=>node")
                assert retained.as_element() is not None
                return retained.as_element()
        finally:
            for item in existing:
                await item.dispose()
        self._uncertain = True
        await opener.click(no_wait_after=True)
        after = await self._snapshot()
        if any(row["raw"] not in self._seen for row in after):
            raise ChannelError(
                "channel.options", "list opening changed conversation evidence; retry prohibited"
            )
        while True:
            candidates = await self._page.locator(
                '[role="dialog"],[role="listbox"]'
            ).element_handles()
            matches = []
            try:
                for item in candidates:
                    if not await item.is_visible():
                        continue
                    if controlled:
                        match = await item.get_attribute("id") == controlled
                    else:
                        match = await item.evaluate(
                            """(node,label) =>
                          node.getAttribute('role') === 'dialog' &&
                          node.querySelectorAll('[data-testid="confirm-popup"]').length === 1 &&
                          node.querySelectorAll('[data-testid="popup-contents"]').length === 1 &&
                          node.querySelectorAll('[data-testid="drawer-title-body"]').length === 1 &&
                          node.querySelector('[data-testid="drawer-title-body"]').textContent.trim()
                            === label && node.querySelector('[role="radio"]') !== null
                        """,
                            label,
                        )
                    if match:
                        matches.append(item)
                if len(matches) > 1:
                    raise ChannelError("channel.options", "list dialog is ambiguous")
                if matches:
                    dialog_handle = await matches[0].evaluate_handle("node=>node")
                    dialog = dialog_handle.as_element()
                    assert dialog is not None
                    break
            finally:
                for item in candidates:
                    await item.dispose()
            await self._guard()
            await asyncio.sleep(0.025)
        if native:
            self._native_list_proof = await source.evaluate_handle(
                """(source,args) => {
              const {root,opener,dialog,label,owner} = args;
              return {valid: (current,node,currentDialog) =>
                source.isConnected && current === source && source.dataset.id === owner &&
                root.isConnected && root.contains(source) && source.contains(opener) &&
                opener.isConnected && opener.getAttribute('title') === label &&
                dialog.isConnected && currentDialog === dialog && dialog.contains(node) &&
                document.querySelectorAll('[role="dialog"],[role="listbox"]').length === 1 &&
                dialog.querySelectorAll('[data-testid="drawer-title-body"]').length === 1 &&
                dialog.querySelector('[data-testid="drawer-title-body"]').textContent.trim()
                  === label
              };
            }""",
                {
                    "root": self._root,
                    "opener": opener,
                    "dialog": dialog,
                    "label": label,
                    "owner": owner["raw"],
                },
            )
        self._uncertain = False
        return dialog

    async def _collect_options(self) -> list[ReplyOption]:
        assert self._page is not None
        rows = await self._snapshot(baseline=not self._has_submitted)
        if not rows or rows[-1]["actor"] != "assistant" or rows[-1]["raw"] not in self._emitted:
            await self._invalidate_options()
            return []
        owner = rows[-1]
        if self._options:
            valid = [
                item.owner == owner["raw"]
                and await self._safe_option(item.node, item.owner, item.dialog)
                and await self._option_label(item.node) == item.label
                for item in self._options.values()
            ]
            dialog = next(iter(self._options.values())).dialog
            dialog_text = await dialog.inner_text() if dialog is not None and all(valid) else ""
            revision = (owner["raw"], owner["fallback"], dialog_text)
            if all(valid) and revision == self._option_revision:
                return [item.public for item in self._options.values()]
            await self._invalidate_options()
        # Resolve by native-held identity without embedding sensitive DOM IDs in selectors.
        bubbles = await self._page.locator("#main [data-id]").element_handles()
        source: ElementHandle | None = None
        try:
            for node in bubbles:
                if await node.get_attribute("data-id") == owner["raw"]:
                    source = node
                    break
            if source is None:
                raise ChannelError("channel.options", "message owner is unavailable")
            nodes = await source.query_selector_all(_BUTTONS)
            dialogs: list[ElementHandle] = []
            try:
                if len(nodes) > 30:
                    raise ChannelError(
                        "channel.options", "message option inspection limit exceeded"
                    )
                supported = []
                for node in nodes:
                    if (await node.inner_text()).strip() and await self._safe_option(
                        node, owner["raw"]
                    ):
                        supported.append(node)
                    else:
                        await node.dispose()
                nodes = supported
                openers = [
                    node
                    for node in nodes
                    if await node.get_attribute("aria-haspopup") in {"dialog", "listbox"}
                    or await self._is_native_list_opener(node)
                ]
                if len(openers) > 1:
                    raise ChannelError("channel.options", "list opener is ambiguous")
                if openers:
                    opener = openers[0]
                    label = (await opener.inner_text()).strip()
                    if (
                        not await self._safe_option(opener, owner["raw"])
                        or eligibility(
                            owner["text"],
                            label,
                            {"safe": True, "visible": True, "enabled": True},
                            dynamic_runtime=not self._discovery_only,
                            authorizations=self._authorizations,
                        )
                        is not None
                    ):
                        raise ChannelError("channel.options", "list opener is unsafe")
                    dialog = await self._list_dialog(opener, source, owner)
                    dialogs = [dialog]
                    for node in nodes:
                        await node.dispose()
                    nodes = await dialog.query_selector_all('[role="option"],[role="radio"]')
                else:
                    dialog = None
                labels = [await self._option_label(node) for node in nodes]
                if len(labels) > 30 or len(labels) != len(set(labels)):
                    raise ChannelError("channel.options", "message options are ambiguous")
                prose = owner["text"]
                if dialog is not None:
                    prose += "\n" + await self._dialog_prose(dialog)
                for node, label in zip(nodes, labels, strict=True):
                    safe = await self._safe_option(node, owner["raw"], dialog)
                    option_prose = prose
                    if self._native_list_proof is not None:
                        option_prose += "\n" + await node.evaluate("""node =>
                          node.closest('[data-testid="cell-frame-container"]')?.innerText || ''
                        """)
                    if (
                        not label
                        or eligibility(
                            option_prose,
                            label,
                            {"safe": safe, "visible": safe, "enabled": safe},
                            labels,
                            dynamic_runtime=not self._discovery_only,
                            authorizations=self._authorizations,
                        )
                        is not None
                    ):
                        continue
                    public = ReplyOption(id=uuid4().hex, label=label)
                    retained_dialog = await dialog.evaluate_handle("node=>node") if dialog else None
                    self._options[public.id] = _Option(
                        public,
                        owner["raw"],
                        node,
                        label,
                        option_prose,
                        retained_dialog.as_element() if retained_dialog else None,
                    )
                retained = {id(item.node) for item in self._options.values()}
                nodes = [node for node in nodes if id(node) not in retained]
                self._option_revision = (
                    owner["raw"],
                    owner["fallback"],
                    await dialog.inner_text() if dialog else "",
                )
                return [item.public for item in self._options.values()]
            finally:
                for node in nodes:
                    await node.dispose()
                for item in dialogs:
                    await item.dispose()
        finally:
            for node in bubbles:
                await node.dispose()

    @_bounded("channel.controls")
    async def observe_controls(self) -> ObservedControls:
        self._require_ready()
        options = await self._collect_options()
        composer = await self._composer()
        text_available = composer is not None and self._native_list_proof is None
        if composer is not None:
            await composer.dispose()
        if not text_available and not options:
            raise ChannelError("channel.controls", "no supported customer controls are available")
        return {"text_available": text_available, "options": options}

    def _require_ready(self) -> None:
        if not self._started or self._closing is not None or self._pending or self._uncertain:
            raise ChannelError(
                "channel.state", "channel is unavailable, pending, or uncertain; retry prohibited"
            )

    async def _pre_dispatch(self) -> list[dict[str, str]]:
        self._require_ready()
        rows = await self._snapshot(baseline=not self._has_submitted)
        if self._discovery_only:
            for row in reversed(rows):
                if row["actor"] == "customer":
                    break
                if portability_flow(row["text"]):
                    raise ChannelError(
                        "channel.controls", "number-portability flows are forbidden in discovery"
                    )
        if any(row["actor"] == "assistant" and row["raw"] not in self._seen for row in rows):
            raise ChannelError(
                "channel.observe", "fresh incoming messages must be observed before dispatch"
            )
        return rows

    async def _echo(self, text: str, previous: set[str]) -> SendReceipt:
        deadline = time.monotonic() + self._limits.echo_observation_seconds
        while True:
            rows = await self._snapshot()
            outgoing = [
                row for row in rows if row["actor"] == "customer" and row["raw"] not in previous
            ]
            matches = [
                row for row in outgoing if normalize_echo(row["text"]) == normalize_echo(text)
            ]
            if len(outgoing) > 1 or len(matches) > 1:
                raise ChannelError(
                    "channel.reader_evidence", "submission evidence is ambiguous; retry prohibited"
                )
            if len(matches) == 1:
                if matches[0].get("failed") == "true":
                    raise ChannelError(
                        "channel.delivery",
                        "WhatsApp reported message was not sent; retry prohibited",
                    )
                echo_index = next(
                    i for i, row in enumerate(rows) if row["raw"] == matches[0]["raw"]
                )
                # The first owned outgoing echo establishes the chronological run
                # boundary even when existing history hydrates asynchronously.
                self._advance(rows[: echo_index + 1])
                self._has_submitted = True
                self._pending = True
                self._uncertain = False
                self._response_bubble_attached = False
                self._response_deadline = time.monotonic() + self._limits.response_timeout_seconds
                return SendReceipt(
                    message_id=uuid4().hex,
                    submitted_at=datetime.now(UTC),
                    evidence={"observed_echo": True, "channel": "whatsapp_web_playwright"},
                )
            if time.monotonic() >= deadline:
                raise ChannelError("channel.state", "submission is uncertain; retry prohibited")
            await asyncio.sleep(0.025)

    @_bounded("channel.state")
    async def send(self, message: str) -> SendReceipt:
        if not message.strip():
            raise ChannelError("channel.state", "customer message cannot be blank")
        if self._native_list_proof is not None:
            raise ChannelError(
                "channel.controls", "text is unavailable while the native list is open"
            )
        rows = await self._pre_dispatch()
        composer = await self._composer()
        if composer is None:
            raise ChannelError("channel.controls", "text composer is unavailable")
        try:
            await self._invalidate_options()
            # Fill dispatches page input handlers: latch uncertainty before any effect.
            self._uncertain = True
            await self._clear_composer(composer)
            # WhatsApp's framework-owned editor can ignore DOM-style fill events.
            # Use native key input, reserving plain Enter for the one submission.
            lines = message.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            for index, line in enumerate(lines):
                await self._guard()
                if index:
                    await composer.press("Shift+Enter")
                if line:
                    await composer.type(line)
            await self._guard()
            await self._snapshot()
            if not await composer.evaluate(
                "(node,root)=>node.isConnected && root.contains(node)", self._root
            ):
                raise ChannelError(
                    "channel.controls", "composer identity changed; retry prohibited"
                )
            await composer.press("Enter", no_wait_after=True)
            return await self._echo(message, {row["raw"] for row in rows})
        finally:
            await composer.dispose()

    @_bounded("channel.select_option")
    async def select_option(self, identifier: str) -> SelectionReceipt:
        rows = await self._pre_dispatch()
        option = self._options.get(identifier)
        if (
            option is None
            or not rows
            or rows[-1]["raw"] != option.owner
            or not await self._safe_option(option.node, option.owner, option.dialog)
            or await self._option_label(option.node) != option.label
        ):
            raise ChannelError("channel.select_option", "option is stale, unsafe, or unavailable")
        revision = (
            rows[-1]["raw"],
            rows[-1]["fallback"],
            await option.dialog.inner_text() if option.dialog is not None else "",
        )
        if revision != self._option_revision:
            raise ChannelError(
                "channel.select_option", "option revision changed; observe current controls again"
            )
        if (
            eligibility(
                option.prose,
                option.label,
                {"safe": True, "visible": True, "enabled": True},
                dynamic_runtime=not self._discovery_only,
                authorizations=self._authorizations,
            )
            is not None
        ):
            raise ChannelError("channel.select_option", "option is not authorized")
        source_id = self._emitted[option.owner][0]
        label = option.label
        expected_echo = label
        self._uncertain = True
        await option.node.click(no_wait_after=True)
        await self._guard()
        if option.dialog is not None:
            assert self._page is not None
            after = await self._snapshot()
            previous = {row["raw"] for row in rows}
            dispatched = any(
                row["actor"] == "customer" and row["raw"] not in previous for row in after
            )
            radio_selected = (
                await option.node.get_attribute("role") == "radio"
                and await option.node.get_attribute("aria-checked") == "true"
            )
            selected_prose = ""
            if not dispatched and self._native_list_proof is not None:
                ready = await self._page.wait_for_function(
                    """dialog => {
                  if (!dialog.isConnected) return true;
                  const checked = dialog.querySelectorAll('[role="radio"][aria-checked="true"]');
                  if (checked.length > 1) return true;
                  const sends = [...dialog.querySelectorAll(
                    '[data-testid="list-msg-modal-button"],button[aria-label]')]
                    .filter(n => n.matches('[data-testid="list-msg-modal-button"]') &&
                      n.querySelector('[data-icon="wds-ic-send-filled"]') ||
                      /^(send|enviar)$/i.test(n.getAttribute('aria-label') || ''));
                  return checked.length === 1 && sends.some(n =>
                    n.getClientRects().length > 0 &&
                    !n.closest('[disabled],[aria-disabled="true"]'));
                }""",
                    arg=option.dialog,
                    polling=25,
                    timeout=0,
                )
                await ready.dispose()
                await self._guard()
                after = await self._snapshot()
                if any(row["actor"] == "customer" and row["raw"] not in previous for row in after):
                    raise ChannelError(
                        "channel.select_option",
                        "list dispatched before confirmation; retry prohibited",
                    )
                # Current Web replaces its radio after selection. Re-attest the
                # unique checked control inside this same owned dialog, not the
                # stale pre-click handle or an arbitrary same-labelled element.
                checked = await option.dialog.query_selector_all(
                    '[role="radio"][aria-checked="true"]'
                )
                try:
                    radio_selected = (
                        len(checked) == 1
                        and await self._option_label(checked[0]) == option.label
                        and await self._safe_option(checked[0], option.owner, option.dialog)
                    )
                    if radio_selected:
                        selected_prose = await checked[0].evaluate("""node =>
                          node.closest('[data-testid="cell-frame-container"]')?.innerText || ''
                        """)
                        expected_echo = selected_prose.strip()
                        if not expected_echo:
                            raise ChannelError(
                                "channel.select_option",
                                "selected list response text is unavailable",
                            )
                finally:
                    for node in checked:
                        await node.dispose()
                if not radio_selected:
                    raise ChannelError(
                        "channel.select_option",
                        "native list selection is unverified; retry prohibited",
                    )
            # Only a natively checked radio authorizes a separate final Send.
            # A direct-action option is never followed by a second dispatch.
            needs_confirmation = (
                not dispatched
                and radio_selected
                and await option.dialog.evaluate("node=>node.isConnected")
                and await option.dialog.is_visible()
            )
            if needs_confirmation:
                if any(
                    row["actor"] == "assistant" and row["raw"] not in self._seen for row in after
                ):
                    raise ChannelError(
                        "channel.select_option",
                        "conversation changed before list confirmation; retry prohibited",
                    )
                controls = await option.dialog.query_selector_all(_BUTTONS)
                try:
                    sends = [
                        node
                        for node in controls
                        if (
                            re.fullmatch(
                                r"send|enviar",
                                (await node.inner_text()).strip()
                                or (await node.get_attribute("aria-label") or "").strip(),
                                re.IGNORECASE,
                            )
                            or self._native_list_proof is not None
                            and await node.evaluate("""n =>
                          n.matches('[data-testid="list-msg-modal-button"]') &&
                          n.querySelector('[data-icon="wds-ic-send-filled"]') !== null
                        """)
                        )
                        and await node.is_visible()
                        and await node.is_enabled()
                    ]
                    if len(sends) != 1 or not await self._safe_option(
                        sends[0], option.owner, option.dialog
                    ):
                        raise ChannelError(
                            "channel.select_option",
                            "list confirmation is unsupported; retry prohibited",
                        )
                    confirmation_prose = (
                        option.prose
                        + "\n"
                        + selected_prose
                        + "\n"
                        + await self._dialog_prose(option.dialog)
                    )
                    if (
                        eligibility(
                            confirmation_prose,
                            option.label,
                            {"safe": True, "visible": True, "enabled": True},
                            dynamic_runtime=not self._discovery_only,
                            authorizations=self._authorizations,
                        )
                        is not None
                    ):
                        raise ChannelError(
                            "channel.select_option",
                            "list confirmation is not authorized; retry prohibited",
                        )
                    await sends[0].click(no_wait_after=True)
                finally:
                    for node in controls:
                        await node.dispose()
        receipt = await self._echo(expected_echo, {row["raw"] for row in rows})
        await self._invalidate_options()
        return SelectionReceipt(
            **receipt.model_dump(),
            selected_option_id=identifier,
            selected_label=label,
            source_message_id=source_id,
        )

    @_bounded("channel.observe", response=True)
    async def observe_initial(self) -> CompletedResponse | None:
        self._require_ready()
        if not self._has_submitted:
            # Before an owned submission, history hydration and unsolicited input
            # are indistinguishable. Neither can establish current-run goal evidence.
            await self._snapshot(baseline=True)
            return None
        return await self._observe_fresh()

    @_bounded("channel.observe", response=True)
    async def observe_pending(self) -> CompletedResponse | None:
        self._require_ready()
        if not self._has_submitted:
            return None
        return await self._observe_fresh()

    async def _observe_fresh(self) -> CompletedResponse | None:
        rows = await self._snapshot()
        if not any(row["actor"] == "assistant" and row["raw"] not in self._seen for row in rows):
            return None
        return await self._receive_batch(time.monotonic() + self._limits.response_timeout_seconds)

    @_bounded("channel.observe", response=True)
    async def receive(self) -> CompletedResponse:
        if not self._started or not self._pending or self._uncertain or self._closing is not None:
            raise ChannelError("channel.state", "no certain submission is awaiting a response")
        assert self._response_deadline is not None
        return await self._receive_batch(self._response_deadline)

    async def _receive_batch(self, deadline: float) -> CompletedResponse:
        started = datetime.now(UTC)
        changed = time.monotonic()
        last: list[tuple[str, str]] = []
        while True:
            rows = await self._snapshot()
            fresh = [
                row for row in rows if row["actor"] == "assistant" and row["raw"] not in self._seen
            ]
            signature = [(row["raw"], row["text"] or row["fallback"]) for row in fresh]
            if signature != last:
                changed = time.monotonic()
                last = signature
            if fresh:
                self._response_bubble_attached = True
                if any(not content for _, content in signature):
                    raise ChannelError(
                        "channel.reader", "incoming message requires an unsupported interaction"
                    )
                if time.monotonic() - changed >= self._config.quiet_interval_seconds:
                    messages: list[Message] = []
                    for row, (_, content) in zip(fresh, signature, strict=True):
                        public_id = uuid4().hex
                        messages.append(
                            Message(
                                id=public_id,
                                sequence=self._sequence,
                                actor="assistant",
                                content=content,
                                observed_at=datetime.now(UTC),
                            )
                        )
                        self._sequence += 1
                        self._emitted[row["raw"]] = (public_id, row["text"])
                    self._advance(rows)
                    self._pending = False
                    self._response_deadline = None
                    # Deliver acknowledged text before probing optional controls.
                    # Unsupported menus must not discard partial conversation evidence;
                    # observe_controls publishes current choices for the next decision.
                    return CompletedResponse(
                        messages=messages,
                        started_at=started,
                        completed_at=datetime.now(UTC),
                        completion_strategy="quiet_interval",
                    )
            if time.monotonic() >= deadline:
                raise ChannelError("channel.observe", "no settled fresh response before deadline")
            await asyncio.sleep(0.025)

    async def close(self) -> None:
        if self._closing is None:
            self._closing = asyncio.create_task(self._close_owned())
        await asyncio.shield(self._closing)

    async def _close_owned(self) -> None:
        failed = False
        try:
            if self._context is not None:
                try:
                    async with asyncio.timeout(self._limits.close_timeout_seconds * 0.7):
                        await self._context.close()
                except Exception:
                    failed = True
            if self._playwright is not None:
                try:
                    async with asyncio.timeout(self._limits.close_timeout_seconds * 0.3):
                        await self._playwright.stop()
                except Exception:
                    failed = True
        finally:
            self._context = None
            self._playwright = None
            self._page = None
            self._root = None
            self._dispatch_guard = None
            self._contact_proof = None
            self._native_list_proof = None
            self._options.clear()
            self._rows.clear()
            self._seen.clear()
            self._emitted.clear()
            if self._profile_lock is not None:
                descriptor, self._profile_lock = self._profile_lock, None
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                    os.close(descriptor)
                except OSError:
                    failed = True
        if failed:
            raise ChannelError(
                "channel.state", "owned browser cleanup exceeded its deadline or failed"
            )

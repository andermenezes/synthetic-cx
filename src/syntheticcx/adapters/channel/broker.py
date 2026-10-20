"""Native browser broker: owned observations, opaque proposals and one-use actions.

The public catalog is data, never browser authority. Only this module retains
native nodes and resolves the controller's proposals inside an independently
attested root. Replay uses exactly the same checks, without a model or autoplay.
"""

from __future__ import annotations

import asyncio
import re
import secrets
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, TypeVar, cast

from playwright.async_api import ElementHandle, Frame, Locator, Page, async_playwright
from playwright.async_api import Error as PlaywrightError

from syntheticcx.adapters.channel.bindings import NativeControlBinding, NativeOptionConfig
from syntheticcx.adapters.channel.cursor import Bubble, MessageCursor
from syntheticcx.adapters.channel.options import ReplyOptions, normalized_label
from syntheticcx.adapters.channel.ownership import (
    CHAT_NAME,
    CHAT_PANEL_HELPER,
    OwnedScope,
    UnprovenChatRoot,
    attest_root,
    frame_origin_matches,
    origin,
)
from syntheticcx.adapters.channel.playwright import locate, normalize_echo
from syntheticcx.adapters.channel.reader import ExchangeReaderEvidence
from syntheticcx.adapters.channel.reading import READING_HELPERS
from syntheticcx.adapters.channel.website import WebsiteNavigator
from syntheticcx.config.models import (
    ChatOwnership,
    ChatState,
    ExecutionLimits,
    InterfaceChannelConfig,
    LocatorConfig,
    MessageIdentityConfig,
    ReplyOptionsConfig,
    ResponseCompletionConfig,
    SendStrategyConfig,
    StatefulChannelConfig,
)
from syntheticcx.discovery.hints import launcher_id
from syntheticcx.discovery.observations import (
    LearningObservation,
    ReaderBinding,
    WebsiteObservation,
    learning_observation,
    reader_binding,
    website_observation,
)
from syntheticcx.discovery.policy import portability_flow, require_discovery_probe
from syntheticcx.domain.models import CompletedResponse, Message, SelectionReceipt, SendReceipt
from syntheticcx.errors import ChannelError

SAFE_SEND_CONTROL = """node => node.isConnected &&
  node.matches('button[type=button],[role=button]') &&
  !node.closest('a,form') && !node.matches(':disabled') &&
  !!node.getClientRects().length &&
  /^(send|enviar)( message| mensagem)?$/i.test(
    (node.getAttribute('aria-label') || node.innerText || '').trim()) &&
  !['href','formaction','data-url','data-href','data-action']
    .some(a => node.hasAttribute(a))"""

PREOWNED_SCAN_SELECTOR = (
    "button,[role=button],iframe,dialog,[role=dialog],[role=alertdialog],[aria-controls],"
    "[data-chat-root],[data-chat-widget],[aria-label*='chat' i],"
    "[aria-label*='atendimento' i],[title*='chat' i],[title*='atendimento' i]"
)

ReadResult = TypeVar("ReadResult")


@dataclass
class Proposal:
    kind: str
    frame: Frame
    node: ElementHandle
    locator: LocatorConfig
    description: str
    evidence: str = "marked"


class NativeChatBroker:
    """One browser, one origin, one root and an append-only actual-node cursor."""

    def __init__(
        self,
        url: str | None = None,
        limits: ExecutionLimits | None = None,
        *,
        target_url: str | None = None,
        contract: StatefulChannelConfig | InterfaceChannelConfig | None = None,
        headed: bool = False,
        page: Page | None = None,
        chat_hint: str | None = None,
        website_recipient: str | None = None,
    ) -> None:
        self.contract = contract
        self.url = str(contract.url) if contract else str(target_url or url or "")
        if not self.url:
            raise ValueError("broker requires url or contract")
        self.limits = limits or ExecutionLimits(
            max_turns=3,
            journey_timeout_seconds=60,
            greeting_grace_seconds=0,
            browser_action_timeout_seconds=5,
            response_timeout_seconds=10,
            model_timeout_seconds=10,
            evaluator_timeout_seconds=10,
            persistence_timeout_seconds=5,
            close_timeout_seconds=5,
        )
        self.headed = headed
        self.chat_hint = chat_hint
        self._launcher_id = launcher_id(chat_hint)
        self.page = page
        self._external_page = page is not None
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self.owned: OwnedScope | None = None
        self._proposals: dict[str, Proposal] = {}
        self._catalog_signature: Any = None
        self._revision = secrets.token_urlsafe(18)
        self._known_nodes: list[ElementHandle] = []
        self._open_node: ElementHandle | None = None
        self._open_locator: LocatorConfig | None = None
        self._frame_locator: LocatorConfig | None = None
        self._root_locator: LocatorConfig | None = None
        self._assistant: LocatorConfig | None = None
        self._customer: LocatorConfig | None = None
        self._reader_evidence = contract.reader_evidence if contract else "marked"
        self._exchange: ExchangeReaderEvidence | None = None
        self._exchange_message: str | None = None
        self._response_deadline: float | None = None
        self._startup_deadline: float | None = None
        self._startup_ready = False
        self._identity = (
            contract.message_identity if contract else MessageIdentityConfig(strategy="dom_node")
        )
        self._completion = (
            contract.response_completion
            if contract
            else ResponseCompletionConfig(
                strategy="quiet_interval",
                quiet_interval_seconds=0.1,
            )
        )
        self._cursor = MessageCursor(self._identity)
        self._options: ReplyOptions | None = None
        self._state: ChatState | None = None
        self._binding: NativeControlBinding | None = None
        self._initial_state: ChatState | None = None
        self._states: dict[str, ChatState] = {}
        self._input: ElementHandle | None = None
        self._send: ElementHandle | None = None
        self._pending = False
        self._uncertain = False
        self._initial_observed = False
        self._started_at: datetime | None = None
        self._submission = 0
        self._action: tuple[str, str | None] | None = None
        self._lock = asyncio.Lock()
        self._closed = False
        self._started = False
        self._closing: asyncio.Task[None] | None = None
        self._response_bubble_attached = False
        self._marker_nodes: list[ElementHandle] = []
        self._indicator_seen = False
        self._indicator_task: asyncio.Task[Any] | None = None
        self._completed_ids: dict[int, str] = {}
        self._learn_transition_allowed = False
        self._resolved_transition = False
        self._initial_response: CompletedResponse | None = None
        self._bootstrap_before: list[ElementHandle] = []
        self._learning_source: ChatState | None = None
        self._learning_edges: list[tuple[str, str, str, str | None]] = []
        self._website: WebsiteNavigator | None = None
        self._website_recipient = website_recipient
        self.discovery_only = False

    @property
    def current_state(self) -> ChatState | None:
        return self._state

    @property
    def current_url(self) -> str | None:
        return self.page.url if self.page else None

    @property
    def response_bubble_attached(self) -> bool:
        return self._response_bubble_attached

    def _response_window(self) -> float:
        timeout = float(self.limits.response_timeout_seconds)
        window = timeout - min(0.1, timeout * 0.1)
        if self._response_deadline is not None:
            window = min(window, max(0, self._response_deadline - time.monotonic()))
        return window

    async def wait_until_ready(self) -> None:
        from syntheticcx.adapters.channel.readiness import interaction_ready, wait_for_interaction

        await self._check_parent()
        assert self.page is not None
        if self._startup_deadline is None:
            self._startup_deadline = (
                time.monotonic() + self.limits.browser_action_timeout_seconds * 0.9
            )
        if self.owned is None:
            await wait_for_interaction(self.page, self.url, self._open_node, self._startup_deadline)
        else:
            try:
                async with asyncio.timeout_at(self._startup_deadline):
                    while not await interaction_ready(self.owned, self._assistant, locate):
                        await asyncio.sleep(0.05)
            except TimeoutError:
                raise ChannelError(
                    "discover readiness", "chat interaction did not become ready before deadline"
                ) from None
        self._startup_ready = True

    def _response_timeout(self) -> ChannelError:
        return ChannelError(
            "channel.receive",
            (
                "assistant completion not observed before response deadline"
                if self._response_bubble_attached
                else "assistant bubble not observed before response deadline"
            ),
        )

    async def start(self, *, on_progress: Callable[[str], None] | None = None) -> None:
        if self._closed or self._started:
            raise ChannelError("channel.start", "broker is already started or closed")
        self._started = True
        if self.page is None:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=not self.headed)
            self._context = await self._browser.new_context()
            self._context.set_default_timeout(self.limits.browser_action_timeout_seconds * 1000)
            self.page = await self._context.new_page()
            response = await self.page.goto(self.url, wait_until="load")
            if response is None or response.status >= 400:
                raise ChannelError("channel.start", "native target navigation failed")
        await self._check_parent()
        if on_progress:
            on_progress("native target navigation succeeded")
        if self.contract is None and self._launcher_id:
            config = LocatorConfig(strategy="css", value=f'[id="{self._launcher_id}"]')
            opener = await self._unique(self.page, config)
            if on_progress:
                on_progress("activating explicitly hinted chat launcher")
            await self._activate_opener(opener, config, explicit_id=True)
            if on_progress:
                on_progress("explicitly hinted chat launcher activated")
        if self.contract is not None:
            config = self.contract
            if config.open_locator is not None:
                opener = await self._unique(self.page, config.open_locator)
                explicit = config.open_locator.strategy == "css" and bool(
                    re.fullmatch(
                        r'\[id="[A-Za-z][A-Za-z0-9_:-]{0,127}"\]', config.open_locator.value
                    )
                )
                await self._activate_opener(opener, config.open_locator, explicit_id=explicit)
            if config.navigation_steps:
                navigator = self.website_navigator
                await navigator.replay(config.navigation_steps)
                self.page = navigator.page
                await self._remember_website_opener()
            frame = self.page.main_frame
            if config.frame_locator is not None:
                frame_node = await self._unique(self.page, config.frame_locator)
                try:
                    frame = await frame_node.content_frame()
                finally:
                    await frame_node.dispose()
                if frame is None:
                    raise ChannelError("channel.start", "configured frame is unavailable")
                self._frame_locator = config.frame_locator
            root_locator = locate(frame, config.ownership.root_locator)
            try:
                await self._wait_for_locator(
                    root_locator,
                    visible=True,
                    timeout=self.limits.browser_action_timeout_seconds * 0.9,
                )
            except (PlaywrightError, TimeoutError):
                raise ChannelError(
                    "channel.start", "verified chat root is missing or ambiguous"
                ) from None
            self.owned = await attest_root(
                self.page,
                frame,
                root_locator,
                self.url,
                opener=self._open_node,
                expected_proof=config.ownership.proof,
            )
            self._root_locator = config.ownership.root_locator
            self._assistant, self._customer = (
                config.assistant_message_locator,
                config.customer_message_locator,
            )
            await self._validate_readers(require_present=False)
            await self._bootstrap_contract()

    async def _bootstrap_contract(self) -> None:
        """Replay startup policy, separate from shared browser/root/reader binding."""
        assert isinstance(self.contract, StatefulChannelConfig)
        state = next(
            item for item in self.contract.states if item.id == self.contract.initial_state_id
        )
        await self._wait_initial_state(state)

    async def _check_parent(self) -> None:
        if (
            not self._started
            or self.page is None
            or self._closed
            or origin(self.page.url) != origin(self.url)
        ):
            raise ChannelError("channel.ownership", "native target origin is unavailable")
        if self.owned is not None:
            await self.owned.check()

    def _matches_chat_hint(self, label: str) -> bool:
        return bool(
            self.chat_hint
            and self._launcher_id is None
            and normalized_label(self.chat_hint) in normalized_label(label)
        )

    async def _unique(self, scope: Any, config: LocatorConfig) -> ElementHandle:
        selector = locate(scope, config, include_hidden=True)
        if await selector.count() == 0:
            try:
                await self._wait_for_locator(
                    selector,
                    visible=False,
                    timeout=self.limits.browser_action_timeout_seconds * 0.9,
                )
            except (PlaywrightError, TimeoutError):
                raise ChannelError("channel.bind", "control is missing or ambiguous") from None
        if await selector.count() != 1:
            raise ChannelError("channel.bind", "control is missing or ambiguous")
        node = await selector.element_handle()
        if node is None:
            raise ChannelError("channel.bind", "control is absent")
        return node

    @staticmethod
    async def _wait_for_locator(selector: Locator, *, visible: bool, timeout: float) -> None:
        # Locator.wait_for discards the returned remote handle in our pinned
        # Playwright version. Boolean polling creates no retained reference.
        async with asyncio.timeout(timeout):
            while not await selector.count() or (visible and not await selector.is_visible()):
                await asyncio.sleep(0.025)

    async def _node_key(self, node: ElementHandle) -> int:
        index = await node.evaluate("(node, known) => known.indexOf(node)", self._known_nodes)
        if index < 0:
            if len(self._known_nodes) >= 2000:
                raise ChannelError("channel.observe", "native node history limit exceeded")
            index = len(self._known_nodes)
            retained = (await node.evaluate_handle("node => node")).as_element()
            assert retained is not None
            self._known_nodes.append(retained)
        return int(index)

    @staticmethod
    async def _dispose_handles(handles: list[ElementHandle | None]) -> None:
        """Dispose references once by handle identity, not by DOM-node identity."""
        unique = {id(handle): handle for handle in handles if handle is not None}
        results = await asyncio.gather(
            *(handle.dispose() for handle in unique.values()), return_exceptions=True
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result

    @staticmethod
    async def _finish_native_read(read: Coroutine[Any, Any, ReadResult]) -> ReadResult:
        """Drain read-only inspection before propagating caller cancellation.

        A cancelled protocol await can still create remote handles, but never
        return them to the local finally block. Keep the whole bounded inspection
        alive so it can receive and dispose its temporary references. This helper
        must never wrap dispatch, fill, click, or another side-effecting action.
        """
        task = asyncio.create_task(read)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not task.cancelled():
                task.exception()
            raise

    @property
    def website_navigator(self) -> WebsiteNavigator:
        if self._website is None:
            if self.page is None:
                raise ChannelError("channel.observe", "website page is unavailable")
            self._website = WebsiteNavigator(
                self.page, self.url, self.limits, recipient=self._website_recipient
            )
        self._website.discovery_only = self.discovery_only
        return self._website

    async def _remember_website_opener(self) -> None:
        node = self.website_navigator.last_node
        # Retain a separate handle: the website inventory owns its own references.
        if self._open_node is not None:
            await self._open_node.dispose()
        self._open_node = None
        if node is not None:
            try:
                if await node.evaluate("node => node.isConnected"):
                    self._open_node = (await node.evaluate_handle("node => node")).as_element()
            except PlaywrightError:
                pass
        self._startup_ready = False
        self._startup_deadline = None

    async def observe_website(self) -> WebsiteObservation:
        navigator = self.website_navigator
        # URL/recipient validation always precedes any account-wide DOM inspection.
        if await navigator.terminal():
            return {"terminal": "whatsapp"}
        self.page = navigator.page
        observed_channel: Literal["webchat", "whatsapp"] | None = None
        if origin(self.page.url) == origin(self.url):
            if (
                self._open_locator is not None
                and not navigator.steps
                and self._open_node is not None
            ):
                await self.wait_until_ready()
            observation = await self._observe_once()
            if self.owned is not None or any(
                item["kind"] == "scope" for item in observation["catalog"]["candidates"]
            ):
                observed_channel = "webchat"
                if self._website_recipient is None:
                    return {"terminal": "webchat"}
        initial_launcher_id = (
            launcher_id(self.chat_hint)
            if not navigator.steps and self._open_locator is None
            else None
        )
        website = await navigator.observe(
            initial_launcher_id=initial_launcher_id, chat_hint=self.chat_hint
        )
        website["observed_channel"] = observed_channel
        self.page = navigator.page
        return website_observation(website)

    async def act_website(self, revision: str, action: str, target_id: str | None = None) -> None:
        async with self._lock:
            if self._closed or self._pending or self._submission:
                raise ChannelError("channel.open", "website phase is no longer available")
            navigator = self.website_navigator
            await navigator.act(revision, action, target_id)
            self.page = navigator.page
            # Website actions invalidate discovery proposals, even on the same URL:
            # hover, modal dismissal and lazy content can replace their actual nodes.
            await self._release_native_handles()
            self._catalog_signature = None
            self._revision = secrets.token_urlsafe(18)
            self._startup_ready = False
            self._startup_deadline = None
            self._options = None
            self._state = None
            self._binding = None
            self._initial_state = None
            self._states.clear()
            self._initial_observed = False
            self._initial_response = None
            self._assistant = None
            self._customer = None
            self._frame_locator = None
            self._root_locator = None
            self._cursor = MessageCursor(self._identity)
            if self._exchange is not None:
                await self._exchange.close()
                self._exchange = None
            if origin(self.page.url) == origin(self.url):
                await self._remember_website_opener()

    async def observe(self) -> LearningObservation:
        """Allow lazy widget startup before offering an empty inventory to the planner."""
        deadline = time.monotonic() + self.limits.browser_action_timeout_seconds * 0.9
        while True:
            observation = await self._observe_once()
            candidates = observation["catalog"]["candidates"]
            if (
                self.owned is not None
                or self._open_node is not None
                or any(item["kind"] in {"opener", "scope"} for item in candidates)
                or time.monotonic() >= deadline
            ):
                return observation
            await asyncio.sleep(min(0.1, max(0, deadline - time.monotonic())))

    async def _observe_once(self) -> LearningObservation:
        from syntheticcx.adapters.channel.readiness import interaction_ready

        await self._check_parent()
        if self.contract is not None and self._state is not None and not self._pending:
            await self._validate_current_controls()
        assert self.page is not None
        if self.owned is not None and self._submission == 0:
            self._startup_ready = await interaction_ready(self.owned, self._assistant, locate)
        inventory: list[tuple[Any, Proposal]] = []
        linked_ids = (
            set((await self._open_node.get_attribute("aria-controls") or "").split())
            if self._open_node is not None
            else set()
        )
        frames = [self.owned.frame] if self.owned else self.page.frames
        for frame in frames[:20]:
            if not await frame_origin_matches(frame, self.url):
                continue
            chat_frame = False
            if self.owned is None and frame != self.page.main_frame:
                frame_node = await frame.frame_element()
                try:
                    name = await frame_node.evaluate(
                        "node => node.getAttribute('aria-label') || "
                        "node.getAttribute('title') || ''"
                    )
                    chat_frame = bool(CHAT_NAME.search(name))
                finally:
                    await frame_node.dispose()
            scope = self.owned.locator if self.owned else frame.locator("body")
            handles = await scope.locator(
                "*"
                if self.owned is not None
                else PREOWNED_SCAN_SELECTOR
                + (
                    ",div:has(textarea),section:has(textarea),div:has(input),section:has(input)"
                    if self._open_node is not None
                    else ""
                )
            ).element_handles()
            if self.owned is None and linked_ids:
                for linked_id in linked_ids:
                    linked = await scope.locator("[id=" + repr(linked_id) + "]").element_handles()
                    for candidate in linked:
                        duplicate = False
                        for existing in handles:
                            if await existing.evaluate(
                                "(node, other) => node === other", candidate
                            ):
                                duplicate = True
                                break
                        if duplicate:
                            await candidate.dispose()
                        else:
                            handles.append(candidate)
            if len(handles) > 2000:
                raise ChannelError("channel.observe", "catalog inspection limit exceeded")
            for node in handles:
                data = await node.evaluate(
                    "node => {"
                    + READING_HELPERS
                    + CHAT_PANEL_HELPER
                    + """
                  const refs = (node.getAttribute('aria-labelledby') || '')
                    .trim().split(/\\s+/).filter(Boolean);
                  let labels = [];
                  if (refs.length && refs.length <= 8) {
                    const scan = walkDOM(node.getRootNode(), false);
                    const matches = refs.map(id => scan.nodes.filter(el => el.id === id));
                    if (scan.complete && matches.every(entries => entries.length === 1))
                      labels = matches.flat();
                  }
                  const editor = node.matches('input,textarea,select,[role=textbox],' +
                    '[contenteditable=true]');
                  return {
                  visible: renderedElement(node), tag: node.tagName.toLowerCase(),
                  role: node.getAttribute('role') || '', id: node.id,
                  selector: node.id ? '#' + CSS.escape(node.id) :
                    node.classList.length ? node.tagName.toLowerCase() +
                      [...node.classList].map(c => '.' + CSS.escape(c)).join('') :
                      node.tagName.toLowerCase(),
                  label: (node.getAttribute('aria-label') || labels.map(labelText).join(' ') ||
                    node.getAttribute('placeholder') || node.getAttribute('title') ||
                    (editor ? '' : proseText(node))).slice(0, 1000),
                  editor,
                  safeComposer: editor && safeComposerElement(node),
                  linked: node.hasAttribute('aria-controls'),
                   embed: node.hasAttribute('data-chat-root') ||
                     node.hasAttribute('data-chat-widget') || /^chat[-_]/i.test(node.tagName),
                   structural: structuralChatPanel(node),
                  text: [...node.childNodes].some(n => n.nodeType === 3 && n.textContent.trim())
                  , actor: observedActor(node)
                  , readerSelector: observedActorSelector(node)
                }; }"""
                )
                if not data["visible"]:
                    continue
                kinds: list[str] = []
                button = data["tag"] == "button" or data["role"] == "button"
                if self.owned is None:
                    hinted_opener = self._matches_chat_hint(data["label"])
                    if (
                        self._open_node is None
                        and button
                        and (CHAT_NAME.search(data["label"]) or hinted_opener)
                    ):
                        kinds.append("opener")
                    if data["tag"] == "iframe":
                        kinds.append("frame")
                    if (
                        data["embed"]
                        or (self._open_node is not None and data["structural"])
                        or (
                            (data["tag"] == "dialog" or data["role"] in {"dialog", "alertdialog"})
                            and CHAT_NAME.search(data["label"])
                        )
                        or data["id"] in linked_ids
                        or (
                            chat_frame
                            and data["tag"]
                            not in {
                                "html",
                                "body",
                                "form",
                                "nav",
                                "main",
                                "button",
                                "input",
                                "textarea",
                                "select",
                                "a",
                                "iframe",
                                "script",
                                "style",
                            }
                        )
                    ):
                        kinds.append("scope")
                else:
                    if data["editor"]:
                        if data["safeComposer"]:
                            kinds.append("input")
                    elif button:
                        if (
                            await node.evaluate(SAFE_SEND_CONTROL)
                            and await self.owned.locator.locator(data["selector"]).count() == 1
                        ):
                            kinds.append("send")
                        if self._assistant is None or await locate(
                            self.owned.locator, self._assistant
                        ).evaluate_all(
                            "(wrappers, node) => {"
                            + READING_HELPERS
                            + "return wrappers.length > 0 && "
                            "composedContains(wrappers[wrappers.length - 1], node);}",
                            node,
                        ):
                            kinds.append("option")
                    elif data["tag"] not in {
                        "html",
                        "body",
                        "form",
                        "nav",
                        "main",
                        "dialog",
                        "a",
                    } and data["role"] not in {"dialog", "log", "feed", "status"}:
                        prose = await node.evaluate(
                            "node => {" + READING_HELPERS + "return proseText(node);}"
                        )
                        if prose or await node.query_selector_all("button,[role=button]"):
                            if data["actor"] in {"assistant", "customer"}:
                                kinds.append(data["actor"])
                            data["label"] = prose
                if not kinds:
                    continue
                key = await self._node_key(node)
                for kind in kinds:
                    description = (data["label"] or f"Observed {data['tag']} {kind}")[:256]
                    selector = (
                        data["readerSelector"]
                        if kind in {"assistant", "customer"}
                        else ("button" if data["tag"] == "button" else '[role="button"]')
                        if kind == "option"
                        else data["selector"]
                    )
                    locator = LocatorConfig(strategy="css", value=selector)
                    proposal = Proposal(
                        kind,
                        frame,
                        node,
                        locator,
                        description,
                    )
                    if kind == "scope" and not await self._attestable_scope(proposal):
                        continue
                    scope_rank = int(
                        not (
                            data["embed"]
                            or (
                                (
                                    data["tag"] == "dialog"
                                    or data["role"] in {"dialog", "alertdialog"}
                                )
                                and CHAT_NAME.search(data["label"])
                            )
                        )
                    )
                    inventory.append(
                        ((key, kind, description, data["selector"], scope_rank), proposal)
                    )
        if self._pending and self._assistant is None and self._exchange is not None:
            assert self.owned is not None
            existing = {(key[0], item.kind, item.locator.value) for key, item in inventory}
            for kind, locator, node, description in await self._exchange.candidates():
                key = await self._node_key(node)
                if (key, kind, locator.value) in existing:
                    continue
                inventory.append(
                    (
                        (key, kind, description, locator.value, 0),
                        Proposal(
                            kind, self.owned.frame, node, locator, description, "observed_exchange"
                        ),
                    )
                )
        priorities = (
            {"scope": 0, "opener": 1, "frame": 2}
            if self.owned is None
            else {
                "input": 0,
                "send": 1,
                "option": 2,
                "assistant": 3,
                "customer": 4,
            }
        )
        inventory.sort(
            key=lambda item: (
                priorities.get(item[1].kind, 5),
                item[0][4] if item[1].kind == "scope" else 0,
                -item[0][0],
            )
        )
        candidate_count = len(inventory)
        inventory = inventory[:160]
        signature = tuple(key for key, _ in inventory)
        if signature != self._catalog_signature:
            self._catalog_signature = signature
            self._revision = secrets.token_urlsafe(18)
            self._proposals = {secrets.token_urlsafe(18): item for _, item in inventory}
        if self._options is not None and not self._pending and not self._learn_transition_allowed:
            await self._options.collect()
        available = [
            {"id": identifier, "label": cap.option.label}
            for identifier, cap in (self._options.registry.items() if self._options else [])
        ]
        composer_id = None
        for identifier, item in self._proposals.items():
            if (
                item.kind == "input"
                and self._input is not None
                and await item.node.evaluate("(node, input) => node === input", self._input)
            ):
                composer_id = identifier
                break
        return learning_observation(
            {
                "revision": self._revision,
                "catalog": {
                    "revision": self._revision,
                    "truncated": candidate_count > 160,
                    "candidates": [
                        {
                            "id": identifier,
                            "kind": item.kind,
                            "description": item.description,
                            "evidence": item.evidence,
                        }
                        for identifier, item in self._proposals.items()
                    ],
                },
                "ownership_bound": self.owned is not None,
                "opener_opened": self._open_node is not None,
                "reader_bound": self._assistant is not None,
                "interaction_ready": self._startup_ready,
                "state": {
                    "id": self._state.id,
                    "kind": self._state.kind,
                    "option_labels": list(self._state.option_labels),
                }
                if self._state
                else None,
                "available_options": available,
                "composer_id": composer_id,
            }
        )

    async def _attestable_scope(self, proposal: Proposal) -> bool:
        """Offer only currently proven roots; binding still rechecks their identity."""
        assert self.page is not None
        candidate = None
        try:
            candidate = await attest_root(
                self.page,
                proposal.frame,
                locate(proposal.frame, proposal.locator),
                self.url,
                opener=self._open_node,
            )
            if not await candidate.root.evaluate(
                "(root, observed) => root === observed", proposal.node
            ):
                return False
            if proposal.frame != self.page.main_frame:
                frame_node = await proposal.frame.frame_element()
                try:
                    selector = await frame_node.evaluate(
                        "node => node.id ? '#' + CSS.escape(node.id) : null"
                    )
                    if not selector or await self.page.locator(selector).count() != 1:
                        return False
                finally:
                    await frame_node.dispose()
            return True
        except UnprovenChatRoot:
            # Proof/uniqueness rejection is read-only. Origin loss is never recoverable.
            await self._check_parent()
            if not await frame_origin_matches(proposal.frame, self.url):
                raise ChannelError("channel.ownership", "scope candidate origin changed") from None
            return False
        finally:
            if candidate is not None:
                await candidate.root.dispose()
                if candidate.embedded_frame is not None:
                    await candidate.embedded_frame.dispose()

    async def _proposal(self, identifier: str, kind: str, revision: str | None = None) -> Proposal:
        await self._check_parent()
        if revision is not None and revision != self._revision:
            raise ChannelError("channel.bind", "catalog revision is stale")
        item = self._proposals.get(identifier)
        if (
            item is None
            or item.kind != kind
            or not await item.node.evaluate("node => node.isConnected")
        ):
            raise ChannelError("channel.bind", "catalog proposal is missing or stale")
        if self.owned and not await self.owned.contains([item.node]):
            raise ChannelError("channel.bind", "proposal is outside owned scope")
        scope = self.owned.locator if self.owned else item.frame
        observed = await locate(scope, item.locator, include_hidden=True).element_handles()
        if not await item.node.evaluate("(node, nodes) => nodes.includes(node)", observed):
            raise ChannelError("channel.bind", "proposal locator no longer resolves observed node")
        return item

    async def _activate_opener(
        self,
        node: ElementHandle,
        config: LocatorConfig,
        *,
        explicit_id: bool = False,
    ) -> None:
        await self._check_parent()
        data = await node.evaluate(
            "node => {"
            + READING_HELPERS
            + """
          const refs = (node.getAttribute('aria-labelledby') || '')
            .trim().split(/\\s+/).filter(Boolean);
          const scan = refs.length ? walkDOM(node.getRootNode(), false) :
            {nodes: [], complete: true};
          const labels = refs.map(id => scan.nodes.filter(el => el.id === id));
          const text = [node.id, node.textContent || '', node.getAttribute('aria-label') || '',
            node.getAttribute('title') || '',
            ...labels.flat().map(labelText)].join(' ');
          const unsafe = new RegExp('\\\\b(pay|payment|purchase|buy|order|submit|delete|' +
            'confirm|login|consent|agree|accept|upload|log in|sign in|comprar|pagar|' +
            'enviar|excluir|confirmar|entrar|aceitar|assinar)\\\\b', 'i');
          const ancestry = [];
          for (let p = node; p; p = p.assignedSlot || composedParent(p)) ancestry.push(p);
          const roleAnchor = node.tagName === 'A' && node.getAttribute('role') === 'button' &&
            !node.hasAttribute('href');
          const blocked = ancestry.some(p => p.matches('form,[role=link],[inert],[disabled]') ||
            (p.tagName === 'A' && (p !== node || !roleAnchor)) ||
            (p.getAttribute('aria-disabled') || '').toLowerCase() === 'true' ||
            ['href','target','download','formaction','form','src','action',
              'data-url','data-href','data-action'].some(a => p.hasAttribute(a)));
          return {name: text, safe: node.isConnected &&
            (node.tagName === 'BUTTON' && node.type === 'button' ||
              roleAnchor) && !blocked &&
            !node.matches(':disabled') && renderedElement(node) &&
            node.getAttribute('aria-expanded') !== 'true' && !unsafe.test(text) &&
            scan.complete && refs.length <= 8 && labels.every(entries => entries.length === 1)};
        }"""
        )
        if not data["safe"] or (
            not explicit_id
            and not self._matches_chat_hint(data["name"])
            and not CHAT_NAME.search(data["name"])
        ):
            raise ChannelError(
                "channel.open", "opener is not a positively identified safe chat control"
            )
        if self._uncertain or self._open_node is not None:
            raise ChannelError("channel.open", "opener was already dispatched")
        self._uncertain = True
        await node.click(timeout=self.limits.browser_action_timeout_seconds * 1000)
        await self._check_parent()
        self._open_node, self._open_locator = node, config
        self._startup_deadline = time.monotonic() + self.limits.browser_action_timeout_seconds * 0.9
        self._uncertain = False

    async def open(
        self, target_id: str, catalog_target_id: str | None = None, *, revision: str | None = None
    ) -> None:
        if catalog_target_id is not None:
            revision, target_id = target_id, catalog_target_id
        item = await self._proposal(target_id, "opener", revision)
        current = await self._unique(item.frame, item.locator)
        if not await current.evaluate("(node, expected) => node === expected", item.node):
            raise ChannelError("channel.open", "opener changed")
        await self._activate_opener(item.node, item.locator)

    async def bind_scope(
        self, target_id: str, catalog_target_id: str | None = None, *, revision: str | None = None
    ) -> ChatOwnership:
        if catalog_target_id is not None:
            revision, target_id = target_id, catalog_target_id
        if self.owned is not None:
            raise ChannelError("channel.bind_scope", "chat root is already pinned")
        item = await self._proposal(target_id, "scope", revision)
        assert self.page is not None
        self.owned = await attest_root(
            self.page,
            item.frame,
            locate(item.frame, item.locator),
            self.url,
            opener=self._open_node,
        )
        self._root_locator = item.locator
        if item.frame != self.page.main_frame:
            frame_node = await item.frame.frame_element()
            selector = await frame_node.evaluate(
                "node => node.id ? '#' + CSS.escape(node.id) : null"
            )
            if not selector or await self.page.locator(selector).count() != 1:
                self.owned = None
                raise ChannelError(
                    "channel.bind_scope", "selected frame has no unique replay locator"
                )
            self._frame_locator = LocatorConfig(strategy="css", value=selector)
        return ChatOwnership(root_locator=item.locator, proof=self.owned.proof)

    async def bind_reader(
        self,
        assistant_id: str,
        customer_id: str | None = None,
        *catalog_args: Any,
        revision: str | None = None,
        response_completion: ResponseCompletionConfig | dict[str, Any] | None = None,
    ) -> ReaderBinding:
        if catalog_args:
            if customer_id is None:
                raise ChannelError("channel.bind_reader", "assistant catalog ID is required")
            revision, assistant_id, customer_id = assistant_id, customer_id, catalog_args[0]
        if self.owned is None:
            raise ChannelError("channel.bind_reader", "independent scope proof is required")
        assistant = await self._proposal(assistant_id, "assistant", revision)
        customer = await self._proposal(customer_id, "customer", revision) if customer_id else None
        if self._assistant is not None and assistant.locator != self._assistant:
            raise ChannelError("channel.bind_reader", "proved assistant reader is immutable")
        if (
            customer is not None
            and self._customer is not None
            and customer.locator != self._customer
        ):
            raise ChannelError("channel.bind_reader", "proved customer reader is immutable")
        old_assistant, old_customer, old_completion, old_evidence = (
            self._assistant,
            self._customer,
            self._completion,
            self._reader_evidence,
        )
        self._assistant = assistant.locator
        self._customer = customer.locator if customer else self._customer
        if response_completion is not None:
            self._completion = ResponseCompletionConfig.model_validate(response_completion)
        try:
            if assistant.evidence == "observed_exchange":
                if self._exchange is None or self._customer is None or not self._pending:
                    raise ChannelError(
                        "channel.reader_evidence", "witnessed exchange readers are incomplete"
                    )
                try:
                    await self._exchange.validate(self._assistant, self._customer)
                except ChannelError as error:
                    if error.operation == "channel.ownership":
                        raise
                    raise ChannelError(
                        "channel.reader_evidence", "reader partition could not be corroborated"
                    ) from None
                self._reader_evidence = "observed_exchange"
            await self._validate_readers(require_present=True)
        except Exception:
            self._assistant, self._customer, self._completion, self._reader_evidence = (
                old_assistant,
                old_customer,
                old_completion,
                old_evidence,
            )
            raise
        if self._pending and self._bootstrap_before:
            before_keys = {str(await self._node_key(node)) for node in self._bootstrap_before}
            baseline = [
                part
                for part in await self._snapshot("assistant")
                if part.attribute_id in before_keys
            ]
            self._cursor.observe("assistant", baseline, baseline=True)
            if self._customer is not None:
                self._cursor.observe("customer", await self._snapshot("customer"), baseline=True)
        return reader_binding(
            {
                "assistant_message_locator": self._assistant.model_dump(mode="json"),
                "customer_message_locator": self._customer.model_dump(mode="json")
                if self._customer
                else None,
            }
        )

    async def _validate_readers(self, *, require_present: bool) -> None:
        if self.owned is None or self._assistant is None:
            raise ChannelError("channel.reader", "owned assistant reader is not bound")
        await self.owned.check()
        assistants = await locate(
            self.owned.locator, self._assistant, include_hidden=True
        ).element_handles()
        customers = (
            await locate(self.owned.locator, self._customer, include_hidden=True).element_handles()
            if self._customer
            else []
        )
        try:
            if (
                len(assistants) > 120
                or len(customers) > 120
                or (require_present and not assistants)
            ):
                raise ChannelError(
                    "channel.reader", "assistant ownership evidence is missing or oversized"
                )
            if not await self._reader_partition_valid(assistants, customers):
                raise ChannelError(
                    "channel.reader", "message wrappers lack exclusive assistant/customer ownership"
                )
        finally:
            await self._dispose_handles([*assistants, *customers])

    async def _reader_partition_valid(
        self, assistants: list[ElementHandle], customers: list[ElementHandle]
    ) -> bool:
        assert self.owned is not None
        return bool(
            await self.owned.root.evaluate(
                "(root, args) => {"
                + READING_HELPERS
                + """
           const forbidden = 'html,body,form,button,a,input,textarea,select,dialog,nav,main,' +
             'script,style,template,noscript,' +
            '[role=dialog],[role=log],[role=feed],[role=status],[role=button],[role=textbox]';
          const all = [...args.assistants, ...args.customers];
          return all.every(node => node !== root && node.isConnected &&
            composedContains(root,node) &&
            !node.matches(forbidden) && walkDOM(node).complete &&
            !walkDOM(node).nodes.some(child =>
              child.matches('input,textarea,select,[role=textbox]')) &&
            !all.some(other => other !== node &&
              (composedContains(node,other) || composedContains(other,node)))) &&
             args.assistants.every(node => !args.customers.includes(node) &&
               (args.exchange ? observedActor(node) !== 'customer' :
                 observedActor(node) === 'assistant')) &&
             args.customers.every(node => args.exchange ? observedActor(node) !== 'assistant' :
               observedActor(node) === 'customer');
        }""",
                {
                    "assistants": assistants,
                    "customers": customers,
                    "exchange": self._reader_evidence == "observed_exchange",
                },
            )
        )

    async def bind_state(
        self,
        state_id: str | ChatState | dict[str, Any] | None = None,
        *catalog_args: Any,
        input_id: str | None = None,
        send_id: str | None = None,
        option_id: str | None = None,
        revision: str | None = None,
        kind: str | None = None,
        send_mode: str | None = None,
    ) -> ChatState:
        if catalog_args:
            if len(catalog_args) != 4:
                raise ChannelError("channel.bind_state", "invalid catalog state envelope")
            revision = str(state_id)
            state_id = None
            input_id, send_id, option_id, send_mode = catalog_args
            if (input_id is None) != (send_mode is None) or (send_mode == "click") != (
                send_id is not None
            ):
                raise ChannelError("channel.bind_state", "invalid catalog send controls")
        if self.owned is None:
            raise ChannelError("channel.bind_state", "independent scope proof is required")
        state: ChatState
        if isinstance(state_id, (ChatState, dict)):
            state = ChatState.model_validate(state_id)
        else:
            input_item = await self._proposal(input_id, "input", revision) if input_id else None
            send_item = await self._proposal(send_id, "send", revision) if send_id else None
            option_item = await self._proposal(option_id, "option", revision) if option_id else None
            labels: list[str] = []
            if option_item:
                if self._assistant is None:
                    raise ChannelError(
                        "channel.bind_state", "menu requires a proved assistant reader"
                    )
                # Only the eligible guarded subset is recorded; utility or unsafe
                # controls inside the reply are ignored and never become options.
                labels = await self._eligible_menu_labels(option_item.locator)
                if not labels:
                    raise ChannelError("channel.bind_state", "menu has no eligible guarded options")
            state = ChatState(
                id=state_id or f"state-{len(self._states) + 1}",
                kind=kind
                or (
                    "mixed"
                    if input_item and option_item
                    else "composer"
                    if input_item
                    else "menu"
                    if option_item
                    else "terminal"
                ),
                input_locator=input_item.locator if input_item else None,
                send_strategy=SendStrategyConfig(type="click", locator=send_item.locator)
                if send_item
                else SendStrategyConfig(type="press_enter")
                if input_item
                else None,
                option_locator=option_item.locator if option_item else None,
                option_labels=labels,
            )
            if state_id is None:
                controls = state.model_dump(exclude={"id", "option_labels"})
                normalized = [normalized_label(label) for label in state.option_labels]
                for previous in self._states.values():
                    if (
                        previous.model_dump(exclude={"id", "option_labels"}) == controls
                        and [normalized_label(label) for label in previous.option_labels]
                        == normalized
                    ):
                        state = previous
                        break
        await self._bind_typed_state(state)
        if state.option_locator is not None and not self._initial_observed:
            self._initial_response = await self.observe_initial()
        return state

    def _option_config(self, binding: NativeControlBinding) -> NativeOptionConfig:
        assert self.owned is not None and self._assistant is not None
        assert binding.option_locator is not None
        return NativeOptionConfig(
            url=self.url,
            input_locator=binding.input_locator,
            assistant_message_locator=self._assistant,
            customer_message_locator=self._customer
            or LocatorConfig(strategy="css", value=".__syntheticcx_absent_customer__"),
            send_strategy=binding.send_strategy or SendStrategyConfig(type="press_enter"),
            reply_options=ReplyOptionsConfig(
                scope_locator=LocatorConfig(strategy="css", value=":scope"),
                option_locator=binding.option_locator,
            ),
        )

    def _make_reply_options(self, binding: NativeControlBinding) -> ReplyOptions:
        assert self.owned is not None
        return ReplyOptions(
            self.owned.locator, self._option_config(binding), locate, attested_scope=self.owned
        )

    async def _eligible_menu_labels(self, option_locator: LocatorConfig) -> list[str]:
        """Ordered eligible guarded-option labels of the latest assistant reply."""
        assert self.owned is not None and self._assistant is not None
        evidence = NativeControlBinding(option_locator=option_locator)
        probe = ReplyOptions(
            self.owned.locator, self._option_config(evidence), locate, attested_scope=self.owned
        )
        try:
            await probe.bind()
            replies = await probe.collect()
        finally:
            await probe.dispose()
        if not replies:
            return []
        return [node.label for node in replies[-1].options if node.reason is None]

    async def _inspect_state(
        self, state: ChatState
    ) -> tuple[ElementHandle | None, ElementHandle | None, ReplyOptions | None]:
        assert self.owned is not None
        await self.owned.check()
        visible_inputs = [
            node
            for node in await self.owned.locator.locator(
                "textarea,input:not([type]),input[type=text],[role=textbox],[contenteditable=true]"
            ).element_handles()
            if await node.is_visible()
        ]
        if len(visible_inputs) != (1 if state.input_locator else 0):
            raise ChannelError(
                "channel.bind_state", "observed composer inventory does not match state"
            )
        input_node = (
            await self._unique(self.owned.locator, state.input_locator)
            if state.input_locator
            else None
        )
        send_node = (
            await self._unique(self.owned.locator, state.send_strategy.locator)
            if state.send_strategy and state.send_strategy.locator
            else None
        )
        controls = [node for node in [input_node, send_node] if node is not None]
        if not await self.owned.contains(controls):
            raise ChannelError("channel.bind_state", "controls are outside pinned root")
        if input_node:
            valid = await input_node.evaluate(
                "node => {" + READING_HELPERS + "return safeComposerElement(node);}"
            )
            if not valid:
                raise ChannelError(
                    "channel.bind_state", "composer is not an available text control"
                )
        if send_node:
            valid = await send_node.evaluate(SAFE_SEND_CONTROL)
            if not valid:
                raise ChannelError(
                    "channel.bind_state", "send control is not a safe chat submission"
                )
        # Existing composer nodes can only be exchanged by an observed transition.
        if (
            self._input is not None
            and input_node is not None
            and not self._pending
            and not self._learn_transition_allowed
        ):
            same = await self._input.evaluate("(node, current) => node === current", input_node)
            if not same:
                raise ChannelError(
                    "channel.bind_state", "composer replacement requires a verified transition"
                )
        options = None
        if state.option_locator:
            await self._validate_readers(require_present=True)
            options = ReplyOptions(
                self.owned.locator,
                self._option_config(NativeControlBinding.from_state(state)),
                locate,
                attested_scope=self.owned,
            )
            await options.bind()
            replies = await options.collect()
            eligible = (
                [node.label for node in replies[-1].options if node.reason is None]
                if replies
                else []
            )
            if not eligible:
                raise ChannelError("channel.bind_state", "menu has unsafe or ambiguous options")
            if [normalized_label(label) for label in eligible] != [
                normalized_label(label) for label in state.option_labels
            ]:
                raise ChannelError(
                    "channel.bind_state", "observed menu does not match state labels"
                )
        elif self._assistant is not None:
            # Ineligible utility controls are ignored; an unbound *eligible* menu
            # still fails closed because the persona could never receive it.
            if await self._eligible_menu_labels(
                LocatorConfig(strategy="css", value="button,[role=button]")
            ):
                raise ChannelError("channel.bind_state", "unconfigured current menu controls")
        return input_node, send_node, options

    async def _bind_typed_state(self, state: ChatState) -> None:
        learning_transition = self._learn_transition_allowed
        input_node, send_node, options = await self._inspect_state(state)
        if self._options:
            self._options.invalidate()
        self._options, self._input, self._send = options, input_node, send_node
        self._state = state
        self._binding = NativeControlBinding.from_state(state)
        self._states[state.id] = state
        if self._initial_state is None:
            self._initial_state = state
        self._learn_transition_allowed = False
        if learning_transition:
            if self._learning_source is None or self._action is None:
                raise ChannelError("channel.bind_state", "completed transition evidence is missing")
            self._learning_edges.append((self._learning_source.id, state.id, *self._action))
            self._learning_source = None
        # Learning binds target controls after receive; republish only the completed
        # latest owner, not an uncompleted replacement or a catalog proposal.
        if options is not None and self._completed_ids and not self._pending:
            options.publish(await options.collect(), self._completed_ids)

    async def _wait_initial_state(self, state: ChatState) -> None:
        deadline = time.monotonic() + self.limits.browser_action_timeout_seconds * 0.9
        while time.monotonic() < deadline:
            await self._check_parent()
            try:
                await self._inspect_state(state)
            except ChannelError as error:
                if error.operation == "channel.ownership":
                    raise
                await asyncio.sleep(0.025)
                continue
            await self._bind_typed_state(state)
            return
        raise ChannelError("channel.start", "verified initial state is unavailable or ambiguous")

    async def _snapshot(self, actor: Literal["assistant", "customer"]) -> list[Bubble]:
        return await self._finish_native_read(self._snapshot_owned(actor))

    async def _snapshot_owned(self, actor: Literal["assistant", "customer"]) -> list[Bubble]:
        assert self.owned is not None
        await self.owned.check()
        config = self._assistant if actor == "assistant" else self._customer
        if config is None:
            return []
        await self._validate_readers(require_present=False)
        nodes = await locate(self.owned.locator, config, include_hidden=True).element_handles()
        try:
            bubbles = []
            for ordinal, node in enumerate(nodes):
                key = await self._node_key(node)
                content = await node.evaluate(
                    "node => {" + READING_HELPERS + "return proseText(node);}"
                )
                identity: str | None = str(key)
                if self._identity.strategy == "attribute":
                    identity = await node.get_attribute(self._identity.attribute_name)
                bubbles.append(Bubble(actor, ordinal, content, identity))
            return bubbles
        finally:
            await self._dispose_handles(list(nodes))

    async def _prepare_completion(self) -> None:
        if self._indicator_task is not None:
            self._indicator_task.cancel()
            await asyncio.gather(self._indicator_task, return_exceptions=True)
        self._indicator_task = None
        self._indicator_seen = False
        await self._dispose_handles(list(self._marker_nodes))
        self._marker_nodes = []
        if self._completion.locator is not None and self.owned is not None:
            marker = locate(self.owned.locator, self._completion.locator)
            self._marker_nodes = await marker.element_handles()
            if self._completion.strategy == "streaming_indicator_disappears":
                if await marker.is_visible():
                    raise ChannelError("channel.receive", "streaming indicator was already active")
                self._indicator_task = asyncio.create_task(
                    self._wait_for_locator(
                        marker, visible=True, timeout=self.limits.response_timeout_seconds
                    )
                )

    async def _completion_ready(self, stable_since: float) -> bool:
        quiet = self._completion.quiet_interval_seconds or 0.1
        if time.monotonic() - stable_since < quiet:
            return False
        if self._completion.strategy == "quiet_interval":
            return True
        assert self.owned is not None and self._completion.locator is not None
        marker = locate(self.owned.locator, self._completion.locator)
        if self._completion.strategy == "streaming_indicator_disappears":
            task = self._indicator_task
            return bool(
                task is not None
                and task.done()
                and not task.cancelled()
                and task.exception() is None
                and not await marker.is_visible()
            )
        nodes = await marker.element_handles()
        try:
            for node in nodes:
                if await node.is_visible() and not await node.evaluate(
                    "(node, old) => old.includes(node)", self._marker_nodes
                ):
                    return True
            return False
        finally:
            await self._dispose_handles(list(nodes))

    async def observe_initial(
        self, *, on_progress: Callable[[str], None] | None = None
    ) -> CompletedResponse | None:
        if self._initial_response is not None:
            response, self._initial_response = self._initial_response, None
            return response
        if self._initial_observed:
            return None
        if self._assistant is None:
            return None
        # Initial menu completion is mandatory even when greeting grace is zero.
        snapshot = await self._snapshot("assistant")
        if not snapshot:
            if self.limits.greeting_grace_seconds <= 0:
                self._initial_observed = True
                return None
            deadline = time.monotonic() + self.limits.greeting_grace_seconds
            while not snapshot and time.monotonic() < deadline:
                await asyncio.sleep(0.025)
                snapshot = await self._snapshot("assistant")
            if not snapshot:
                self._initial_observed = True
                return None
        result = await self._receive(initial=True, on_progress=on_progress)
        self._initial_observed = True
        return result

    async def submit_probe(
        self,
        message: str,
        target_id: str | None = None,
        text: str | None = None,
        *,
        revision: str | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> SendReceipt:
        if text is not None:
            revision, message = message, text
        if self.discovery_only:
            require_discovery_probe(message)
        if target_id is not None:
            item = await self._proposal(target_id, "input", revision)
            if self._input is None or not await item.node.evaluate(
                "(node, input) => node === input", self._input
            ):
                raise ChannelError("channel.send", "probe target is not the bound composer")
        return await self.send(message, on_progress=on_progress)

    async def _baseline(self) -> None:
        self._initial_response = None
        if self._assistant is not None:
            self._cursor.observe("assistant", await self._snapshot("assistant"), baseline=True)
        else:
            assert self.owned is not None
            self._bootstrap_before = await self.owned.locator.locator("*").element_handles()
            if len(self._bootstrap_before) > 2000:
                raise ChannelError("channel.receive", "bootstrap node inventory exceeds limit")
        if self._customer is not None:
            observed = self._cursor.observe(
                "customer", await self._snapshot("customer"), baseline=self._submission == 0
            )
            if observed:
                raise ChannelError("channel.send", "unexpected customer message before submission")
        await self._prepare_completion()

    async def _check_dispatch(self) -> None:
        await self._check_parent()
        if (
            self.owned is None
            or self._binding is None
            or self._uncertain
            or self._pending
            or self._learn_transition_allowed
        ):
            raise ChannelError("channel.send", "dispatch is unbound, pending, or uncertain")
        if self.discovery_only and self._assistant is not None:
            current = await self._snapshot("assistant")
            if current and portability_flow(current[-1].content):
                raise ChannelError(
                    "channel.send", "number-portability flows are forbidden in discovery"
                )
        await self._validate_current_controls()

    def _guard_transition(self, kind: str, label: str | None) -> None:
        if self.contract is None:
            if self._submission >= 3:
                raise ChannelError("channel.send", "learning action budget is exhausted")
            return
        assert self._state is not None
        if kind == "probe":
            allowed = self._state.input_locator is not None
        else:
            allowed = label is not None and normalized_label(label) in {
                normalized_label(item) for item in self._state.option_labels
            }
        if not allowed:
            raise ChannelError(
                "channel.state", "action is unavailable in the verified source state; no dispatch"
            )

    async def _observe_echo(self, message: str | None) -> tuple[str, dict[str, str]]:
        if self._customer is None:
            return f"customer:submitted-{self._submission + 1}", {"echo": "absent"}
        await asyncio.sleep(self.limits.echo_observation_seconds)
        await self._check_parent()
        echoes = self._cursor.observe("customer", await self._snapshot("customer"))
        if len(echoes) > 1 or (
            echoes
            and message is not None
            and normalize_echo(echoes[0].content) != normalize_echo(message)
        ):
            raise ChannelError(
                "channel.send", "customer echo is ambiguous or differs from submission"
            )
        if not echoes:
            return f"customer:submitted-{self._submission + 1}", {"echo": "absent"}
        identifier = self._cursor.finalize("customer", echoes)[0]
        return identifier, {
            "echo": "matched" if message is not None else "observed",
            "dom_message_id": identifier,
        }

    async def send(
        self, message: str, *, on_progress: Callable[[str], None] | None = None
    ) -> SendReceipt:
        async with self._lock:
            await self._check_dispatch()
            if (
                not message.strip()
                or self._input is None
                or self._binding is None
                or self._binding.send_strategy is None
            ):
                raise ChannelError(
                    "channel.send", "current verified state has no composer or message is empty"
                )
            if self._assistant is not None and not self._initial_observed:
                await self.observe_initial()
            self._guard_transition("probe", None)
            await self._validate_current_controls()
            await self._baseline()
            if self._assistant is None or self._reader_evidence == "observed_exchange":
                assert self.owned is not None
                if self._exchange is None:
                    self._exchange = ExchangeReaderEvidence(self.owned)
                await self._exchange.begin(message)
                self._exchange_message = message
            if self._options:
                self._options.invalidate()
            # Filling can itself dispatch application input handlers; latch first.
            self._uncertain = True
            try:
                await self._input.fill(
                    message, timeout=self.limits.browser_action_timeout_seconds * 1000
                )
                await self._validate_current_controls()
                if self._binding.send_strategy.type == "press_enter":
                    await self._input.press(
                        "Enter", timeout=self.limits.browser_action_timeout_seconds * 1000
                    )
                else:
                    assert self._send is not None
                    await self._send.click(
                        timeout=self.limits.browser_action_timeout_seconds * 1000
                    )
                await self._check_parent()
                receipt_id, evidence = await self._observe_echo(message)
            except (PlaywrightError, TimeoutError):
                raise ChannelError(
                    "channel.send", "native submission is uncertain; retry prohibited"
                ) from None
            now = datetime.now(UTC)
            self._submission += 1
            self._response_deadline = time.monotonic() + self.limits.response_timeout_seconds
            self._pending, self._uncertain = True, False
            self._started_at, self._action = now, ("probe", None)
            self._learning_source = self._state
            self._resolved_transition = False
            return SendReceipt(
                message_id=receipt_id,
                submitted_at=now,
                evidence=evidence,
            )

    async def _validate_current_controls(self) -> None:
        assert self.owned is not None and self._binding is not None
        await self.owned.check()
        inputs = await self.owned.locator.locator(
            "textarea,input:not([type]),input[type=text],[role=textbox],[contenteditable=true]"
        ).element_handles()
        try:
            visible_inputs = [node for node in inputs if await node.is_visible()]
            if len(visible_inputs) != (1 if self._binding.input_locator else 0):
                raise ChannelError(
                    "channel.state", "current composer inventory no longer matches verified state"
                )
        finally:
            await self._dispose_handles(list(inputs))
        for config, retained in [
            (self._binding.input_locator, self._input),
            (
                self._binding.send_strategy.locator if self._binding.send_strategy else None,
                self._send,
            ),
        ]:
            if config is None:
                continue
            current = await self._unique(self.owned.locator, config)
            try:
                if retained is None or not await retained.evaluate(
                    "(node, current) => node === current && node.isConnected", current
                ):
                    raise ChannelError("channel.send", "pinned current control changed")
            finally:
                await current.dispose()
            safe = await self._safe_current_control(retained, is_input=retained is self._input)
            if not safe:
                raise ChannelError("channel.send", "current control is no longer safe or available")

    async def _safe_current_control(
        self, node: ElementHandle, *, is_input: bool, allow_disabled_send: bool = False
    ) -> bool:
        return bool(
            await node.evaluate(
                "(node, args) => {"
                + READING_HELPERS
                + """
              const input = args.input;
              const structural = !input && args.allowDisabledSend;
              const ancestry = [];
              for (let p = node; p; p = composedParent(p)) ancestry.push(p);
              const unsafe = ancestry.some(p => p.matches('a,form,[inert]') ||
                (!(structural && p === node) && (p.matches('[disabled]') ||
                  (p.getAttribute('aria-disabled') || '').toLowerCase() === 'true')) ||
                ['href','formaction','data-url','data-href','data-action']
                  .some(a => p.hasAttribute(a)));
              if (!node.isConnected || unsafe || !renderedElement(node) ||
                (!structural && node.matches(':disabled'))) return false;
               if (input) return safeComposerElement(node);
              return node.matches('button,[role=button]') &&
                (node.tagName !== 'BUTTON' || node.type === 'button') &&
                /^(send|enviar)( message| mensagem)?$/i.test(
                  (node.getAttribute('aria-label') || node.innerText || '').trim());
            }""",
                {"input": is_input, "allowDisabledSend": allow_disabled_send},
            )
        )

    async def select_option(self, identifier: str) -> SelectionReceipt:
        async with self._lock:
            await self._check_dispatch()
            if self._options is None or not self._initial_observed:
                raise ChannelError(
                    "channel.select_option", "completed assistant-owned menu is required"
                )
            cap = await self._options.validate(identifier)
            self._guard_transition("select_option", cap.option.label)
            await self._baseline()
            await self._options.validate(identifier)
            if self._reader_evidence == "observed_exchange":
                assert self.owned is not None
                if self._exchange is None:
                    self._exchange = ExchangeReaderEvidence(self.owned)
                await self._exchange.begin(cap.option.label)
                self._exchange_message = cap.option.label
            self._options.invalidate()
            self._uncertain = True
            try:
                await cap.option.node.click(
                    timeout=self.limits.browser_action_timeout_seconds * 1000
                )
                await self._check_parent()
                receipt_id, evidence = await self._observe_echo(None)
            except (PlaywrightError, TimeoutError):
                raise ChannelError(
                    "channel.select_option", "native selection is uncertain; retry prohibited"
                ) from None
            now = datetime.now(UTC)
            self._submission += 1
            self._pending, self._uncertain = True, False
            self._started_at, self._action = now, ("select_option", cap.option.label)
            self._response_deadline = time.monotonic() + self.limits.response_timeout_seconds
            self._learning_source = self._state
            self._resolved_transition = False
            return SelectionReceipt(
                message_id=receipt_id,
                submitted_at=now,
                evidence=evidence,
                selected_option_id=identifier,
                selected_label=cap.option.label,
                source_message_id=cap.message_id,
            )

    async def receive(
        self, *, on_progress: Callable[[str], None] | None = None
    ) -> CompletedResponse:
        if not self._pending:
            raise ChannelError("channel.receive", "no response is pending")
        try:
            async with asyncio.timeout(self._response_window()):
                if self._assistant is None:
                    await self._bootstrap_reader()
                return await self._receive(initial=False, on_progress=on_progress)
        except TimeoutError:
            raise self._response_timeout() from None

    async def wait_for_reader_candidates(self) -> None:
        while self._response_deadline is not None and time.monotonic() < self._response_deadline:
            observation = await self.observe()
            if any(item["kind"] == "assistant" for item in observation["catalog"]["candidates"]):
                return
            await asyncio.sleep(0.1)
        raise ChannelError(
            "discover reader", "no corroborated reader candidates before response deadline"
        )

    async def _bootstrap_reader(self) -> None:
        """Attribute a first reply only to a unique *new*, positively marked actor.

        No generic panel/prose inference, guessed selector, global search or second
        probe. Unmarked or ambiguous replies require controller reader learning;
        failure retains the pending submission and forbids resubmission.
        """
        if self.owned is None or self.contract is not None or self._submission != 1:
            raise ChannelError(
                "channel.receive", "assistant reader must be learned before receiving"
            )
        deadline = time.monotonic() + self._response_window()
        while time.monotonic() < deadline:
            await self.owned.check()
            nodes = await self.owned.locator.locator("*").element_handles()
            if len(nodes) > 2000:
                raise ChannelError("channel.receive", "bootstrap reply inventory exceeds limit")
            candidates: list[tuple[ElementHandle, str]] = []
            for node in nodes:
                evidence = await node.evaluate(
                    "(node, before) => {"
                    + READING_HELPERS
                    + """
                   const marked = observedActor(node) === 'assistant';
                  if (!marked || before.includes(node) || !renderedElement(node) ||
                     node.matches('html,body,form,button,a,input,textarea,select,dialog,nav,main,' +
                       'script,style,template,noscript,' +
                      '[role=dialog],[role=log],[role=feed],[role=status],[role=button]') ||
                    !walkDOM(node).complete || walkDOM(node).nodes.some(child =>
                      child.matches('input,textarea,select,[role=textbox]'))) return null;
                  // This observed locator is a replay proposal only; actual new-node
                  // membership, actor metadata, exclusivity and root pin prove it.
                   return observedActorSelector(node);
                }""",
                    self._bootstrap_before,
                )
                if evidence:
                    candidates.append((node, evidence))
            if len(candidates) > 1:
                raise ChannelError(
                    "channel.receive",
                    "first reply attribution is ambiguous; reader learning required",
                )
            if candidates:
                node, selector = candidates[0]
                self._assistant = LocatorConfig(strategy="css", value=selector)
                try:
                    await self._validate_readers(require_present=True)
                    resolved = await locate(self.owned.locator, self._assistant).element_handles()
                    if not await node.evaluate("(node, nodes) => nodes.includes(node)", resolved):
                        raise ChannelError(
                            "channel.receive", "new assistant node no longer resolves"
                        )
                    snapshot = await self._snapshot("assistant")
                    before_keys = {str(await self._node_key(old)) for old in self._bootstrap_before}
                    self._cursor.observe(
                        "assistant",
                        [part for part in snapshot if part.attribute_id in before_keys],
                        baseline=True,
                    )
                except Exception:
                    self._assistant = None
                    raise
                return
            await asyncio.sleep(0.025)
        raise ChannelError(
            "channel.receive",
            "no uniquely attributed owned assistant reply; reader learning required",
        )

    async def _receive(
        self, *, initial: bool, on_progress: Callable[[str], None] | None
    ) -> CompletedResponse:
        if self._assistant is None:
            raise ChannelError(
                "channel.receive", "assistant reader must be learned before receiving"
            )
        response_options: ReplyOptions | None = None
        if not initial:
            assert self.owned is not None
            # Capture all native controls on the new owned assistant message;
            # this is evidence, not an inferred replay state or locator authority.
            response_options = self._make_reply_options(
                NativeControlBinding(
                    option_locator=LocatorConfig(strategy="css", value="button,[role=button]")
                )
            )
        try:
            if response_options is not None:
                await self._finish_native_read(response_options.bind())
            return await self._receive_completed(
                initial=initial, on_progress=on_progress, response_options=response_options
            )
        finally:
            # Discovery may transfer this inspector into the runtime option owner.
            # All other paths, including timeout/cancellation, release its handles.
            if response_options is not None and response_options is not self._options:
                await self._finish_native_read(response_options.dispose())

    async def _receive_completed(
        self,
        *,
        initial: bool,
        on_progress: Callable[[str], None] | None,
        response_options: ReplyOptions | None,
    ) -> CompletedResponse:
        if self._assistant is None:
            raise ChannelError(
                "channel.receive", "assistant reader must be learned before receiving"
            )
        deadline = time.monotonic() + self._response_window()
        signature: Any = None
        stable_since = time.monotonic()
        self._response_bubble_attached = False
        while time.monotonic() < deadline:
            snapshot = await self._snapshot("assistant")
            pending = self._cursor.observe("assistant", snapshot)
            self._response_bubble_attached = bool(pending)
            # After a dispatch the old composer/options may legitimately disappear.
            # Inspect new menu through immutable reader/root without rebinding controls.
            replies = []
            if response_options is not None:
                replies = await self._finish_native_read(response_options.collect())
            elif self._options and (not self._pending or self._resolved_transition):
                replies = await self._finish_native_read(self._options.collect())
            current = (
                tuple((part.attribute_id, part.content) for part in pending),
                ReplyOptions.signature(replies),
            )
            if current != signature:
                signature, stable_since = current, time.monotonic()
            ready = (
                (
                    time.monotonic() - stable_since
                    >= (self._completion.quiet_interval_seconds or 0.1)
                )
                if initial
                else await self._completion_ready(stable_since)
            )
            if initial and ready and self._completion.locator is not None:
                assert self.owned is not None
                control = locate(self.owned.locator, self._completion.locator)
                if self._completion.strategy == "completion_locator":
                    ready = await control.count() == 1 and await control.is_visible()
                else:
                    ready = not await control.is_visible()
            offered = {
                reply.ordinal
                for reply in replies
                if any(option.reason is None for option in reply.options)
            }
            if (
                pending
                and ready
                and all(part.content.strip() or part.ordinal in offered for part in pending)
            ):
                if not initial and self._reader_evidence == "observed_exchange":
                    if self._exchange is None or self._customer is None:
                        raise ChannelError("channel.reader_evidence", "exchange context is missing")
                    await self._exchange.validate(self._assistant, self._customer)
                if not initial and self.contract and not self._resolved_transition:
                    await self._finish_native_read(self._resolve_transition())
                    self._resolved_transition = True
                    replies = (
                        await self._finish_native_read(self._options.collect())
                        if self._options
                        else []
                    )
                    # Options also need a quiet snapshot after resolving target controls.
                    option_signature = ReplyOptions.signature(replies)
                    await asyncio.sleep(self._completion.quiet_interval_seconds or 0.1)
                    if (
                        self._options
                        and ReplyOptions.signature(
                            await self._finish_native_read(self._options.collect())
                        )
                        != option_signature
                    ):
                        signature = None
                        continue
                    fresh = self._cursor.observe("assistant", await self._snapshot("assistant"))
                    if tuple((part.attribute_id, part.content) for part in fresh) != tuple(
                        (part.attribute_id, part.content) for part in pending
                    ):
                        signature = None
                        continue
                now = datetime.now(UTC)
                ids = self._cursor.finalize("assistant", pending)
                self._completed_ids = {
                    part.ordinal: identifier for part, identifier in zip(pending, ids, strict=True)
                }
                publisher = self._options if self.contract else response_options or self._options
                published = (
                    publisher.publish(
                        replies,
                        {
                            part.ordinal: identifier
                            for part, identifier in zip(pending, ids, strict=True)
                        },
                    )
                    if publisher
                    else {}
                )
                result = CompletedResponse(
                    messages=[
                        Message(
                            id=identifier,
                            sequence=part.ordinal,
                            actor="assistant",
                            content=part.content,
                            observed_at=now,
                            options=published.get(part.ordinal, []),
                        )
                        for part, identifier in zip(pending, ids, strict=True)
                    ],
                    started_at=self._started_at or now,
                    completed_at=now,
                    completion_strategy=self._completion.strategy,
                )
                self._pending = False
                if response_options is not None and self.contract is None:
                    if self._options is not None and self._options is not response_options:
                        await self._options.dispose()
                    self._options = response_options
                self._learn_transition_allowed = not initial and self.contract is None
                self._initial_observed = True
                if self._indicator_task:
                    self._indicator_task.cancel()
                    await asyncio.gather(self._indicator_task, return_exceptions=True)
                    self._indicator_task = None
                return result
            await asyncio.sleep(0.025)
        raise self._response_timeout()

    async def _resolve_transition(self) -> None:
        if isinstance(self.contract, InterfaceChannelConfig):
            # Runtime completion is evidence, never a sampled-state transition.
            return
        assert self.contract is not None and self._state is not None and self._action is not None
        kind, label = self._action
        transitions = [
            item
            for item in self.contract.transitions
            if item.from_state == self._state.id
            and item.action_kind == kind
            and (
                kind == "probe"
                or (
                    item.option_label is not None
                    and label is not None
                    and normalized_label(item.option_label) == normalized_label(label)
                )
            )
        ]
        target_ids = {item.to_state for item in transitions}
        # Replay follows recorded edges exactly. A persona can instead choose
        # another currently guarded option; classify its result using *only*
        # the published state bindings inside the same immutable root.
        candidates = [
            state for state in self.contract.states if not target_ids or state.id in target_ids
        ]
        matches = []
        for state in candidates:
            try:
                await self._inspect_state(state)
            except ChannelError:
                continue
            matches.append(state)
        if len(matches) != 1:
            raise ChannelError("channel.state", "resulting state is unknown or ambiguous")
        await self._bind_typed_state(matches[0])

    def export_config(
        self, states: list[Any], transitions: list[Any], recipe: list[Any]
    ) -> StatefulChannelConfig:
        if self.owned is None or self._initial_state is None or self._assistant is None:
            raise ChannelError(
                "channel.export", "ownership, reader, and initial state are required"
            )
        if self._closed or self._pending or self._uncertain:
            raise ChannelError("channel.export", "broker has an incomplete or uncertain action")
        known = {state.id for state in states if isinstance(state, ChatState)} | {
            state.get("id") for state in states if isinstance(state, dict)
        }
        if not known.issubset(self._states) or any(
            ChatState.model_validate(state) != self._states[ChatState.model_validate(state).id]
            for state in states
        ):
            raise ChannelError("channel.export", "unvisited states cannot be exported")
        # Root locator belongs to the selected frame, child readers are root-relative.
        assert self._root_locator is not None
        root_config = self._root_locator
        config = StatefulChannelConfig(
            type="webchat_playwright_stateful",
            url=self.url,
            open_locator=self._open_locator,
            navigation_steps=list(self._website.steps) if self._website else [],
            frame_locator=self._frame_locator,
            ownership=ChatOwnership(root_locator=root_config, proof=self.owned.proof),
            assistant_message_locator=self._assistant,
            customer_message_locator=self._customer,
            reader_evidence=self._reader_evidence,
            message_identity=self._identity,
            response_completion=self._completion,
            initial_state_id=self._initial_state.id,
            states=states,
            transitions=transitions,
            verification_recipe=recipe,
        )
        if self.contract is None:
            by_id = {item.id: item for item in config.transitions}
            exported = [by_id[step.transition_id] for step in config.verification_recipe]
            edges = [
                (item.from_state, item.to_state, item.action_kind, item.option_label)
                for item in exported
            ]
            if edges != self._learning_edges:
                raise ChannelError(
                    "channel.export", "recipe does not match completed native transitions"
                )
        return config

    async def capture_failure_screenshot(self) -> bytes:
        if self.page is None:
            raise ChannelError("channel.capture", "native page is unavailable")
        return cast(
            bytes,
            await self.page.screenshot(
                type="png", timeout=self.limits.browser_action_timeout_seconds * 1000
            ),
        )

    async def close(self, *, deadline: float | None = None) -> None:
        if self._closing is None:
            limit = time.monotonic() + self.limits.close_timeout_seconds
            if deadline is not None:
                limit = min(limit, deadline)
            limit -= min(0.01, max(0, limit - time.monotonic()) * 0.05)
            self._closing = asyncio.create_task(self._close_resources(limit))
        await asyncio.shield(self._closing)

    async def _close_resources(self, deadline: float) -> None:
        self._closed = True
        if self._options:
            self._options.invalidate()
        if self._indicator_task:
            self._indicator_task.cancel()
            await asyncio.gather(self._indicator_task, return_exceptions=True)
        resources = [
            (resource, method)
            for resource, method in (
                (self, "_release_native_handles"),
                (self._exchange, "close"),
                (self._website, "close"),
                (self._context if not self._external_page else None, "close"),
                (self._browser, "close"),
                (self._playwright, "stop"),
            )
            if resource is not None
        ]
        failed = False
        for index, (resource, method) in enumerate(resources):
            allowance = max(0, deadline - time.monotonic()) / (len(resources) - index)
            try:
                await asyncio.wait_for(getattr(resource, method)(), timeout=allowance)
            except Exception:
                failed = True
        if failed:
            raise ChannelError(
                "channel.close", "native resource cleanup did not complete within budget"
            )

    async def _release_native_handles(self) -> None:
        """Release broker-owned references even when the caller owns the page."""
        handles: list[ElementHandle | None] = [
            *self._known_nodes,
            *self._bootstrap_before,
            *self._marker_nodes,
            self._input,
            self._send,
            self._open_node,
            *(proposal.node for proposal in self._proposals.values()),
        ]
        if self.owned is not None:
            handles.extend([self.owned.root, self.owned.embedded_frame, self.owned.opener])
        if self._options is not None:
            # The attested root and composer may alias broker-owned handles.
            handles.extend([self._options.root, self._options.composer, *self._options._known])
            self._options.invalidate()
            self._options._known.clear()
            self._options.root = self._options.composer = None
        self._known_nodes.clear()
        self._bootstrap_before.clear()
        self._marker_nodes.clear()
        self._proposals.clear()
        self._input = self._send = self._open_node = None
        self.owned = None
        await self._dispose_handles(handles)


# Stable shared name for discovery and runtime factories.
ChatBroker = NativeChatBroker

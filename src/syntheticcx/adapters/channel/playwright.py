"""Per-run async Playwright channel for configured text chat."""

from __future__ import annotations

import asyncio
import time
import unicodedata
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, TypeAlias, cast

from playwright.async_api import (
    Browser,
    BrowserContext,
    FrameLocator,
    Locator,
    Page,
    Playwright,
    async_playwright,
)
from playwright.async_api import Error as PlaywrightError

from syntheticcx.adapters.channel.cursor import Actor, Bubble, MessageCursor
from syntheticcx.config.models import ChannelConfig, ExecutionLimits, LocatorConfig
from syntheticcx.domain.models import CompletedResponse, Message, SendReceipt
from syntheticcx.errors import ChannelError

Scope: TypeAlias = Page | FrameLocator


def normalize_echo(text: str) -> str:
    """Compare visible text ignoring Unicode presentation, case, and whitespace only."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def locate(scope: Scope, config: LocatorConfig, *, include_hidden: bool = False) -> Locator:
    """Resolve only explicitly configured strategies, using semantic locators first."""
    if config.strategy == "role":
        return scope.get_by_role(
            config.value, name=config.name, exact=config.exact, include_hidden=include_hidden
        )
    if config.strategy == "label":
        return scope.get_by_label(config.value, exact=config.exact)
    if config.strategy == "placeholder":
        return scope.get_by_placeholder(config.value, exact=config.exact)
    if config.strategy == "text":
        return scope.get_by_text(config.value, exact=config.exact)
    if config.strategy == "test_id":
        return scope.get_by_test_id(config.value)
    return scope.locator(f"css={config.value}")


class PlaywrightChannel:
    """Own a browser/context/page and a cursor; never expose SDK objects to the domain."""

    def __init__(
        self, config: ChannelConfig, limits: ExecutionLimits, *, headed: bool = False
    ) -> None:
        self._config = config
        self._limits = limits
        self._headed = headed
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._scope: Scope | None = None
        self._cursor = MessageCursor(config.message_identity)
        self._history_count = 0
        self._greeting_received = False
        self._closing: asyncio.Task[None] | None = None
        self._indicator_task: asyncio.Task[bool] | None = None
        self._marker_task: asyncio.Task[bool] | None = None
        self._response_baseline = 0
        self._response_started: datetime | None = None
        self._awaiting_response = False
        self._submission_uncertain = False
        self._submission_count = 0

    @property
    def current_url(self) -> str | None:
        """Observed page URL, without exposing Playwright's page object."""
        return self._page.url if self._page is not None else None

    async def _snapshot(self, actor: Actor) -> list[Bubble]:
        scope = self._scope
        if scope is None:
            raise ChannelError("channel.observe", "channel is not started")
        selector = (
            self._config.assistant_message_locator
            if actor == "assistant"
            else self._config.customer_message_locator
        )
        attribute = self._config.message_identity.attribute_name
        raw: Any = await locate(scope, selector, include_hidden=True).evaluate_all(
            """(nodes, attr) => nodes.map((node, ordinal) => ({
                ordinal, content: node.textContent ?? '',
                attribute_id: attr ? node.getAttribute(attr) : null
            }))""",
            attribute,
        )
        return [
            Bubble(
                actor=actor,
                ordinal=entry["ordinal"],
                content=entry["content"],
                attribute_id=entry["attribute_id"],
            )
            for entry in raw
        ]

    async def start(self, *, on_progress: Callable[[str], None] | None = None) -> None:
        if self._playwright is not None or self._closing is not None:
            raise ChannelError("channel.start", "channel already started")
        try:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=not self._headed)
            self._context = await self._browser.new_context()
            timeout_ms = self._limits.browser_action_timeout_seconds * 1000
            self._context.set_default_timeout(timeout_ms)
            self._context.set_default_navigation_timeout(timeout_ms)
            self._page = await self._context.new_page()
            response = await self._page.goto(
                str(self._config.url), wait_until="load", timeout=timeout_ms
            )
            if response is None:
                raise ChannelError("channel.start", "target navigation returned no HTTP response")
            if response.status >= 400:
                raise ChannelError(
                    "channel.start", f"target navigation returned HTTP {response.status}"
                )
            if on_progress is not None:
                on_progress("native target navigation succeeded")
            if self._config.frame_locator is not None:
                frame = locate(self._page, self._config.frame_locator)
                await frame.wait_for(state="attached", timeout=timeout_ms)
                if await frame.count() != 1:
                    raise ChannelError("channel.start", "configured frame is missing or ambiguous")
                self._scope = frame.content_frame
                await self._scope.locator("body").wait_for(state="attached", timeout=timeout_ms)
            else:
                self._scope = self._page
            # Capture history in both actor streams before activating the opener.
            for actor in ("customer", "assistant"):
                self._cursor.observe(actor, await self._snapshot(actor), baseline=True)
            self._history_count = len(await self._snapshot("assistant"))
            await self._prepare_completion()
            if self._config.open_locator is not None:
                if on_progress is not None:
                    on_progress("waiting for configured chat opener in native browser")
                opener = locate(self._page, self._config.open_locator)
                try:
                    await opener.wait_for(state="visible", timeout=timeout_ms)
                    if await opener.count() != 1:
                        raise ChannelError(
                            "channel.start", "configured chat opener is missing or ambiguous"
                        )
                    await opener.click(timeout=timeout_ms)
                except (PlaywrightError, TimeoutError):
                    raise ChannelError(
                        "channel.start", "configured chat opener could not be activated"
                    ) from None
                if on_progress is not None:
                    on_progress("configured chat opener clicked in native browser")
            chat_input = locate(self._scope, self._config.input_locator)
            await chat_input.wait_for(state="visible", timeout=timeout_ms)
            if await chat_input.count() != 1:
                raise ChannelError("channel.start", "configured chat input is missing or ambiguous")
            if on_progress is not None:
                on_progress("configured chat input visible in native browser")
        except asyncio.CancelledError:
            raise
        except (PlaywrightError, TimeoutError):
            raise ChannelError(
                "channel.start", "browser startup or control resolution failed"
            ) from None

    def _completion_locator(self) -> Locator:
        scope = self._scope
        config = self._config.response_completion.locator
        if scope is None or config is None:
            raise ChannelError("channel.receive", "completion control is not configured")
        return locate(scope, config)

    async def _prepare_completion(self) -> None:
        """Distinguish a new completion element or indicator from an old one."""
        await self._clear_watchers()
        completion = self._config.response_completion
        if completion.strategy == "completion_locator":
            marker = self._completion_locator()
            already_visible = await marker.is_visible()
            await marker.evaluate_all(
                "nodes => nodes.forEach(node => { node.__syntheticcxSeen = true })"
            )
            self._marker_task = asyncio.create_task(
                self._watch_marker(marker, already_visible=already_visible)
            )
        elif completion.strategy == "streaming_indicator_disappears":
            indicator = self._completion_locator()
            if await indicator.is_visible():
                raise ChannelError("channel.receive", "streaming indicator is already active")
            self._indicator_task = asyncio.create_task(self._watch_indicator(indicator))

    async def _watch_indicator(self, indicator: Locator) -> bool:
        try:
            await indicator.wait_for(
                state="visible", timeout=self._limits.response_timeout_seconds * 1000
            )
            return True
        except PlaywrightError:
            return False

    async def _watch_marker(self, marker: Locator, *, already_visible: bool) -> bool:
        try:
            timeout = self._limits.response_timeout_seconds * 1000
            if already_visible:
                await marker.wait_for(state="hidden", timeout=timeout)
            await marker.wait_for(state="visible", timeout=timeout)
            return True
        except PlaywrightError:
            return False

    async def _clear_watchers(self) -> None:
        for task in (self._indicator_task, self._marker_task):
            if task is not None:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self._indicator_task = None
        self._marker_task = None

    async def _new_completion_marker(self) -> bool:
        unseen_visible = (
            "nodes => nodes.some(node => !node.__syntheticcxSeen && node.getClientRects().length)"
        )
        new_node = bool(await self._completion_locator().evaluate_all(unseen_visible))
        return new_node or bool(
            self._marker_task is not None
            and self._marker_task.done()
            and self._marker_task.result()
        )

    async def send(
        self, message: str, *, on_progress: Callable[[str], None] | None = None
    ) -> SendReceipt:
        scope = self._scope
        if scope is None or self._closing is not None:
            raise ChannelError("channel.send", "channel is not available")
        if self._awaiting_response or self._submission_uncertain:
            raise ChannelError("channel.send", "previous submission is not resolved")
        if not message.strip():
            raise ChannelError("channel.send", "customer message is empty")
        deadline = time.monotonic() + self._limits.browser_action_timeout_seconds
        stage = "customer message baseline observation"
        try:
            async with asyncio.timeout(self._limits.browser_action_timeout_seconds):
                customer_snapshot = await self._snapshot("customer")
                customer_before = self._cursor.observe("customer", customer_snapshot)
                if customer_before:
                    raise ChannelError("channel.send", "unexpected customer echo before submission")
                stage = "assistant message baseline observation"
                self._response_baseline = len(await self._snapshot("assistant"))
                stage = "response completion preparation"
                await self._prepare_completion()
                stage = "configured chat input fill"
                if on_progress is not None:
                    on_progress("filling configured chat input for synthetic probe")
                input_box = locate(scope, self._config.input_locator)
                await input_box.fill(message)
                # A UI action may have occurred even if Playwright reports a failure later.
                self._submission_uncertain = True
                strategy = self._config.send_strategy
                stage = "configured send control activation"
                if on_progress is not None:
                    on_progress("activating configured send control for synthetic probe")
                if strategy.type == "press_enter":
                    await input_box.press("Enter")
                else:
                    assert strategy.locator is not None
                    send_control = locate(scope, strategy.locator)
                    await send_control.wait_for(state="visible")
                    if await send_control.count() != 1:
                        raise ChannelError(
                            "channel.send", "configured send control is missing or ambiguous"
                        )
                    await send_control.click()
                submitted = datetime.now(UTC)
                self._submission_count += 1
                if on_progress is not None:
                    on_progress("send action completed; observing optional customer echo")
                remaining = max(0, deadline - time.monotonic())
                # Leave time within the action deadline for the final DOM snapshot.
                echo_window = min(
                    self._limits.echo_observation_seconds,
                    max(0, remaining - min(0.05, remaining * 0.2)),
                )
                echoes: list[Bubble] = []
                stage = "customer echo observation"
                if echo_window > 0:
                    await asyncio.sleep(echo_window)
                    echoes = self._cursor.observe("customer", await self._snapshot("customer"))
                if len(echoes) > 1:
                    raise ChannelError("channel.send", "customer echo is ambiguous")
                if echoes and normalize_echo(echoes[0].content) != normalize_echo(message):
                    raise ChannelError("channel.send", "customer echo differs from submission")
                evidence: dict[str, str] = {"echo": "absent"}
                receipt_id = f"customer:submitted-{self._submission_count}"
                if echoes:
                    receipt_id = self._cursor.finalize("customer", echoes)[0]
                    evidence = {"echo": "matched", "dom_message_id": receipt_id}
                self._response_started = submitted
                self._awaiting_response = True
                self._submission_uncertain = False
                return SendReceipt(message_id=receipt_id, submitted_at=submitted, evidence=evidence)
        except asyncio.CancelledError:
            raise
        except (PlaywrightError, TimeoutError):
            raise ChannelError("channel.send", f"{stage} failed") from None

    async def receive(self) -> CompletedResponse:
        scope = self._scope
        if scope is None or self._closing is not None:
            raise ChannelError("channel.receive", "channel is not available")
        if not self._awaiting_response and self._greeting_received:
            raise ChannelError("channel.receive", "no response is pending")
        completion = self._config.response_completion
        try:
            async with asyncio.timeout(self._limits.response_timeout_seconds):
                baseline = (
                    self._response_baseline if self._awaiting_response else self._history_count
                )
                await (
                    locate(scope, self._config.assistant_message_locator, include_hidden=True)
                    .nth(baseline)
                    .wait_for(state="attached")
                )
                last_snapshot: tuple[tuple[str | None, str], ...] = ()
                stable_since = time.monotonic()
                fragments: dict[int, int] = {}
                while True:
                    observed = self._cursor.observe("assistant", await self._snapshot("assistant"))
                    pending = [part for part in observed if part.ordinal >= baseline]
                    signature = tuple((part.attribute_id, part.content) for part in pending)
                    if signature != last_snapshot:
                        for part in pending:
                            if (part.attribute_id, part.content) not in last_snapshot:
                                fragments[part.ordinal] = fragments.get(part.ordinal, 0) + 1
                        last_snapshot = signature
                        stable_since = time.monotonic()
                    if pending:
                        if completion.strategy == "quiet_interval":
                            assert completion.quiet_interval_seconds is not None
                            ready = (
                                time.monotonic() - stable_since >= completion.quiet_interval_seconds
                            )
                        elif completion.strategy == "completion_locator":
                            ready = await self._new_completion_marker()
                        else:
                            task = self._indicator_task
                            ready = bool(
                                task is not None
                                and task.done()
                                and task.result()
                                and not await self._completion_locator().is_visible()
                            )
                        if ready:
                            now = datetime.now(UTC)
                            ids = self._cursor.finalize("assistant", pending)
                            self._greeting_received = True
                            self._awaiting_response = False
                            await self._clear_watchers()
                            return CompletedResponse(
                                messages=[
                                    Message(
                                        id=identifier,
                                        sequence=part.ordinal,
                                        actor="assistant",
                                        content=part.content,
                                        observed_at=now,
                                    )
                                    for identifier, part in zip(ids, pending)
                                ],
                                started_at=self._response_started or now,
                                completed_at=now,
                                completion_strategy=completion.strategy,
                                streaming_metadata={
                                    identifier: {"observed_fragments": fragments[part.ordinal]}
                                    for identifier, part in zip(ids, pending)
                                    if fragments.get(part.ordinal, 0) > 1
                                },
                            )
                    await asyncio.sleep(0.025)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            raise ChannelError("channel.receive", "assistant response timed out") from None
        except PlaywrightError:
            raise ChannelError("channel.receive", "assistant observation failed") from None

    async def close(self) -> None:
        """Start one bounded cleanup that continues after caller cancellation."""
        if self._closing is None:
            self._closing = asyncio.create_task(self._close_resources())
        await asyncio.shield(self._closing)

    async def capture_failure_screenshot(self) -> bytes:
        """Capture in memory; the runner decides whether storage is permitted."""
        if self._page is None or self._closing is not None:
            raise ChannelError("channel.capture", "browser page is unavailable")
        try:
            return cast(
                bytes,
                await self._page.screenshot(
                    type="png", timeout=self._limits.browser_action_timeout_seconds * 1000
                ),
            )
        except PlaywrightError:
            raise ChannelError("channel.capture", "browser screenshot failed") from None

    async def _close_resources(self) -> None:
        await self._clear_watchers()
        context, browser, playwright = self._context, self._browser, self._playwright
        self._context = None
        self._browser = None
        self._playwright = None
        self._page = None
        self._scope = None
        failed = False
        deadline = time.monotonic() + self._limits.close_timeout_seconds
        for resource in (context, browser, playwright):
            if resource is None:
                continue
            try:
                if isinstance(resource, Playwright):
                    await asyncio.wait_for(resource.stop(), max(0.001, deadline - time.monotonic()))
                else:
                    await asyncio.wait_for(
                        resource.close(), max(0.001, deadline - time.monotonic())
                    )
            except Exception:
                failed = True
        if failed:
            raise ChannelError("channel.close", "browser cleanup failed")

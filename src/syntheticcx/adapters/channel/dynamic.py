"""Current native controls, without discovery states, edges, or recipe autoplay."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

from playwright.async_api import ElementHandle, Page

from syntheticcx.adapters.channel.broker import NativeChatBroker
from syntheticcx.adapters.channel.options import ReplyOptions
from syntheticcx.adapters.channel.playwright import locate
from syntheticcx.config.models import (
    ChatState,
    ExecutionLimits,
    InterfaceChannelConfig,
    LocatorConfig,
    SendStrategyConfig,
    StatefulChannelConfig,
)
from syntheticcx.domain.models import CompletedResponse, ReplyOption, SelectionReceipt, SendReceipt
from syntheticcx.errors import ChannelError

COMPOSERS = "textarea,input:not([type]),input[type=text],[role=textbox],[contenteditable=true]"
BUTTONS = "button,[role=button]"


class DynamicChatBroker(NativeChatBroker):
    """Reuse native transactions; only the current owned DOM authorizes controls."""

    def __init__(
        self,
        config: InterfaceChannelConfig,
        limits: ExecutionLimits,
        *,
        headed: bool = False,
        page: Page | None = None,
        action_authorizations: list[dict[str, Any]] | None = None,
    ) -> None:
        self._action_authorizations = action_authorizations or []
        # Shared initialization/startup only reads the interface portion of the
        # contract. Replay-specific policies are overridden below.
        super().__init__(
            contract=cast(StatefulChannelConfig, config), limits=limits, headed=headed, page=page
        )

    async def _bootstrap_contract(self) -> None:
        await self._refresh_controls(wait=True)

    async def _current_controls(self) -> tuple[ElementHandle | None, ElementHandle | None]:
        await self._check_parent()
        await self._validate_readers(require_present=bool(self._completed_ids))
        assert self.owned is not None
        inputs = await self.owned.locator.locator(COMPOSERS).element_handles()
        buttons: list[ElementHandle] = []
        retained: tuple[ElementHandle | None, ElementHandle | None] = (None, None)
        try:
            buttons = await self.owned.locator.locator(BUTTONS).element_handles()
            retained = await self._inspect_controls(inputs, buttons)
            return retained
        finally:
            # Returned references transfer to the caller; every other inspection
            # handle, including invisible/rejected controls, remains temporary.
            await self._dispose_handles(
                [node for node in [*inputs, *buttons] if all(node is not keep for keep in retained)]
            )

    async def _inspect_controls(
        self, input_handles: list[ElementHandle], buttons: list[ElementHandle]
    ) -> tuple[ElementHandle | None, ElementHandle | None]:
        inputs = [node for node in input_handles if await node.is_visible()]
        if len(inputs) > 1:
            raise ChannelError("channel.controls", "current composer is ambiguous")
        input_node = inputs[0] if inputs else None
        if input_node is not None and not await self._safe_current_control(
            input_node, is_input=True
        ):
            raise ChannelError("channel.controls", "current composer is unsafe or unavailable")
        send_candidates = [
            node
            for node in buttons
            if await node.is_visible()
            and await node.evaluate(
                """node => /^(send|enviar)( message| mensagem)?$/i.test(
                  (node.getAttribute('aria-label') || node.innerText || '').trim())"""
            )
        ]
        if len(send_candidates) > 1:
            raise ChannelError("channel.controls", "current send control is ambiguous")
        send_node = send_candidates[0] if send_candidates and input_node is not None else None
        if send_node is not None and not await self._safe_current_control(
            send_node, is_input=False, allow_disabled_send=True
        ):
            raise ChannelError("channel.controls", "current send control is unsafe or unavailable")
        return input_node, send_node

    async def _refresh_controls(self, *, wait: bool) -> None:
        if self._uncertain:
            raise ChannelError("channel.controls", "native action is uncertain; retry prohibited")
        deadline = time.monotonic() + self.limits.browser_action_timeout_seconds * 0.9
        if self._pending and self._response_deadline is not None:
            margin = min(0.1, self.limits.response_timeout_seconds * 0.1)
            deadline = min(deadline, self._response_deadline - margin - 0.025)
        while True:
            input_node, send_node = await self._current_controls()
            try:
                if await self._bind_current_controls(input_node, send_node):
                    return
            finally:
                await self._dispose_handles(
                    [
                        node
                        for node in [input_node, send_node]
                        if node is not self._input and node is not self._send
                    ]
                )
            if not wait or time.monotonic() >= deadline:
                raise ChannelError("channel.controls", "no safe current controls before deadline")
            await asyncio.sleep(0.025)

    def _make_reply_options(self, state: ChatState) -> ReplyOptions:
        assert self.owned is not None
        return ReplyOptions(
            self.owned.locator,
            self._option_config(state),
            locate,
            attested_scope=self.owned,
            dynamic_runtime=True,
            authorizations=self._action_authorizations,
        )

    async def _bind_current_controls(
        self, input_node: ElementHandle | None, send_node: ElementHandle | None
    ) -> bool:
        assert self.owned is not None
        # Options deliberately do not pin a composer: they are independently
        # pinned to their actual latest assistant owner and the immutable root.
        if self._options is None:
            evidence = SimpleNamespace(
                input_locator=None,
                send_strategy=None,
                option_locator=LocatorConfig(strategy="css", value=BUTTONS),
            )
            self._options = self._make_reply_options(cast(ChatState, evidence))
            await self._options.bind()
        replies = await self._options.collect()
        latest_options = bool(
            replies and any(option.reason is None for option in replies[-1].options)
        )
        if input_node is not None or latest_options:
            superseded = [self._input, self._send]
            self._input, self._send = input_node, send_node
            await self._dispose_handles(
                [node for node in superseded if node is not self._input and node is not self._send]
            )
            # This ephemeral binding is not a discovery state or label graph.
            self._state = cast(
                ChatState,
                SimpleNamespace(
                    input_locator=LocatorConfig(strategy="css", value=COMPOSERS)
                    if input_node is not None
                    else None,
                    send_strategy=SendStrategyConfig(
                        type="click", locator=LocatorConfig(strategy="css", value=BUTTONS)
                    )
                    if send_node is not None
                    else SendStrategyConfig(type="press_enter")
                    if input_node is not None
                    else None,
                ),
            )
            # Reuse all shared safety checks, but resolve the pinned send by
            # actual identity rather than a selector encompassing menu buttons.
            await self._validate_current_controls()
            if self._completed_ids and not self._pending and not self._options.registry:
                fresh = self._cursor.observe("assistant", await self._snapshot("assistant"))
                if not fresh:
                    self._options.publish(replies, self._completed_ids)
            return True
        return False

    async def _validate_current_controls(self) -> None:
        # Never refresh here: this also runs after fill, with uncertainty latched.
        input_node, send_node = await self._current_controls()
        try:
            await self._validate_pinned_controls(input_node, send_node)
        finally:
            await self._dispose_handles([input_node, send_node])

    async def _validate_pinned_controls(
        self, input_node: ElementHandle | None, send_node: ElementHandle | None
    ) -> None:
        for current, retained in [(input_node, self._input), (send_node, self._send)]:
            if (current is None) != (retained is None) or (
                current is not None
                and retained is not None
                and not await retained.evaluate(
                    "(node, current) => node === current && node.isConnected", current
                )
            ):
                raise ChannelError("channel.send", "pinned current control changed")
        # Shared ancestry/transaction guards still apply to the retained node.
        if send_node is not None:
            # Before fill, disabled is a readiness condition, not structural
            # rejection. Once fill dispatches input handlers, this exact retained
            # Send must be enabled; failure stays uncertain and cannot retry.
            safe = await self._safe_current_control(
                send_node, is_input=False, allow_disabled_send=not self._uncertain
            )
            if not safe:
                raise ChannelError("channel.send", "current send control is unsafe or unavailable")

    async def _check_dispatch(self) -> None:
        await self._check_parent()
        if self._pending or self._uncertain:
            raise ChannelError("channel.send", "dispatch is pending or uncertain")
        await self._refresh_controls(wait=True)
        await super()._check_dispatch()

    def _guard_transition(self, kind: str, label: str | None) -> None:
        if kind == "probe" and self._input is None:
            raise ChannelError("channel.send", "current controls have no safe composer")
        # Option authority is exclusively ReplyOptions.validate, never sample labels.

    async def _resolve_transition(self) -> None:
        # Completed replies must reach the journal even when the next controls
        # are missing. observe_controls gates the next persona decision instead.
        pass

    async def send(
        self, message: str, *, on_progress: Callable[[str], None] | None = None
    ) -> SendReceipt:
        result = await super().send(message, on_progress=on_progress)
        self._learning_source = None
        return result

    async def select_option(self, identifier: str) -> SelectionReceipt:
        result = await super().select_option(identifier)
        self._learning_source = None
        return result

    async def observe_controls(self) -> dict[str, Any]:
        async with self._lock:
            if self._pending or self._uncertain:
                raise ChannelError("channel.controls", "dispatch is pending or uncertain")
            await self._refresh_controls(wait=True)
            await self.observe_initial()
            assert self._options is not None
            await self._options.collect()
            return {
                "text_available": self._input is not None,
                "options": [
                    # Only public opaque capabilities leave the native broker.
                    ReplyOption(id=identifier, label=cap.option.label)
                    for identifier, cap in self._options.registry.items()
                ],
            }


class DynamicPlaywrightChannel:
    def __init__(
        self,
        config: InterfaceChannelConfig,
        limits: ExecutionLimits,
        *,
        headed: bool = False,
        action_authorizations: list[dict[str, Any]] | None = None,
    ) -> None:
        self.broker = DynamicChatBroker(
            config, limits, headed=headed, action_authorizations=action_authorizations
        )

    @property
    def response_bubble_attached(self) -> bool:
        return self.broker.response_bubble_attached

    @property
    def current_url(self) -> str | None:
        return self.broker.current_url

    async def start(self, *, on_progress: Callable[[str], None] | None = None) -> None:
        await self.broker.start(on_progress=on_progress)

    async def observe_initial(
        self, *, on_progress: Callable[[str], None] | None = None
    ) -> CompletedResponse | None:
        return await self.broker.observe_initial(on_progress=on_progress)

    async def observe_controls(self) -> dict[str, Any]:
        return await self.broker.observe_controls()

    async def send(
        self, message: str, *, on_progress: Callable[[str], None] | None = None
    ) -> SendReceipt:
        return await self.broker.send(message, on_progress=on_progress)

    async def select_option(self, identifier: str) -> SelectionReceipt:
        return await self.broker.select_option(identifier)

    async def receive(
        self, *, on_progress: Callable[[str], None] | None = None
    ) -> CompletedResponse:
        return await self.broker.receive(on_progress=on_progress)

    async def capture_failure_screenshot(self) -> bytes:
        return await self.broker.capture_failure_screenshot()

    async def close(self) -> None:
        await self.broker.close()

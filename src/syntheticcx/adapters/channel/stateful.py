"""Stateful runtime channel; delegates every action to the native broker."""

from __future__ import annotations

from collections.abc import Callable

from syntheticcx.adapters.channel.broker import NativeChatBroker
from syntheticcx.config.models import ExecutionLimits, StatefulChannelConfig
from syntheticcx.domain.models import CompletedResponse, SelectionReceipt, SendReceipt


class StatefulPlaywrightChannel:
    """No probing, model calls, recipe autoplay, or legacy fallback at startup."""

    def __init__(
        self, config: StatefulChannelConfig, limits: ExecutionLimits, *, headed: bool = False
    ) -> None:
        self.broker = NativeChatBroker(contract=config, limits=limits, headed=headed)

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


RuntimeStatefulPlaywrightChannel = StatefulPlaywrightChannel

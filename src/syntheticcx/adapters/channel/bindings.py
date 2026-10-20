"""Native transaction bindings, independent of discovery states and sample edges."""

from dataclasses import dataclass

from syntheticcx.config.models import (
    ChatState,
    LocatorConfig,
    ReplyOptionsConfig,
    SendStrategyConfig,
)


@dataclass(frozen=True)
class NativeControlBinding:
    input_locator: LocatorConfig | None = None
    send_strategy: SendStrategyConfig | None = None
    option_locator: LocatorConfig | None = None

    @classmethod
    def from_state(cls, state: ChatState) -> "NativeControlBinding":
        return cls(state.input_locator, state.send_strategy, state.option_locator)


@dataclass(frozen=True)
class NativeOptionConfig:
    url: str
    input_locator: LocatorConfig | None
    assistant_message_locator: LocatorConfig
    customer_message_locator: LocatorConfig
    send_strategy: SendStrategyConfig
    reply_options: ReplyOptionsConfig

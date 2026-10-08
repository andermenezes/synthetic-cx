"""Controller decisions and private, in-memory learning results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from syntheticcx.config.models import (
    ChatOwnership,
    ChatState,
    ChatTransition,
    RecipeStep,
    StatefulChannelConfig,
    StrictModel,
)
from syntheticcx.domain.models import CompletedResponse, SendReceipt


class DiscoveryDecision(StrictModel):
    action: Literal[
        "observe",
        "wait",
        "open",
        "bind_scope",
        "bind_reader",
        "bind_state",
        "submit_probe",
        "select_option",
        "finish",
        "stop",
    ]
    revision: str
    target_id: str | None
    assistant_id: str | None
    customer_id: str | None
    input_id: str | None
    send_id: str | None
    option_id: str | None
    send_mode: Literal["press_enter", "click"] | None
    probe_id: Literal["first", "second", "third", "explicit"] | None
    stop_reason: (
        Literal[
            "no_supported_controls",
            "missing_reader_evidence",
            "ambiguous_evidence",
            "unsafe_interface",
            "budget_exhausted",
            "unsupported_interface",
        ]
        | None
    )


class LearningBroker(Protocol):
    @property
    def current_state(self) -> ChatState | None: ...

    async def start(self) -> None: ...
    async def close(self) -> None: ...
    async def observe(self) -> dict[str, Any]: ...
    async def wait_until_ready(self) -> None: ...
    async def wait_for_reader_candidates(self) -> None: ...
    async def open(self, revision: str, target_id: str, /) -> None: ...
    async def bind_scope(self, revision: str, target_id: str, /) -> ChatOwnership: ...
    async def bind_reader(
        self, *, revision: str, assistant_id: str, customer_id: str | None
    ) -> dict[str, Any]: ...
    async def bind_state(
        self,
        *,
        revision: str,
        input_id: str | None,
        send_id: str | None,
        option_id: str | None,
        send_mode: str | None,
    ) -> ChatState: ...
    async def submit_probe(self, revision: str, target_id: str, text: str, /) -> SendReceipt: ...
    async def select_option(self, option_id: str, /) -> SendReceipt: ...
    async def receive(self) -> CompletedResponse: ...
    async def observe_initial(self) -> CompletedResponse | None: ...
    def export_config(
        self, states: list[ChatState], transitions: list[ChatTransition], recipe: list[RecipeStep]
    ) -> StatefulChannelConfig: ...


@dataclass(repr=False)
class LearnResult:
    config: StatefulChannelConfig
    probes: dict[str, str] = field(repr=False)

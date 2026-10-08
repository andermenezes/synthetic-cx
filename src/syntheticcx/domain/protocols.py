"""Interfaces implemented by future adapters."""

from __future__ import annotations

from collections.abc import Awaitable
from datetime import datetime
from typing import Protocol, TypedDict, TypeVar, runtime_checkable

from syntheticcx.domain.artifacts import ConfigArtifact, EvaluationContext, Run
from syntheticcx.domain.events import TypedTraceEvent
from syntheticcx.domain.models import (
    CompletedResponse,
    ConversationObservation,
    CustomerScenarioContext,
    EvaluationResult,
    LLMRequest,
    LLMResponse,
    PersonaAction,
    PersonaDefinition,
    PersonaFeedbackResponse,
    PersonaReflectionRequest,
    PersonaState,
    ReplyOption,
    SelectionReceipt,
    SendReceipt,
)

T = TypeVar("T")


class ChannelAdapter(Protocol):
    async def start(self) -> None: ...
    async def send(self, message: str) -> SendReceipt: ...
    async def receive(self) -> CompletedResponse: ...
    async def close(self) -> None: ...


@runtime_checkable
class OptionSelectingChannel(Protocol):
    """Optional capability; ordinary text channels need not implement it."""

    async def select_option(self, option_id: str) -> SelectionReceipt: ...


@runtime_checkable
class InitialObservingChannel(Protocol):
    """Current startup context, separate from any later submission response baseline."""

    async def observe_initial(self) -> CompletedResponse | None: ...


class ObservedControls(TypedDict):
    text_available: bool
    options: list[ReplyOption]


@runtime_checkable
class ControlsObservingChannel(Protocol):
    """Optional live controls; no DOM or executable targets cross this seam."""

    async def observe_controls(self) -> ObservedControls: ...


class PersonaSimulator(Protocol):
    async def initialize(
        self, definition: PersonaDefinition, scenario_context: CustomerScenarioContext
    ) -> PersonaState: ...
    async def respond(
        self, observation: ConversationObservation, state: PersonaState
    ) -> PersonaAction: ...


@runtime_checkable
class PersonaReflector(Protocol):
    @property
    def reflection_available(self) -> bool: ...
    async def reflect(self, request: PersonaReflectionRequest) -> PersonaFeedbackResponse: ...


class Evaluator(Protocol):
    async def evaluate(self, context: EvaluationContext) -> EvaluationResult: ...


class StorageBackend(Protocol):
    async def create_run(self, run: Run, config_artifact: ConfigArtifact) -> None: ...
    async def append_event(self, run_id: str, event: TypedTraceEvent) -> None: ...
    async def finalize_run(self, run: Run) -> None: ...


class LLMProvider(Protocol):
    async def generate(self, request: LLMRequest) -> LLMResponse: ...


class Clock(Protocol):
    def now(self) -> datetime: ...


class WaitController(Protocol):
    async def run(self, operation: Awaitable[T], timeout: float) -> T: ...

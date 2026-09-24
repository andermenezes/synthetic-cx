"""Interfaces implemented by future adapters."""

from __future__ import annotations

from collections.abc import Awaitable
from datetime import datetime
from typing import Protocol, TypeVar

from syntheticcx.domain.artifacts import EvaluationContext, Run
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
    PersonaState,
    SendReceipt,
)

T = TypeVar("T")


class ChannelAdapter(Protocol):
    async def start(self) -> None: ...
    async def send(self, message: str) -> SendReceipt: ...
    async def receive(self) -> CompletedResponse: ...
    async def close(self) -> None: ...


class PersonaSimulator(Protocol):
    async def initialize(
        self, definition: PersonaDefinition, scenario_context: CustomerScenarioContext
    ) -> PersonaState: ...
    async def respond(
        self, observation: ConversationObservation, state: PersonaState
    ) -> PersonaAction: ...


class Evaluator(Protocol):
    async def evaluate(self, context: EvaluationContext) -> EvaluationResult: ...


class StorageBackend(Protocol):
    async def create_run(self, run: Run) -> None: ...
    async def append_event(self, run_id: str, event: TypedTraceEvent) -> None: ...
    async def finalize_run(self, run: Run) -> None: ...


class LLMProvider(Protocol):
    async def generate(self, request: LLMRequest) -> LLMResponse: ...


class Clock(Protocol):
    def now(self) -> datetime: ...


class WaitController(Protocol):
    async def run(self, operation: Awaitable[T], timeout: float) -> T: ...

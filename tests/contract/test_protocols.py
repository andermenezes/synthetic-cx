"""Static substitution checks and SDK-free public annotation checks."""

import inspect
from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import TypeVar, get_args, get_type_hints

from syntheticcx.adapters.storage.filesystem import FilesystemStorage
from syntheticcx.domain import protocols
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
    PersonaState,
    ReplyOption,
    SendReceipt,
)

T = TypeVar("T")


class FakeAdapters:
    """In-memory protocol signatures checked by mypy, with no SDK dependencies."""

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def send(self, message: str) -> SendReceipt:
        return SendReceipt(message_id="submitted", submitted_at=self.now())

    async def receive(self) -> CompletedResponse:
        raise NotImplementedError

    async def initialize(
        self, definition: PersonaDefinition, scenario_context: CustomerScenarioContext
    ) -> PersonaState:
        return definition.fresh_state()

    async def respond(
        self, observation: ConversationObservation, state: PersonaState
    ) -> PersonaAction:
        return PersonaAction(state=state, signal="goal_believed_complete")

    async def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        raise NotImplementedError

    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError

    async def create_run(self, run: Run, config_artifact: ConfigArtifact) -> None:
        return None

    async def append_event(self, run_id: str, event: TypedTraceEvent) -> None:
        return None

    async def finalize_run(self, run: Run) -> None:
        return None

    def now(self) -> datetime:
        return datetime.now(UTC)

    async def run(self, operation: Awaitable[T], timeout: float) -> T:
        return await operation


class FakeControlsChannel(FakeAdapters):
    async def observe_controls(self) -> protocols.ObservedControls:
        return {"text_available": False, "options": [ReplyOption(id="choice", label="Billing")]}


channel: protocols.ChannelAdapter = FakeAdapters()
controls_channel: protocols.ControlsObservingChannel = FakeControlsChannel()
persona: protocols.PersonaSimulator = FakeAdapters()
evaluator: protocols.Evaluator = FakeAdapters()
provider: protocols.LLMProvider = FakeAdapters()
storage: protocols.StorageBackend = FakeAdapters()
filesystem_type: type[protocols.StorageBackend] = FilesystemStorage
clock: protocols.Clock = FakeAdapters()
wait: protocols.WaitController = FakeAdapters()


def test_protocol_annotations_expose_no_external_sdk_types() -> None:
    forbidden = ("tinytroupe", "playwright", "langgraph", "openai", "opentelemetry", "typer")

    def inspect_annotation(annotation: object) -> None:
        module = getattr(annotation, "__module__", "")
        assert not module.startswith(forbidden)
        for argument in get_args(annotation):
            inspect_annotation(argument)

    for name in [
        "ChannelAdapter",
        "ControlsObservingChannel",
        "PersonaSimulator",
        "Evaluator",
        "LLMProvider",
        "StorageBackend",
        "Clock",
        "WaitController",
    ]:
        for _, method in inspect.getmembers(getattr(protocols, name), inspect.isfunction):
            for annotation in get_type_hints(method).values():
                inspect_annotation(annotation)


def test_current_controls_are_optional_for_legacy_channel_adapters() -> None:
    assert not isinstance(FakeAdapters(), protocols.ControlsObservingChannel)
    assert isinstance(FakeControlsChannel(), protocols.ControlsObservingChannel)

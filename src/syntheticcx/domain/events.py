"""Typed durable trace envelopes and behavioral batch reference validation."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import UUID4, Field, model_validator

from syntheticcx.config.models import BehaviorIdentifier, Criterion, JsonObject
from syntheticcx.domain.behavior import BehavioralSignal, EvidenceReference
from syntheticcx.domain.models import (
    ArtifactRecord,
    BehavioralStateTransition,
    DomainModel,
    ErrorDetail,
    EvaluationResult,
    Identifier,
    Lifecycle,
    Message,
    ModelInteraction,
    Outcome,
    PersonaFeedback,
    PersonaState,
    PersonaStateTransition,
    UtcDatetime,
)


class RunLifecycleChange(DomainModel):
    previous: Lifecycle | None = None
    current: Lifecycle
    termination_reason: str | None = None
    outcome: Outcome | None = None
    conversation_duration_seconds: float | None = Field(default=None, ge=0)


class ConversationStarted(DomainModel):
    greeting_observed: bool
    grace_seconds: float = Field(ge=0)
    greeting_message_ids: list[Identifier] = Field(default_factory=list)


class MessageObserved(DomainModel):
    message: Message
    identity_evidence: JsonObject = Field(default_factory=dict)


class RepeatedInformationObserved(DomainModel):
    information_id: Identifier
    original_message_id: Identifier
    request_message_id: Identifier


class BehavioralObservationRecorded(DomainModel):
    """Validated batch marker; signal_count makes interrupted commits observable."""

    observation_id: UUID4
    kind: Literal["greeting", "customer_turn_completed"]
    evidence_event_ids: list[EvidenceReference] = Field(min_length=1)
    message_ids: list[EvidenceReference] = Field(default_factory=list)
    detector_ids: list[BehaviorIdentifier] = Field(default_factory=list, max_length=8)
    state: PersonaState
    turn_count: int = Field(ge=0, strict=True)
    signal_count: int = Field(ge=0, le=256, strict=True)

    @model_validator(mode="after")
    def validate_boundary(self) -> "BehavioralObservationRecorded":
        for values in (self.evidence_event_ids, self.message_ids, self.detector_ids):
            if len(values) != len(set(values)):
                raise ValueError("behavioral observation references must be unique")
        if self.signal_count > 32 * len(self.detector_ids):
            raise ValueError("signal count exceeds configured detector output bounds")
        if self.kind == "greeting" and self.turn_count != 0:
            raise ValueError("greeting cannot complete a customer turn")
        if self.kind == "customer_turn_completed" and (
            self.turn_count == 0 or not self.message_ids
        ):
            raise ValueError("completed turn requires response messages")
        return self


class BehavioralSignalRecorded(DomainModel):
    signal: BehavioralSignal


class PersonaOutputAmbiguityObserved(DomainModel):
    """Audit competing persona actions and the deterministic resolution."""

    candidate_action_count: int = Field(ge=2)
    selected_action_order: Literal[1] = 1
    discarded_action_count: int = Field(ge=1)
    resolution: Literal["first_external_action_selected"] = "first_external_action_selected"


class EvaluationCompleted(DomainModel):
    result: EvaluationResult
    criterion: Criterion


class OperationFailure(DomainModel):
    error: ErrorDetail
    related_event_ids: list[Identifier] = Field(default_factory=list)


class EventEnvelope(DomainModel):
    id: Identifier
    sequence: int = Field(ge=0)
    timestamp: UtcDatetime


class RunLifecycleEvent(EventEnvelope):
    type: Literal["run_lifecycle_changed"]
    data: RunLifecycleChange


class ConversationStartedEvent(EventEnvelope):
    type: Literal["conversation_started"]
    data: ConversationStarted


class MessageObservedEvent(EventEnvelope):
    type: Literal["message_observed"]
    data: MessageObserved


class PersonaStateChangedEvent(EventEnvelope):
    type: Literal["persona_state_changed"]
    data: PersonaStateTransition


class RepeatedInformationEvent(EventEnvelope):
    type: Literal["repeated_information"]
    data: RepeatedInformationObserved


class BehavioralObservationRecordedEvent(EventEnvelope):
    type: Literal["behavior_observation_recorded"]
    data: BehavioralObservationRecorded


class BehavioralSignalRecordedEvent(EventEnvelope):
    type: Literal["behavioral_signal_recorded"]
    data: BehavioralSignalRecorded


class BehavioralStateChangedEvent(EventEnvelope):
    type: Literal["behavioral_state_changed"]
    data: BehavioralStateTransition


class PersonaOutputAmbiguityEvent(EventEnvelope):
    type: Literal["persona_output_ambiguity_observed"]
    data: PersonaOutputAmbiguityObserved


class ModelInteractionCompletedEvent(EventEnvelope):
    type: Literal["model_interaction_completed"]
    data: ModelInteraction


class EvaluationCompletedEvent(EventEnvelope):
    type: Literal["evaluation_completed"]
    data: EvaluationCompleted


class OperationFailedEvent(EventEnvelope):
    type: Literal["operation_failed"]
    data: OperationFailure


class ArtifactRecordedEvent(EventEnvelope):
    type: Literal["artifact_recorded"]
    data: ArtifactRecord


class PersonaFeedbackRecordedEvent(EventEnvelope):
    type: Literal["persona_feedback_recorded"]
    data: PersonaFeedback


TypedTraceEvent = Annotated[
    RunLifecycleEvent
    | ConversationStartedEvent
    | MessageObservedEvent
    | PersonaStateChangedEvent
    | RepeatedInformationEvent
    | BehavioralObservationRecordedEvent
    | BehavioralSignalRecordedEvent
    | BehavioralStateChangedEvent
    | PersonaOutputAmbiguityEvent
    | ModelInteractionCompletedEvent
    | EvaluationCompletedEvent
    | OperationFailedEvent
    | ArtifactRecordedEvent
    | PersonaFeedbackRecordedEvent,
    Field(discriminator="type"),
]


def committed_errors(
    events: list[TypedTraceEvent] | tuple[TypedTraceEvent, ...],
) -> list[ErrorDetail]:
    """Return detached operation failures in durable trace order."""
    return [
        event.data.error.model_copy(deep=True)
        for event in events
        if isinstance(event, OperationFailedEvent)
    ]


@dataclass(frozen=True)
class CommittedBehaviorBatch:
    """One fully acknowledged observation, signal list, and transition list."""

    observation: BehavioralObservationRecordedEvent
    signals: tuple[BehavioralSignalRecordedEvent, ...]
    transitions: tuple[BehavioralStateChangedEvent, ...]


def committed_behavior_batches(
    events: Sequence[TypedTraceEvent],
) -> tuple[tuple[CommittedBehaviorBatch, ...], str | None]:
    """Validate structural evidence, excluding an interrupted final batch.

    Rule-engine equality and detector metadata/schema checks belong to replay
    and the runtime respectively. This reader never invents missing transitions
    or counts partially acknowledged batches as behavioral effects.
    """
    batches: list[CommittedBehaviorBatch] = []
    boundary: BehavioralObservationRecordedEvent | None = None
    signals: list[BehavioralSignalRecordedEvent] = []
    transitions: list[BehavioralStateChangedEvent] = []
    prior_ids: set[str] = set()
    messages: dict[str, Message] = {}
    observation_ids: set[UUID4] = set()
    keys: set[tuple[str, str]] = set()
    interrupted = False
    boundary_event_ids: set[str] = set()
    boundary_message_ids: set[str] = set()

    def finish() -> None:
        nonlocal boundary, signals, transitions
        if boundary is not None and (
            len(signals) == boundary.data.signal_count
            and len(transitions) == boundary.data.signal_count
        ):
            batches.append(CommittedBehaviorBatch(boundary, tuple(signals), tuple(transitions)))
            boundary, signals, transitions = None, [], []

    for event in events:
        if event.id in prior_ids:
            raise ValueError("trace contains duplicate event identity")
        if boundary is not None and not isinstance(
            event,
            (
                BehavioralObservationRecordedEvent,
                BehavioralSignalRecordedEvent,
                BehavioralStateChangedEvent,
            ),
        ):
            interrupted = True
        if isinstance(event, MessageObservedEvent):
            message = event.data.message
            if message.id in messages:
                raise ValueError("trace contains duplicate message identity")
            messages[message.id] = message
        elif isinstance(event, BehavioralObservationRecordedEvent):
            if boundary is not None or interrupted:
                raise ValueError("behavioral observation follows an incomplete batch")
            data = event.data
            if data.observation_id in observation_ids:
                raise ValueError("duplicate behavioral observation identity")
            if any(ref not in prior_ids for ref in data.evidence_event_ids):
                raise ValueError("observation references uncommitted evidence")
            if any(ref not in messages for ref in data.message_ids):
                raise ValueError("observation references uncommitted messages")
            if any(messages[ref].actor != "assistant" for ref in data.message_ids):
                raise ValueError("observation response messages must belong to assistant")
            sequences = [messages[ref].sequence for ref in data.message_ids]
            if any(left >= right for left, right in zip(sequences, sequences[1:])):
                raise ValueError("observation response messages are out of order")
            observation_ids.add(data.observation_id)
            boundary = event
            boundary_event_ids = set(prior_ids)
            boundary_message_ids = set(messages)
            finish()
        elif isinstance(event, BehavioralSignalRecordedEvent):
            if boundary is None or interrupted:
                raise ValueError("behavioral signal has no active observation")
            signal = event.data.signal
            if transitions or len(signals) >= boundary.data.signal_count:
                raise ValueError("behavioral signals must precede all transitions")
            if signal.observation_id != boundary.data.observation_id:
                raise ValueError("signal observation identity is inconsistent")
            if signal.detector_id not in boundary.data.detector_ids:
                raise ValueError("signal detector is absent from observation")
            if (
                sum(recorded.data.signal.detector_id == signal.detector_id for recorded in signals)
                >= 32
            ):
                raise ValueError("detector signal output exceeds batch bound")
            order = {name: index for index, name in enumerate(boundary.data.detector_ids)}
            if signals and order[signal.detector_id] < order[signals[-1].data.signal.detector_id]:
                raise ValueError("signal detector order is inconsistent")
            key = (signal.detector_id, signal.deduplication_key)
            if key in keys:
                raise ValueError("duplicate committed behavioral signal key")
            if any(ref not in boundary_event_ids for ref in signal.source_event_ids):
                raise ValueError("signal references evidence outside pre-boundary committed trace")
            if any(ref not in boundary_message_ids for ref in signal.source_message_ids):
                raise ValueError("signal references messages outside pre-boundary committed trace")
            keys.add(key)
            signals.append(event)
        elif isinstance(event, BehavioralStateChangedEvent):
            if boundary is None or interrupted or len(signals) != boundary.data.signal_count:
                raise ValueError("behavioral transition precedes complete signal evidence")
            if len(transitions) >= len(signals):
                raise ValueError("behavioral batch has excess transitions")
            data = event.data
            signal_event = signals[len(transitions)]
            if (
                data.observation_id != boundary.data.observation_id
                or data.signal_event_id != signal_event.id
                or data.signal != signal_event.data.signal.signal
            ):
                raise ValueError("transition signal identity or order is inconsistent")
            transitions.append(event)
            finish()
        prior_ids.add(event.id)
    reason = "final behavioral batch is incomplete" if boundary is not None else None
    return tuple(batches), reason

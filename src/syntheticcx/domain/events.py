"""Typed trace envelopes; journal ordering is enforced by the future writer."""

from typing import Annotated, Literal

from pydantic import Field

from syntheticcx.config.models import Criterion, JsonObject
from syntheticcx.domain.models import (
    ArtifactRecord,
    DomainModel,
    ErrorDetail,
    EvaluationResult,
    Identifier,
    Lifecycle,
    Message,
    ModelInteraction,
    Outcome,
    PersonaFeedback,
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

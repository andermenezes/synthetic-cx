"""Owned run aggregates and serialized artifact shapes.

Cross-record evidence checks and contiguous sequencing belong to the journal.
"""

from pydantic import Field, RootModel, model_validator

from syntheticcx.config.models import Criterion, JsonObject
from syntheticcx.domain.events import TypedTraceEvent
from syntheticcx.domain.models import (
    ArtifactRecord,
    BehavioralDetectorMetadata,
    DomainModel,
    ErrorDetail,
    EvaluationResult,
    Identifier,
    Lifecycle,
    Message,
    Outcome,
    RunMetadata,
    Scenario,
    UtcDatetime,
)


class EvaluationContext(DomainModel):
    criterion: Criterion
    scenario: Scenario
    sanitized_trace: list[TypedTraceEvent]


class MetadataArtifact(DomainModel):
    id: Identifier
    scenario_id: Identifier
    persona_id: Identifier
    started_at: UtcDatetime
    ended_at: UtcDatetime | None = None
    lifecycle_status: Lifecycle
    outcome: Outcome | None = None
    termination_reason: str | None = None
    metadata: RunMetadata
    errors: list[ErrorDetail] = Field(default_factory=list)
    artifacts: list[ArtifactRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_lifecycle(self) -> "MetadataArtifact":
        if self.ended_at is not None and self.ended_at < self.started_at:
            raise ValueError("run end precedes its start")
        if self.lifecycle_status == "finalized":
            if self.outcome is None or self.ended_at is None or self.termination_reason is None:
                raise ValueError("finalized runs require outcome, end time, and termination reason")
        elif self.outcome is not None:
            raise ValueError("only finalized runs have a final outcome")
        return self


class Run(MetadataArtifact):
    messages: list[Message] = Field(default_factory=list)
    events: list[TypedTraceEvent] = Field(default_factory=list)
    evaluations: list[EvaluationResult] = Field(default_factory=list)


class ConversationArtifact(RootModel[list[Message]]):  # type: ignore[misc]
    """conversation.json is an ordered JSON array of messages."""


class EvaluationArtifact(DomainModel):
    criteria: list[Criterion]
    results: list[EvaluationResult]


class ConfigArtifact(DomainModel):
    """Redacted evidence snapshot; redacted values are not replayable configuration."""

    source_content_hash: Identifier
    channel_guide_source_hash: Identifier | None = None
    effective_config: JsonObject
    behavior_detectors: tuple[BehavioralDetectorMetadata, ...] = Field(default=(), max_length=8)

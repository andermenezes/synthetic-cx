"""Models shared by orchestration, adapters, and persistence."""

from __future__ import annotations

from datetime import UTC
from typing import Annotated, Literal

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from syntheticcx.config.models import (
    JsonObject,
    PersonaConfig,
    PersonaStateConfig,
    ProviderCapabilities,
    ScenarioConfig,
)
from syntheticcx.errors import ErrorCategory

Identifier = Annotated[str, Field(min_length=1)]
UtcDatetime = Annotated[AwareDatetime, AfterValidator(lambda value: value.astimezone(UTC))]
Outcome = Literal["passed", "failed", "error", "cancelled"]
Lifecycle = Literal["in_progress", "finalized", "incomplete"]


class PersonaState(PersonaStateConfig):
    """Owned state copied from configuration for each run."""


class PersonaDefinition(PersonaConfig):
    """Owned persona definition with behavior and communication contracts."""

    def fresh_state(self) -> PersonaState:
        return PersonaState.model_validate(self.initial_state.model_dump())  # type: ignore[no-any-return]


class Scenario(ScenarioConfig):
    """Owned scenario, including evaluator-only criteria."""


class DomainModel(BaseModel):  # type: ignore[misc]
    """Base model for domain data."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class CustomerScenarioContext(DomainModel):
    scenario_id: str
    goal: str
    starting_context: JsonObject
    knowledge: JsonObject

    @classmethod
    def from_scenario(cls, scenario: ScenarioConfig) -> CustomerScenarioContext:
        return cls(
            scenario_id=scenario.id,
            goal=scenario.goal,
            starting_context=scenario.model_copy(deep=True).starting_context,
            knowledge=scenario.model_copy(deep=True).knowledge,
        )


class Message(DomainModel):
    id: Identifier
    sequence: int = Field(ge=0)
    actor: Literal["customer", "assistant", "system"]
    content: str
    observed_at: UtcDatetime
    metadata: JsonObject = Field(default_factory=dict)


class SendReceipt(DomainModel):
    message_id: Identifier
    submitted_at: UtcDatetime
    evidence: JsonObject = Field(default_factory=dict)


class CompletedResponse(DomainModel):
    messages: list[Message] = Field(min_length=1)
    started_at: UtcDatetime
    completed_at: UtcDatetime
    completion_strategy: Literal[
        "quiet_interval", "streaming_indicator_disappears", "completion_locator"
    ]
    streaming_metadata: dict[str, JsonObject] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_response(self) -> CompletedResponse:
        if any(message.actor != "assistant" for message in self.messages):
            raise ValueError("completed responses contain assistant messages only")
        ids = [message.id for message in self.messages]
        sequences = [message.sequence for message in self.messages]
        if len(set(ids)) != len(ids) or any(a >= b for a, b in zip(sequences, sequences[1:])):
            raise ValueError("response messages must be distinct and ordered")
        if self.completed_at < self.started_at:
            raise ValueError("response completion precedes its start")
        if not set(self.streaming_metadata).issubset(ids):
            raise ValueError("streaming metadata must reference response messages")
        return self


class ConversationObservation(DomainModel):
    kind: Literal["conversation_started", "assistant_response"]
    messages: list[Message] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_observation(self) -> ConversationObservation:
        if self.kind == "conversation_started" and self.messages:
            raise ValueError("no-greeting observation cannot contain messages")
        if self.kind == "assistant_response" and (
            not self.messages or any(message.actor != "assistant" for message in self.messages)
        ):
            raise ValueError("assistant observation requires assistant messages")
        return self


class PersonaStateTransition(DomainModel):
    trigger_ids: list[Identifier]
    before: PersonaState
    after: PersonaState
    applied_rule_ids: list[Identifier]


class PersonaAction(DomainModel):
    message: str | None = None
    state: PersonaState
    signal: Literal["continue", "goal_believed_complete", "abandon"]
    transitions: list[PersonaStateTransition] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_action(self) -> PersonaAction:
        if self.message is not None and not self.message.strip():
            raise ValueError("customer message cannot be blank")
        if self.signal != "continue" and self.message is not None:
            raise ValueError("terminal actions cannot send another message")
        if (self.signal == "abandon") != self.state.abandoned:
            raise ValueError("abandonment signal must agree with state")
        return self


class EvaluationResult(DomainModel):
    evaluator_id: str
    criterion_id: str
    status: Literal["passed", "failed", "unavailable", "error"]
    score: Annotated[float, Field(ge=0, le=1)] | None = None
    required: bool
    explanation: str
    evidence_event_ids: list[str] = Field(default_factory=list)
    evidence_message_ids: list[str] = Field(default_factory=list)
    severity: str | None = None
    confidence: Annotated[float, Field(ge=0, le=1)] | None = None
    metadata: JsonObject = Field(default_factory=dict)


class TokenUsage(DomainModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    unavailable_reason: str | None = None

    @model_validator(mode="after")
    def explain_availability(self) -> TokenUsage:
        missing = self.input_tokens is None or self.output_tokens is None
        if missing != (self.unavailable_reason is not None):
            raise ValueError("missing usage requires an explicit unavailable reason")
        return self


class ModelIdentity(DomainModel):
    provider: Identifier
    model: Identifier
    model_version: str | None = None
    requested_seed: int | None = None
    sampling_parameters: JsonObject = Field(default_factory=dict)


class LLMMessage(DomainModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMRequest(DomainModel):
    id: Identifier
    messages: list[LLMMessage] = Field(min_length=1)
    response_schema: JsonObject | None = None
    sampling_parameters: JsonObject = Field(default_factory=dict)


class ModelInteraction(DomainModel):
    id: Identifier
    identity: ModelIdentity
    started_at: UtcDatetime
    ended_at: UtcDatetime
    usage: TokenUsage
    input_event_ids: list[Identifier] = Field(default_factory=list)
    output_event_ids: list[Identifier] = Field(default_factory=list)
    capture_omissions: list[str] = Field(default_factory=list)
    captured_prompt: str | None = None


class LLMResponse(DomainModel):
    content: str
    interaction: ModelInteraction


class ErrorDetail(DomainModel):
    category: ErrorCategory
    operation: Identifier
    message: str
    retryable: bool = False
    causal_event_id: Identifier | None = None


class RunOwner(DomainModel):
    owner_uuid: Identifier
    hostname: Identifier
    pid: int = Field(gt=0)
    process_start_evidence: Identifier


class ArtifactRecord(DomainModel):
    path: str | None = None
    media_type: Identifier
    content_hash: str | None = None
    status: Literal["captured", "omitted", "failed"]
    reason: str | None = None


class RunMetadata(DomainModel):
    configuration_version: str
    framework_version: str
    dependency_lock_hash: str | None = None
    provider_model_version: str | None = None
    capture_omissions: list[str] = Field(default_factory=list)
    persona_model: ModelIdentity
    evaluation_model: ModelIdentity
    verified_capabilities: dict[str, ProviderCapabilities] = Field(default_factory=dict)
    unavailable_capabilities: dict[str, list[str]] = Field(default_factory=dict)
    usage: TokenUsage
    runtime_platform: Identifier
    owner: RunOwner


class Report(DomainModel):
    run_id: str
    outcome: Literal["passed", "failed", "error", "cancelled"] | None = None
    findings: list[EvaluationResult] = Field(default_factory=list)
    termination_reason: str | None = None
    goal_result: EvaluationResult | None = None
    turn_count: int = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    repetition_count: int = Field(ge=0)
    errors: list[ErrorDetail] = Field(default_factory=list)
    artifact_location: str
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    omissions: list[str] = Field(default_factory=list)

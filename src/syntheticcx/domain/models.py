"""Models shared by orchestration, adapters, and persistence."""

from __future__ import annotations

from datetime import UTC
from typing import Annotated, Literal

from pydantic import (
    UUID4,
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from syntheticcx.config.models import (
    BehaviorDetectorVersion,
    BehaviorIdentifier,
    JsonObject,
    ObservationTopic,
    PersonaConfig,
    PersonaDescription,
    PersonaStateConfig,
    ProviderCapabilities,
    ScenarioConfig,
)
from syntheticcx.errors import ChannelFailureReason, ErrorCategory, ModelFailureReason
from syntheticcx.language import Language

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
    action_authorizations: list[JsonObject] = Field(default_factory=list)

    @classmethod
    def from_scenario(cls, scenario: ScenarioConfig) -> CustomerScenarioContext:
        return cls(
            scenario_id=scenario.id,
            goal=scenario.goal,
            starting_context=scenario.model_copy(deep=True).starting_context,
            knowledge=scenario.model_copy(deep=True).knowledge,
            action_authorizations=[
                authorization.model_dump(mode="json")
                for authorization in scenario.action_authorizations
            ],
        )


class ReplyOption(DomainModel):
    """An adapter-owned opaque choice; no executable browser target crosses this seam."""

    id: Identifier
    label: str = Field(min_length=1, pattern=r"\S")


class Message(DomainModel):
    id: Identifier
    sequence: int = Field(ge=0)
    actor: Literal["customer", "assistant", "system"]
    content: str
    observed_at: UtcDatetime
    metadata: JsonObject = Field(default_factory=dict)
    options: list[ReplyOption] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_options(self) -> Message:
        if self.options and self.actor != "assistant":
            raise ValueError("reply options belong to assistant messages only")
        ids = [option.id for option in self.options]
        if len(ids) != len(set(ids)):
            raise ValueError("reply option IDs must be unique")
        return self


class SendReceipt(DomainModel):
    message_id: Identifier
    submitted_at: UtcDatetime
    evidence: JsonObject = Field(default_factory=dict)


class SelectionReceipt(SendReceipt):
    selected_option_id: Identifier
    selected_label: str = Field(min_length=1, pattern=r"\S")
    source_message_id: Identifier


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


class ConversationCapabilities(DomainModel):
    """Current customer controls, independent of historical assistant evidence."""

    text_available: bool = True
    options: list[ReplyOption] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_options(self) -> ConversationCapabilities:
        ids = [option.id for option in self.options]
        if len(ids) != len(set(ids)):
            raise ValueError("current option IDs must be unique")
        return self


class ConversationObservation(DomainModel):
    kind: Literal["conversation_started", "assistant_response"]
    messages: list[Message] = Field(default_factory=list)
    capabilities: ConversationCapabilities | None = None

    @property
    def text_available(self) -> bool:
        return self.capabilities.text_available if self.capabilities is not None else True

    @property
    def current_options(self) -> list[ReplyOption]:
        if self.capabilities is not None:
            return self.capabilities.options
        return self.messages[-1].options if self.messages else []

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


class BehavioralStateTransition(DomainModel):
    """Acknowledged rule-engine effect, never an evaluator finding."""

    observation_id: UUID4
    signal_event_id: Identifier
    signal: BehaviorIdentifier
    occurrence: int = Field(gt=0, strict=True)
    before: PersonaState
    after: PersonaState
    applied_rule_ids: list[BehaviorIdentifier] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def validate_transition(self) -> BehavioralStateTransition:
        if len(self.applied_rule_ids) != len(set(self.applied_rule_ids)):
            raise ValueError("applied rule IDs must be unique")
        if self.before.turn_number != self.after.turn_number:
            raise ValueError("behavioral transitions cannot change turn number")
        if self.before.abandoned and (self.before != self.after or self.applied_rule_ids):
            raise ValueError("abandoned behavioral state cannot change or fire rules")
        return self


class BehavioralTransitionSummary(BehavioralStateTransition):
    """Payload-free behavioral report citation with exact detector identity."""

    detector_id: BehaviorIdentifier
    detector_type: BehaviorIdentifier
    detector_version: BehaviorDetectorVersion
    source_event_ids: list[Identifier] = Field(min_length=1)
    source_message_ids: list[Identifier] = Field(default_factory=list)


class PersonaAction(DomainModel):
    message: str | None = None
    selected_option_id: Identifier | None = None
    state: PersonaState
    signal: Literal["continue", "goal_believed_complete", "abandon"]
    transitions: list[PersonaStateTransition] = Field(default_factory=list)
    discarded_action_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_action(self) -> PersonaAction:
        if self.message is not None and not self.message.strip():
            raise ValueError("customer message cannot be blank")
        if self.message is not None and self.selected_option_id is not None:
            raise ValueError("text and option selection are mutually exclusive")
        if self.signal != "continue" and (
            self.message is not None or self.selected_option_id is not None
        ):
            raise ValueError("terminal actions cannot send another message or select an option")
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
    reason_code: ChannelFailureReason | ModelFailureReason | None = None


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


class BehavioralDetectorMetadata(DomainModel):
    """Exact configured implementation identity, without executable factories."""

    detector_id: BehaviorIdentifier
    detector_type: BehaviorIdentifier
    detector_version: BehaviorDetectorVersion
    mode: Literal["deterministic", "model_backed"]
    output_signals: tuple[BehaviorIdentifier, ...] = Field(min_length=1, max_length=32)
    signal_schemas: dict[BehaviorIdentifier, JsonObject] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def validate_output_signals(self) -> BehavioralDetectorMetadata:
        if len(self.output_signals) != len(set(self.output_signals)):
            raise ValueError("advertised signal names must be unique")
        if set(self.signal_schemas) != set(self.output_signals):
            raise ValueError("payload schema names must match advertised signal names")
        return self


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
    behavior_detectors: tuple[BehavioralDetectorMetadata, ...] = Field(default=(), max_length=8)


class TopicImpression(DomainModel):
    topic_id: Identifier
    status: Literal["observed", "not_observed"]
    impression: PersonaDescription
    evidence_message_ids: list[Identifier] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def validate_evidence(self) -> TopicImpression:
        if (self.status == "observed") != bool(self.evidence_message_ids):
            raise ValueError("observed impressions require evidence; not_observed forbids it")
        if len(self.evidence_message_ids) != len(set(self.evidence_message_ids)):
            raise ValueError("duplicate feedback evidence")
        return self


class PersonaFeedbackResponse(DomainModel):
    summary: str = Field(min_length=1, max_length=4000, pattern=r"\S")
    topics: list[TopicImpression] = Field(min_length=1, max_length=10)


class PersonaReflectionRequest(DomainModel):
    topics: list[ObservationTopic] = Field(min_length=1, max_length=10)
    language: Language
    final_state: PersonaState
    messages: list[Message] = Field(max_length=100)
    partial_experience: bool
    experience_truncated: bool = False


class PersonaFeedback(DomainModel):
    status: Literal["available", "unavailable"]
    language: Language
    topics: list[ObservationTopic] = Field(min_length=1, max_length=10)
    partial_experience: bool
    experience_truncated: bool = False
    response: PersonaFeedbackResponse | None = None
    reason: (
        Literal[
            "cancelled",
            "no_experience",
            "unsupported",
            "persona_unavailable",
            "reflection_failed",
            "reflection_timeout",
            "invalid_feedback",
        ]
        | None
    ) = None

    @model_validator(mode="after")
    def validate_status(self) -> PersonaFeedback:
        if self.status == "available":
            if self.response is None or self.reason is not None:
                raise ValueError("available feedback requires a response and no reason")
            requested = [topic.id for topic in self.topics]
            actual = [topic.topic_id for topic in self.response.topics]
            if len(actual) != len(set(actual)) or set(actual) != set(requested):
                raise ValueError("feedback must cover each requested topic once")
        elif self.response is not None or self.reason is None:
            raise ValueError("unavailable feedback requires a reason and no response")
        return self


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
    persona_feedback: PersonaFeedback | None = None
    conversation_duration_seconds: float | None = Field(default=None, ge=0)
    behavior_observation_count: int = Field(default=0, ge=0, strict=True)
    behavior_signal_count: int = Field(default=0, ge=0, strict=True)
    behavior_transitions: list[BehavioralTransitionSummary] = Field(default_factory=list)
    behavior_completeness: Literal["unavailable", "complete", "prefix"] = "unavailable"
    behavior_prefix_reason: str | None = None

    @model_validator(mode="after")
    def validate_behavior_evidence(self) -> Report:
        if (self.behavior_completeness == "prefix") != (self.behavior_prefix_reason is not None):
            raise ValueError("only behavioral prefixes require a reason")
        if self.behavior_signal_count != len(self.behavior_transitions):
            raise ValueError("behavior signal count requires acknowledged transitions")
        if self.behavior_completeness == "unavailable" and (
            self.behavior_observation_count or self.behavior_signal_count
        ):
            raise ValueError("unavailable behavioral evidence cannot claim committed batches")
        return self

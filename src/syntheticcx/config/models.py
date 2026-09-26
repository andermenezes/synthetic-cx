"""Version-one YAML models for SyntheticCX."""

from __future__ import annotations

import ipaddress
import re
import unicodedata
from typing import Annotated, Any, Literal, TypeAlias
from urllib.parse import parse_qs, urlsplit

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SecretStr,
    field_validator,
    model_validator,
)

JsonObject: TypeAlias = dict[str, JsonValue]
Normalized = Annotated[float, Field(ge=0, le=1)]
PositiveSeconds = Annotated[float, Field(gt=0)]
EnvironmentVariable = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]


class StrictModel(BaseModel):  # type: ignore[misc]
    """Base model for configuration sections."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class CommunicationConfig(StrictModel):
    tone: str = Field(min_length=1)
    style: str = Field(min_length=1)
    typo_probability: Normalized = 0
    spelling_error_probability: Normalized = 0
    clarification_probability: Normalized = 0


class PersonaStateConfig(StrictModel):
    frustration: Normalized = 0
    willingness_to_continue: Normalized = 1
    abandoned: bool = False
    turn_number: int = Field(default=0, ge=0)


class BehaviorRule(StrictModel):
    id: str = Field(min_length=1)
    priority: int
    event: Literal["repeated_information"]
    frustration_delta: float = Field(default=0, ge=-1, le=1)
    willingness_to_continue_delta: float = Field(default=0, ge=-1, le=1)
    abandonment_threshold: Normalized | None = None


class Criterion(StrictModel):
    id: str = Field(min_length=1)
    evaluator_type: Literal[
        "turn_count", "journey_timeout", "repeated_information", "semantic_goal_completion"
    ]
    result_kind: Literal["scored", "boolean"]
    required: bool = True
    threshold: Normalized | None = None
    evidence_policy: Literal["required", "optional", "none"]
    parameters: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_threshold(self) -> Criterion:
        if self.result_kind == "scored" and self.threshold is None:
            raise ValueError("scored criteria require threshold")
        if self.result_kind == "boolean" and self.threshold is not None:
            raise ValueError("boolean criteria forbid threshold")
        return self


class ScenarioInformation(StrictModel):
    """Information identifiable through configured phrases in a conversation."""

    id: str = Field(min_length=1)
    aliases: list[str] = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def require_meaningful_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("information ID must contain non-whitespace characters")
        return value

    @field_validator("aliases")
    @classmethod
    def require_meaningful_aliases(cls, value: list[str]) -> list[str]:
        if any(
            not any(
                not char.isspace() and not unicodedata.category(char).startswith("P")
                for char in alias
            )
            for alias in value
        ):
            raise ValueError("aliases must contain a word, not only punctuation or whitespace")
        return value


class ScenarioConfig(StrictModel):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    starting_context: JsonObject = Field(default_factory=dict)
    knowledge: JsonObject = Field(default_factory=dict)
    information: list[ScenarioInformation] = Field(default_factory=list)
    success_criteria: list[Criterion] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_criteria(self) -> ScenarioConfig:
        _require_unique_ids(self.success_criteria, "success_criteria")
        _require_unique_ids(self.information, "information")
        return self


class PersonaConfig(StrictModel):
    id: str = Field(min_length=1)
    name: str | None = None
    traits: dict[str, Normalized] = Field(default_factory=dict)
    communication: CommunicationConfig
    initial_state: PersonaStateConfig
    behavioral_rules: list[BehaviorRule] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_unique_rules(self) -> PersonaConfig:
        _require_unique_ids(self.behavioral_rules, "behavioral_rules")
        return self


class LocatorConfig(StrictModel):
    strategy: Literal["role", "label", "placeholder", "text", "test_id", "css"]
    value: str = Field(min_length=1)
    name: str | None = None

    @model_validator(mode="after")
    def forbid_xpath_fallback(self) -> LocatorConfig:
        if self.strategy == "css" and self.value.lstrip().startswith(("//", "xpath=")):
            raise ValueError("CSS locator fallback does not support XPath")
        return self


class MessageIdentityConfig(StrictModel):
    strategy: Literal["attribute", "actor_sequence_content_hash"]
    attribute_name: str | None = None

    @model_validator(mode="after")
    def validate_strategy(self) -> MessageIdentityConfig:
        if self.strategy == "attribute" and not self.attribute_name:
            raise ValueError("attribute strategy requires attribute_name")
        if self.strategy != "attribute" and self.attribute_name is not None:
            raise ValueError("actor_sequence_content_hash forbids attribute_name")
        return self


class SendStrategyConfig(StrictModel):
    type: Literal["press_enter", "click"]
    locator: LocatorConfig | None = None

    @model_validator(mode="after")
    def validate_strategy(self) -> SendStrategyConfig:
        if self.type == "click" and self.locator is None:
            raise ValueError("click send strategy requires locator")
        if self.type == "press_enter" and self.locator is not None:
            raise ValueError("press_enter send strategy forbids locator")
        return self


class ResponseCompletionConfig(StrictModel):
    strategy: Literal["quiet_interval", "streaming_indicator_disappears", "completion_locator"]
    quiet_interval_seconds: PositiveSeconds | None = None
    locator: LocatorConfig | None = None

    @model_validator(mode="after")
    def validate_strategy(self) -> ResponseCompletionConfig:
        if self.strategy == "quiet_interval":
            if self.quiet_interval_seconds is None or self.locator is not None:
                raise ValueError("quiet_interval requires quiet_interval_seconds only")
        elif self.locator is None or self.quiet_interval_seconds is not None:
            raise ValueError("locator completion strategies require locator only")
        return self


class ChannelConfig(StrictModel):
    type: Literal["webchat_playwright"]
    url: AnyHttpUrl
    open_locator: LocatorConfig | None = None
    frame_locator: LocatorConfig | None = None
    input_locator: LocatorConfig
    customer_message_locator: LocatorConfig
    assistant_message_locator: LocatorConfig
    message_identity: MessageIdentityConfig
    send_strategy: SendStrategyConfig
    response_completion: ResponseCompletionConfig

    @model_validator(mode="after")
    def restrict_content_hash_fallback(self) -> ChannelConfig:
        if self.message_identity.strategy != "actor_sequence_content_hash":
            return self
        target = urlsplit(str(self.url))
        host = (target.hostname or "").strip("[]")
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host == "localhost"
        variants = parse_qs(target.query).get("variant", [])
        if (
            not loopback
            or target.path not in {"", "/"}
            or len(variants) != 1
            or variants[0]
            not in {
                "success",
                "repeated_question",
                "streamed_response",
                "missing_control",
                "silent_response",
            }
        ):
            raise ValueError("content-hash identity fallback requires a controlled fixture URL")
        return self


class ExecutionLimits(StrictModel):
    max_turns: int = Field(gt=0)
    journey_timeout_seconds: PositiveSeconds
    greeting_grace_seconds: float = Field(ge=0)
    browser_action_timeout_seconds: PositiveSeconds
    echo_observation_seconds: PositiveSeconds = 0.25
    response_timeout_seconds: PositiveSeconds
    model_timeout_seconds: PositiveSeconds
    evaluator_timeout_seconds: PositiveSeconds
    persistence_timeout_seconds: PositiveSeconds
    close_timeout_seconds: PositiveSeconds


class ProviderCapabilities(StrictModel):
    chat_completions: bool
    structured_output: bool
    embeddings: bool
    token_usage: bool
    model_version: bool


class OpenAIModelConfig(StrictModel):
    provider: Literal["openai"]
    model: str = Field(min_length=1)
    credential_environment_variable: EnvironmentVariable = "OPENAI_API_KEY"
    temperature: Normalized = 0
    seed: int | None = None


class OpenAICompatibleModelConfig(StrictModel):
    provider: Literal["openai_compatible"]
    model: str = Field(min_length=1)
    base_url: AnyHttpUrl
    credential_environment_variable: EnvironmentVariable
    capabilities: ProviderCapabilities
    temperature: Normalized = 0
    seed: int | None = None

    @field_validator("base_url")
    @classmethod
    def require_https_except_loopback(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        if value.username is not None or value.password is not None:
            raise ValueError("endpoint credentials must come from the environment")
        host = (value.host or "").strip("[]")
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host == "localhost"
        if value.scheme != "https" and not loopback:
            raise ValueError("custom endpoints require HTTPS unless they are loopback")
        return value


ModelConfig = Annotated[
    OpenAIModelConfig | OpenAICompatibleModelConfig,
    Field(discriminator="provider"),
]


class TelemetryConfig(StrictModel):
    enabled: bool = True
    remote_export: bool = False
    otlp_endpoint: AnyHttpUrl | None = None

    @model_validator(mode="after")
    def validate_remote_export(self) -> TelemetryConfig:
        if self.remote_export != (self.otlp_endpoint is not None):
            raise ValueError("otlp_endpoint is required exactly when remote_export is enabled")
        return self


class StorageConfig(StrictModel):
    run_root: str = ".runs"
    flush_each_event: Literal[True] = True
    materialize_conversation: bool = True
    materialize_evaluation: bool = True
    materialize_report: bool = True
    store_screenshots: bool = False
    store_video: bool = False


class PrivacyConfig(StrictModel):
    redaction_keys: list[str] = Field(default_factory=list)
    synthetic_id_patterns: list[str] = Field(default_factory=list)
    capture_prompts: bool = False
    allow_screenshots: bool = False
    confirm_screenshot_target_safe: bool = False
    allow_video: bool = False

    @field_validator("synthetic_id_patterns")
    @classmethod
    def validate_patterns(cls, patterns: list[str]) -> list[str]:
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error:
                raise ValueError(
                    "synthetic ID pattern must be a valid regular expression"
                ) from None
        return patterns


class ProjectConfig(StrictModel):
    version: Literal["1"]
    scenario: ScenarioConfig
    persona: PersonaConfig
    channel: ChannelConfig
    execution: ExecutionLimits
    model: ModelConfig
    evaluation_model: ModelConfig | None = None
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)

    @model_validator(mode="after")
    def validate_capture_policy(self) -> ProjectConfig:
        if self.storage.store_screenshots and not self.privacy.allow_screenshots:
            raise ValueError("storage cannot enable screenshots forbidden by privacy")
        if self.storage.store_screenshots and not self.privacy.confirm_screenshot_target_safe:
            raise ValueError("screenshot storage requires explicit safe-target affirmation")
        if self.storage.store_video and not self.privacy.allow_video:
            raise ValueError("storage cannot enable video forbidden by privacy")
        return self

    @property
    def effective_evaluation_model(self) -> OpenAIModelConfig | OpenAICompatibleModelConfig:
        """Return the inherited persona model when no separate evaluator model is configured."""
        return self.evaluation_model or self.model


class ResolvedModelSettings(StrictModel):
    """Provider settings resolved from the environment."""

    provider: Literal["openai", "openai_compatible"]
    model: str
    credential: SecretStr


def _require_unique_ids(items: list[Any], field_name: str) -> None:
    identifiers = [item.id for item in items]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{field_name} IDs must be unique")

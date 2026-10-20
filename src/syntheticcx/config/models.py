"""YAML models for SyntheticCX configuration."""

from __future__ import annotations

import ipaddress
import re
import unicodedata
from typing import Annotated, Any, ClassVar, Literal, TypeAlias
from urllib.parse import parse_qs, unquote, urlsplit

from pydantic import (
    AnyHttpUrl,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SecretStr,
    ValidationInfo,
    field_validator,
    model_validator,
)

from syntheticcx.language import Language

JsonObject: TypeAlias = dict[str, JsonValue]
Normalized = Annotated[float, Field(ge=0, le=1)]
PositiveSeconds = Annotated[float, Field(gt=0)]
EnvironmentVariable = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
PersonaDescription = Annotated[str, Field(min_length=1, max_length=2000, pattern=r"\S")]
BehaviorIdentifier = Annotated[
    str, Field(min_length=1, max_length=80, pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
]
BehaviorDetectorVersion = Annotated[str, Field(min_length=1, max_length=80, pattern=r"\S")]


class StrictModel(BaseModel):  # type: ignore[misc]
    """Base model for configuration sections."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class CommunicationConfig(StrictModel):
    tone: str = Field(min_length=1)
    style: str = Field(min_length=1)
    language: Language = "en-US"
    writing_description: PersonaDescription | None = None
    message_length: Literal["short", "medium", "long"] | None = None
    grammar_error_probability: Normalized = 0
    typo_probability: Normalized = 0
    spelling_error_probability: Normalized = 0
    clarification_probability: Normalized = 0


class PersonaStateConfig(StrictModel):
    frustration: Normalized = 0
    willingness_to_continue: Normalized = 1
    abandoned: bool = False
    turn_number: int = Field(default=0, ge=0)


class BehaviorStateCondition(StrictModel):
    gte: Normalized | None = None
    lte: Normalized | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> BehaviorStateCondition:
        if self.gte is None and self.lte is None:
            raise ValueError("state comparison requires at least one bound")
        if self.gte is not None and self.lte is not None and self.gte > self.lte:
            raise ValueError("state comparison lower bound must not exceed upper bound")
        return self


class BehaviorStateConditions(StrictModel):
    frustration: BehaviorStateCondition | None = None
    willingness_to_continue: BehaviorStateCondition | None = None

    @model_validator(mode="after")
    def require_comparison(self) -> BehaviorStateConditions:
        if self.frustration is None and self.willingness_to_continue is None:
            raise ValueError("state conditions require at least one comparison")
        return self


class BehaviorCondition(StrictModel):
    signal: BehaviorIdentifier
    occurrence_at_least: int = Field(default=1, gt=0, strict=True)
    state: BehaviorStateConditions | None = None


class BehaviorEffects(StrictModel):
    frustration_delta: float = Field(default=0, ge=-1, le=1)
    willingness_to_continue_delta: float = Field(default=0, ge=-1, le=1)
    abandonment_threshold: Normalized | None = None
    abandon: Literal[True] | None = None

    @field_validator("abandon", mode="before")
    @classmethod
    def require_explicit_abandonment(cls, value: Any) -> Any:
        if value is not None and value is not True:
            raise ValueError("abandon must be true or null")
        return value

    @model_validator(mode="after")
    def require_effect(self) -> BehaviorEffects:
        if (
            self.frustration_delta == 0
            and self.willingness_to_continue_delta == 0
            and self.abandonment_threshold is None
            and self.abandon is None
        ):
            raise ValueError("behavior effects require an effective configured effect")
        return self


class BehaviorRule(StrictModel):
    id: BehaviorIdentifier
    priority: int = Field(strict=True)
    description: PersonaDescription | None = None
    when: BehaviorCondition
    effects: BehaviorEffects
    fire: Literal["every_match", "once"] = "every_match"


class BehaviorDetectorConfig(StrictModel):
    id: BehaviorIdentifier
    type: BehaviorIdentifier
    parameters: JsonObject = Field(default_factory=dict)


def _default_behavior_detectors() -> list[BehaviorDetectorConfig]:
    return [
        BehaviorDetectorConfig(id="repeated-information", type="repeated_information"),
        BehaviorDetectorConfig(id="completed-turn", type="customer_turn_completed"),
    ]


class BehaviorConfig(StrictModel):
    detectors: list[BehaviorDetectorConfig] = Field(
        default_factory=_default_behavior_detectors, max_length=8
    )

    @model_validator(mode="after")
    def require_unique_detectors(self) -> BehaviorConfig:
        _require_unique_ids(self.detectors, "detectors")
        return self


class TurnCountParameters(StrictModel):
    max_turns: int = Field(gt=0, strict=True)


class JourneyTimeoutParameters(StrictModel):
    """The observed termination reason supplies the timeout decision."""


class RepeatedInformationParameters(StrictModel):
    max_repeats: int = Field(default=0, ge=0, strict=True)


class SemanticGoalParameters(StrictModel):
    expected_facts: list[Annotated[str, Field(pattern=r"\S")]] = Field(min_length=1)


_PARAMETER_MODELS = {
    "turn_count": TurnCountParameters,
    "journey_timeout": JourneyTimeoutParameters,
    "repeated_information": RepeatedInformationParameters,
    "semantic_goal_completion": SemanticGoalParameters,
}


def _criterion_parameter_schema(schema: dict[str, Any]) -> None:
    """Express the same tagged parameter constraints in generated JSON Schema."""
    conditions: list[dict[str, Any]] = [
        {
            "if": {"properties": {"evaluator_type": {"const": evaluator_type}}},
            "then": {
                "properties": {
                    "parameters": model.model_json_schema(),
                    **(
                        {"result_kind": {"const": "boolean"}}
                        if evaluator_type != "semantic_goal_completion"
                        else {}
                    ),
                },
                **(
                    {"required": ["parameters"]}
                    if evaluator_type in {"turn_count", "semantic_goal_completion"}
                    else {}
                ),
            },
        }
        for evaluator_type, model in _PARAMETER_MODELS.items()
    ] + [
        {
            "if": {"properties": {"result_kind": {"const": "scored"}}},
            "then": {
                "required": ["threshold"],
                "properties": {"threshold": {"type": "number"}},
            },
            "else": {"properties": {"threshold": {"type": "null"}}},
        }
    ]
    schema.setdefault("allOf", []).extend(conditions)


class Criterion(StrictModel):
    model_config = ConfigDict(json_schema_extra=_criterion_parameter_schema)

    id: str = Field(min_length=1)
    evaluator_type: Annotated[
        str, Field(min_length=1, max_length=80, pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
    ]
    result_kind: Literal["scored", "boolean"]
    required: bool = True
    threshold: Normalized | None = None
    evidence_policy: Literal["required", "optional", "none"]
    parameters: JsonObject = Field(default_factory=dict, validate_default=True)

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, value: JsonObject, info: ValidationInfo) -> JsonObject:
        evaluator_type = info.data.get("evaluator_type")
        if evaluator_type not in _PARAMETER_MODELS:
            # Extension parameters are validated against the explicit trusted
            # registration during preparation, before credentials or external work.
            # A preceding identity validation error also leaves it out of info.data.
            return value
        validated = _PARAMETER_MODELS[evaluator_type].model_validate(value)
        return validated.model_dump(mode="json")  # type: ignore[no-any-return]

    @model_validator(mode="after")
    def validate_threshold(self) -> Criterion:
        if (
            self.evaluator_type in _PARAMETER_MODELS
            and self.evaluator_type != "semantic_goal_completion"
            and self.result_kind != "boolean"
        ):
            raise ValueError("deterministic evaluator result_kind must be boolean")
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


class ObservationTopic(StrictModel):
    id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    description: PersonaDescription


class ActionAuthorization(StrictModel):
    """Scenario-scoped permission for consequential customer choices."""

    action: Literal[
        "payment", "purchase", "account_change", "consent", "authentication", "destructive"
    ]
    # Payment authorization is intentionally bounded and sandbox-only in this MVP.
    environment: Literal["sandbox"] | None = None
    max_amount: Annotated[float, Field(gt=0)] | None = None
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")

    @model_validator(mode="after")
    def validate_payment_bounds(self) -> ActionAuthorization:
        payment_fields = (self.environment, self.max_amount, self.currency)
        if self.action == "payment" and any(value is None for value in payment_fields):
            raise ValueError("payment authorization requires sandbox, max_amount, and currency")
        if self.action != "payment" and any(value is not None for value in payment_fields):
            raise ValueError("payment bounds apply only to payment authorization")
        return self


class ScenarioConfig(StrictModel):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    starting_context: JsonObject = Field(default_factory=dict)
    knowledge: JsonObject = Field(default_factory=dict)
    information: list[ScenarioInformation] = Field(default_factory=list)
    observation_topics: list[ObservationTopic] = Field(default_factory=list, max_length=10)
    action_authorizations: list[ActionAuthorization] = Field(default_factory=list, max_length=8)
    success_criteria: list[Criterion] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_criteria(self) -> ScenarioConfig:
        _require_unique_ids(self.success_criteria, "success_criteria")
        _require_unique_ids(self.information, "information")
        _require_unique_ids(self.observation_topics, "observation_topics")
        actions = [authorization.action for authorization in self.action_authorizations]
        if len(actions) != len(set(actions)):
            raise ValueError("scenario action authorizations must be unique")
        return self


class PersonaConfig(StrictModel):
    id: str = Field(min_length=1)
    name: str | None = None
    description: PersonaDescription | None = None
    behavior_description: PersonaDescription | None = None
    traits: dict[str, Normalized] = Field(default_factory=dict)
    communication: CommunicationConfig
    initial_state: PersonaStateConfig
    behavioral_rules: list[BehaviorRule] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def require_unique_rules(self) -> PersonaConfig:
        _require_unique_ids(self.behavioral_rules, "behavioral_rules")
        return self


class LocatorConfig(StrictModel):
    strategy: Literal["role", "label", "placeholder", "text", "test_id", "css"]
    value: str = Field(min_length=1)
    name: str | None = None
    exact: bool = True

    @model_validator(mode="after")
    def forbid_xpath_fallback(self) -> LocatorConfig:
        if self.strategy == "css" and self.value.lstrip().startswith(("//", "xpath=")):
            raise ValueError("CSS locator fallback does not support XPath")
        return self


class WebsiteNavigationStep(StrictModel):
    action: Literal["click", "hover", "follow_link", "scroll_down", "scroll_up", "back", "wait"]
    locator: LocatorConfig | None = None

    @model_validator(mode="after")
    def validate_locator(self) -> WebsiteNavigationStep:
        if (self.action in {"click", "hover", "follow_link"}) != (self.locator is not None):
            raise ValueError("website control actions require a locator; other actions forbid one")
        return self


class MessageIdentityConfig(StrictModel):
    strategy: Literal["attribute", "dom_node", "actor_sequence_content_hash"]
    attribute_name: str | None = None

    @model_validator(mode="after")
    def validate_strategy(self) -> MessageIdentityConfig:
        if self.strategy == "attribute" and not self.attribute_name:
            raise ValueError("attribute strategy requires attribute_name")
        if self.strategy != "attribute" and self.attribute_name is not None:
            raise ValueError("non-attribute identity strategies forbid attribute_name")
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


class ReplyOptionsConfig(StrictModel):
    policy: Literal["guarded_informational_menu"] = "guarded_informational_menu"
    scope_locator: LocatorConfig
    option_locator: LocatorConfig


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
    reply_options: ReplyOptionsConfig | None = None

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


class ChatOwnership(StrictModel):
    """Independently attested chat root; child locators are root-relative."""

    root_locator: LocatorConfig
    proof: Literal["chat_dialog", "chat_embed", "opener_linked", "chat_panel"]

    @field_validator("root_locator")
    @classmethod
    def require_scoped_root(cls, value: LocatorConfig) -> LocatorConfig:
        if value.strategy == "css" and value.value.strip().lower() in {
            "body",
            "html",
            "*",
            ":root",
        }:
            raise ValueError("ownership requires a pinned chat root, not a document root")
        return value


ContractId = Annotated[str, Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")]
OptionLabel = Annotated[str, Field(min_length=1, max_length=200, pattern=r"\S")]


def _normalized_label(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


class ChatState(StrictModel):
    id: ContractId
    kind: Literal["menu", "composer", "mixed", "terminal"]
    input_locator: LocatorConfig | None = None
    send_strategy: SendStrategyConfig | None = None
    option_locator: LocatorConfig | None = None
    option_labels: list[OptionLabel] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def validate_controls(self) -> ChatState:
        composer = self.input_locator is not None
        options = self.option_locator is not None
        if composer != (self.send_strategy is not None):
            raise ValueError("input_locator and send_strategy must be paired")
        if options != bool(self.option_labels):
            raise ValueError("option_locator and option_labels must be paired")
        if (composer, options) != {
            "menu": (False, True),
            "composer": (True, False),
            "mixed": (True, True),
            "terminal": (False, False),
        }[self.kind]:
            raise ValueError("state controls must match kind")
        labels = [_normalized_label(label) for label in self.option_labels]
        if len(labels) != len(set(labels)):
            raise ValueError("option_labels must be unique after normalization")
        return self


class ChatTransition(StrictModel):
    id: ContractId
    from_state: ContractId
    to_state: ContractId
    action_kind: Literal["probe", "select_option"]
    probe_id: Literal["first", "second", "third", "explicit"] | None = None
    option_label: OptionLabel | None = None

    @model_validator(mode="after")
    def validate_descriptor(self) -> ChatTransition:
        if self.action_kind == "probe":
            valid = self.probe_id is not None and self.option_label is None
        else:
            valid = self.option_label is not None and self.probe_id is None
        if not valid:
            raise ValueError("transition requires exactly its matching action descriptor")
        return self


class RecipeStep(StrictModel):
    transition_id: ContractId


class StatefulChannelConfig(StrictModel):
    type: Literal["webchat_playwright_stateful"]
    url: AnyHttpUrl
    open_locator: LocatorConfig | None = None
    navigation_steps: list[WebsiteNavigationStep] = Field(default_factory=list, max_length=128)
    frame_locator: LocatorConfig | None = None
    ownership: ChatOwnership
    assistant_message_locator: LocatorConfig
    customer_message_locator: LocatorConfig | None = None
    reader_evidence: Literal["marked", "observed_exchange"] = "marked"
    message_identity: MessageIdentityConfig
    response_completion: ResponseCompletionConfig
    initial_state_id: ContractId
    states: list[ChatState] = Field(min_length=1, max_length=4)
    transitions: list[ChatTransition] = Field(min_length=1, max_length=3)
    verification_recipe: list[RecipeStep] = Field(min_length=1, max_length=3)

    @field_validator("url")
    @classmethod
    def forbid_url_credentials(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        if value.username is not None or value.password is not None:
            raise ValueError("stateful channel URL must not contain credentials")
        return value

    @model_validator(mode="after")
    def validate_graph(self) -> StatefulChannelConfig:
        if self.reader_evidence == "observed_exchange" and self.customer_message_locator is None:
            raise ValueError("observed exchange readers require a separated customer locator")
        _require_unique_ids(self.states, "states")
        _require_unique_ids(self.transitions, "transitions")
        states = {state.id: state for state in self.states}
        transitions = {transition.id: transition for transition in self.transitions}
        if self.initial_state_id not in states:
            raise ValueError("initial_state_id must reference a configured state")
        for transition in self.transitions:
            if transition.from_state not in states or transition.to_state not in states:
                raise ValueError("transition endpoints must reference configured states")
            source = states[transition.from_state]
            if transition.action_kind == "probe" and source.input_locator is None:
                raise ValueError("probe transition requires a composer in its source state")
            if transition.action_kind == "select_option" and (
                transition.option_label not in source.option_labels
            ):
                raise ValueError("option transition must reference a canonical source option label")
        current = self.initial_state_id
        visited_states = {current}
        visited_transitions: set[str] = set()
        for step in self.verification_recipe:
            recipe_transition = transitions.get(step.transition_id)
            if recipe_transition is None:
                raise ValueError("recipe transition_id must reference a configured transition")
            if recipe_transition.from_state != current:
                raise ValueError(
                    "verification_recipe must form a contiguous path from initial_state_id"
                )
            current = recipe_transition.to_state
            visited_states.add(current)
            visited_transitions.add(recipe_transition.id)
        if visited_states != set(states) or visited_transitions != set(transitions):
            raise ValueError("verification_recipe must cover every configured state and transition")
        if self.reader_evidence == "observed_exchange" and (
            states[self.initial_state_id].kind != "composer"
            or transitions[self.verification_recipe[0].transition_id].action_kind != "probe"
        ):
            raise ValueError(
                "observed exchange readers require a first composer probe in the recipe"
            )
        # Preserve the legacy restriction on content-hash identity fallback.
        if self.message_identity.strategy == "actor_sequence_content_hash":
            ChannelConfig.model_validate(
                {
                    "type": "webchat_playwright",
                    "url": self.url,
                    "input_locator": {"strategy": "css", "value": "input"},
                    "customer_message_locator": self.customer_message_locator
                    or self.assistant_message_locator,
                    "assistant_message_locator": self.assistant_message_locator,
                    "message_identity": self.message_identity,
                    "send_strategy": {"type": "press_enter"},
                    "response_completion": self.response_completion,
                }
            )
        return self


class InterfaceChannelConfig(StrictModel):
    """Fixed chat boundary and readers; interaction controls are discovered per turn."""

    type: Literal["webchat_playwright_dynamic"]
    url: AnyHttpUrl
    open_locator: LocatorConfig | None = None
    navigation_steps: list[WebsiteNavigationStep] = Field(default_factory=list, max_length=128)
    frame_locator: LocatorConfig | None = None
    ownership: ChatOwnership
    assistant_message_locator: LocatorConfig
    customer_message_locator: LocatorConfig | None = None
    reader_evidence: Literal["marked", "observed_exchange"] = "marked"
    message_identity: MessageIdentityConfig
    response_completion: ResponseCompletionConfig
    bootstrap_input_locator: LocatorConfig | None = None
    bootstrap_send_strategy: SendStrategyConfig | None = None

    @field_validator("url")
    @classmethod
    def forbid_url_credentials(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        if value.username is not None or value.password is not None:
            raise ValueError("interface channel URL must not contain credentials")
        return value

    @model_validator(mode="after")
    def validate_boundary(self) -> InterfaceChannelConfig:
        if (self.bootstrap_input_locator is None) != (self.bootstrap_send_strategy is None):
            raise ValueError("bootstrap_input_locator and bootstrap_send_strategy must be paired")
        if self.reader_evidence == "observed_exchange" and self.customer_message_locator is None:
            raise ValueError("observed exchange readers require a separated customer locator")
        if self.message_identity.strategy == "actor_sequence_content_hash":
            ChannelConfig.model_validate(
                {
                    "type": "webchat_playwright",
                    "url": self.url,
                    "input_locator": {"strategy": "css", "value": "input"},
                    "customer_message_locator": self.customer_message_locator
                    or self.assistant_message_locator,
                    "assistant_message_locator": self.assistant_message_locator,
                    "message_identity": self.message_identity,
                    "send_strategy": {"type": "press_enter"},
                    "response_completion": self.response_completion,
                }
            )
        return self


class WhatsAppChannelConfig(StrictModel):
    """Website entry and recipient binding; account credentials remain operational."""

    type: Literal["whatsapp_web_playwright"]
    url: AnyHttpUrl
    navigation_steps: list[WebsiteNavigationStep] = Field(default_factory=list, max_length=128)
    recipient_environment_variable: EnvironmentVariable = "SYNTHETICCX_WHATSAPP_RECIPIENT"
    quiet_interval_seconds: float = Field(default=1.0, gt=0, le=10)

    @field_validator("url")
    @classmethod
    def forbid_url_credentials(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        if value.username is not None or value.password is not None:
            raise ValueError("WhatsApp website URL must not contain credentials")
        if value.query or value.fragment:
            raise ValueError("WhatsApp website URL must not contain query data or a fragment")
        path = value.path or ""
        while (decoded := unquote(path)) != path:
            path = decoded
        if re.search(r"\+?[0-9][0-9 ().-]{4,}[0-9]", path):
            raise ValueError("WhatsApp website URL must not contain a recipient identifier")
        return value


ResolvedChannelConfig = Annotated[
    InterfaceChannelConfig | WhatsAppChannelConfig, Field(discriminator="type")
]


class ExecutionLimits(StrictModel):
    assistant_update_limit: int = Field(default=64, ge=0, le=1024, strict=True)
    max_turns: int = Field(gt=0)
    journey_timeout_seconds: PositiveSeconds
    greeting_grace_seconds: float = Field(ge=0)
    browser_action_timeout_seconds: PositiveSeconds
    website_action_limit: int = Field(default=32, ge=1, le=128, strict=True)
    website_observation_limit: int = Field(default=64, ge=1, le=256, strict=True)
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


class MarkdownReportingConfig(StrictModel):
    enabled: bool = False


class ReportingConfig(StrictModel):
    language: Language | None = None
    markdown: MarkdownReportingConfig = Field(default_factory=MarkdownReportingConfig)


class ProjectSettings(StrictModel):
    """Fields shared by inline runs, referenced runs, and discovery drafts."""

    version: ClassVar[Literal["1"]] = "1"
    scenario: ScenarioConfig
    persona: PersonaConfig
    behavior: BehaviorConfig = Field(default_factory=BehaviorConfig)
    execution: ExecutionLimits
    model: ModelConfig
    evaluation_model: ModelConfig | None = None
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    @property
    def report_language(self) -> Language:
        return self.reporting.language or self.persona.communication.language

    @model_validator(mode="after")
    def validate_capture_policy(self) -> ProjectSettings:
        if self.storage.store_screenshots and not self.privacy.allow_screenshots:
            raise ValueError("storage cannot enable screenshots forbidden by privacy")
        if self.storage.store_screenshots and not self.privacy.confirm_screenshot_target_safe:
            raise ValueError("screenshot storage requires explicit safe-target affirmation")
        if self.storage.store_video and not self.privacy.allow_video:
            raise ValueError("storage cannot enable video forbidden by privacy")
        return self

    @model_validator(mode="after")
    def validate_criteria_limits(self) -> ProjectSettings:
        for criterion in self.scenario.success_criteria:
            if (
                criterion.evaluator_type == "turn_count"
                and criterion.parameters["max_turns"] > self.execution.max_turns
            ):
                raise ValueError(
                    "scenario.success_criteria.parameters.max_turns exceeds execution.max_turns"
                )
        return self

    @property
    def effective_evaluation_model(self) -> OpenAIModelConfig | OpenAICompatibleModelConfig:
        """Return the inherited persona model when no separate evaluator model is configured."""
        return self.evaluation_model or self.model


class ProjectConfig(ProjectSettings):
    """Fully resolved configuration consumed by the existing run pipeline."""

    channel: ResolvedChannelConfig


class ReferencedProjectConfig(ProjectSettings):
    """Run document with exactly one local channel-guide reference."""

    channel_guide: str = Field(min_length=1, pattern=r"\S")


class DiscoveryDraft(ProjectSettings):
    """Scenario and model settings validated without channel controls."""


class GuideValidation(StrictModel):
    status: Literal["verified"]
    verified_at: AwareDatetime
    target_origin: AnyHttpUrl
    observed_behaviors: list[Literal["open", "send", "receive", "completion", "select_option"]] = (
        Field(min_length=4)
    )

    @model_validator(mode="after")
    def validate_behaviors(self) -> GuideValidation:
        if not {"open", "send", "receive", "completion"}.issubset(self.observed_behaviors):
            raise ValueError("guide must verify open, send, receive, and completion")
        return self


class ChannelGuide(StrictModel):
    """Verified channel controls imported by a referenced run document."""

    channel: ChannelConfig
    validation: GuideValidation

    @model_validator(mode="after")
    def validate_origin(self) -> ChannelGuide:
        if (
            "select_option" in self.validation.observed_behaviors
            and self.channel.reply_options is None
        ):
            raise ValueError("verified option selection requires configured reply options")
        target = urlsplit(str(self.channel.url))
        origin = urlsplit(str(self.validation.target_origin))
        if (
            (target.scheme, target.hostname, target.port)
            != (
                origin.scheme,
                origin.hostname,
                origin.port,
            )
            or origin.path not in {"", "/"}
            or origin.query
            or origin.fragment
        ):
            raise ValueError("validation.target_origin must match channel.url origin")
        return self


class StatefulGuideValidation(StrictModel):
    status: Literal["verified"]
    verified_at: AwareDatetime
    target_origin: AnyHttpUrl
    observed_behaviors: list[Literal["open", "send", "select_option", "receive", "completion"]] = (
        Field(min_length=3, max_length=5)
    )
    verified_state_ids: list[ContractId] = Field(min_length=1, max_length=4)
    verified_transition_ids: list[ContractId] = Field(min_length=1, max_length=3)

    @field_validator("target_origin")
    @classmethod
    def forbid_origin_credentials(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        if value.username is not None or value.password is not None:
            raise ValueError("stateful target origin must not contain credentials")
        return value

    @model_validator(mode="after")
    def validate_evidence(self) -> StatefulGuideValidation:
        if not {"open", "receive", "completion"}.issubset(self.observed_behaviors):
            raise ValueError("guide must verify open, receive, and completion")
        for values in (
            self.observed_behaviors,
            self.verified_state_ids,
            self.verified_transition_ids,
        ):
            if len(values) != len(set(values)):
                raise ValueError("verification evidence must not contain duplicates")
        return self


class StatefulChannelGuide(StrictModel):
    channel: StatefulChannelConfig
    validation: StatefulGuideValidation

    @model_validator(mode="after")
    def validate_evidence(self) -> StatefulChannelGuide:
        if set(self.validation.verified_state_ids) != {state.id for state in self.channel.states}:
            raise ValueError("verified_state_ids must exactly cover configured states")
        if set(self.validation.verified_transition_ids) != {
            item.id for item in self.channel.transitions
        }:
            raise ValueError("verified_transition_ids must exactly cover configured transitions")
        actions = {
            "send" if item.action_kind == "probe" else "select_option"
            for item in self.channel.transitions
        }
        if set(self.validation.observed_behaviors) != {"open", "receive", "completion"} | actions:
            raise ValueError("observed_behaviors must exactly cover verified actions")
        target = urlsplit(str(self.channel.url))
        origin = urlsplit(str(self.validation.target_origin))
        if (
            (target.scheme, target.hostname, target.port)
            != (origin.scheme, origin.hostname, origin.port)
            or origin.path not in {"", "/"}
            or origin.query
            or origin.fragment
        ):
            raise ValueError("validation.target_origin must match channel.url origin")
        return self


class InterfaceGuideValidation(StrictModel):
    """Content-free evidence from a complete fresh replay, without its private graph."""

    status: Literal["verified"]
    verified_at: AwareDatetime
    target_origin: AnyHttpUrl
    observed_behaviors: list[Literal["open", "send", "select_option", "receive", "completion"]] = (
        Field(min_length=4, max_length=5)
    )

    @field_validator("target_origin")
    @classmethod
    def forbid_origin_credentials(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        if value.username is not None or value.password is not None:
            raise ValueError("interface target origin must not contain credentials")
        return value

    @model_validator(mode="after")
    def validate_evidence(self) -> InterfaceGuideValidation:
        behaviors = set(self.observed_behaviors)
        if not {"open", "receive", "completion"}.issubset(behaviors) or not behaviors & {
            "send",
            "select_option",
        }:
            raise ValueError("guide must verify open, an interaction, receive, and completion")
        if len(behaviors) != len(self.observed_behaviors):
            raise ValueError("verification evidence must not contain duplicates")
        return self


class InterfaceChannelGuide(StrictModel):
    """Public dynamic interface contract."""

    channel: InterfaceChannelConfig
    validation: InterfaceGuideValidation

    @model_validator(mode="after")
    def validate_origin(self) -> InterfaceChannelGuide:
        target = urlsplit(str(self.channel.url))
        origin = urlsplit(str(self.validation.target_origin))
        if (
            (target.scheme, target.hostname, target.port)
            != (origin.scheme, origin.hostname, origin.port)
            or origin.path not in {"", "/"}
            or origin.query
            or origin.fragment
        ):
            raise ValueError("validation.target_origin must match channel.url origin")
        if self.channel.reader_evidence == "observed_exchange" and (
            "send" not in self.validation.observed_behaviors
        ):
            raise ValueError("observed exchange readers require verified send evidence")
        return self


class WhatsAppGuideValidation(StrictModel):
    """Browser replay evidence; it never implies a reset business conversation."""

    status: Literal["verified"]
    verified_at: AwareDatetime
    target_origin: AnyHttpUrl
    observed_behaviors: list[Literal["open", "send", "receive", "completion"]]
    replay_verified: Literal[True]
    server_state_reset: Literal[False] = False

    @model_validator(mode="after")
    def validate_evidence(self) -> WhatsAppGuideValidation:
        required = {"open", "send", "receive", "completion"}
        if set(self.observed_behaviors) != required or len(self.observed_behaviors) != 4:
            raise ValueError("WhatsApp guide requires a complete exchange and browser replay")
        return self


class WhatsAppChannelGuide(StrictModel):
    """Verified website-to-WhatsApp handoff without account data or conversation history."""

    channel: WhatsAppChannelConfig
    validation: WhatsAppGuideValidation

    @model_validator(mode="after")
    def validate_boundary(self) -> WhatsAppChannelGuide:
        if not self.channel.navigation_steps:
            raise ValueError("verified WhatsApp guide requires a nonempty website navigation route")
        target = urlsplit(str(self.channel.url))
        origin = urlsplit(str(self.validation.target_origin))
        if (
            (target.scheme, target.hostname, target.port)
            != (origin.scheme, origin.hostname, origin.port)
            or origin.username is not None
            or origin.password is not None
            or origin.path not in {"", "/"}
            or origin.query
            or origin.fragment
        ):
            raise ValueError("validation.target_origin must match channel.url origin")
        return self


AnyChannelGuide = InterfaceChannelGuide | WhatsAppChannelGuide


class ResolvedModelSettings(StrictModel):
    """Provider settings resolved from the environment."""

    provider: Literal["openai", "openai_compatible"]
    model: str
    credential: SecretStr


def _require_unique_ids(items: list[Any], field_name: str) -> None:
    identifiers = [item.id for item in items]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{field_name} IDs must be unique")

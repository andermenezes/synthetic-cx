"""SDK-free detector observations, signals, and offline behavioral replay contracts.

Observation identities are random UUID4 boundary identities, never hashes of text.
"""

from __future__ import annotations

import json
from typing import Annotated, Literal, TypeAlias
from uuid import uuid4

from pydantic import UUID4, Field, field_validator, model_validator

from syntheticcx.config.models import (
    BehaviorDetectorVersion,
    BehaviorIdentifier,
    BehaviorRule,
    JsonObject,
)
from syntheticcx.domain.models import (
    DomainModel,
    Message,
    PersonaState,
)

EvidenceReference = Annotated[str, Field(min_length=1, max_length=200, pattern=r"^\S+$")]
DeduplicationKey = Annotated[str, Field(min_length=1, max_length=200, pattern=r"^\S+$")]
Sha256Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ReplayMode: TypeAlias = Literal["verification", "counterfactual"]
MAX_SIGNAL_PAYLOAD_BYTES = 8192
MAX_SIGNAL_PAYLOAD_DEPTH = 16


def _require_unique(values: tuple[str, ...], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")


class BehavioralRepetitionFact(DomainModel):
    """Detached committed evaluator-authoritative repetition evidence."""

    event_id: EvidenceReference
    information_id: EvidenceReference
    original_message_id: EvidenceReference
    request_message_id: EvidenceReference


class BehavioralObservation(DomainModel):
    """One committed greeting or completed response, with customer-visible data.

    messages contains the current assistant response; last_customer_message is
    only a successful typed send, so option actions may leave it absent. No
    evaluator/scenario object or browser/provider handle crosses this boundary.
    """

    observation_id: UUID4 = Field(default_factory=uuid4)
    kind: Literal["greeting", "customer_turn_completed"]
    messages: tuple[Message, ...] = ()
    evidence_event_ids: tuple[EvidenceReference, ...] = Field(min_length=1)
    state: PersonaState
    turn_count: int = Field(ge=0, strict=True)
    last_customer_message: Message | None = None
    repetition_facts: tuple[BehavioralRepetitionFact, ...] = ()

    @model_validator(mode="after")
    def validate_boundary(self) -> BehavioralObservation:
        _require_unique(self.evidence_event_ids, "observation event references")
        _require_unique(tuple(message.id for message in self.messages), "observation messages")
        _require_unique(tuple(fact.event_id for fact in self.repetition_facts), "repetition facts")
        if any(fact.event_id not in self.evidence_event_ids for fact in self.repetition_facts):
            raise ValueError("repetition facts require committed observation evidence")
        if any(message.actor != "assistant" for message in self.messages):
            raise ValueError("observation messages must belong to the assistant")
        if any(a.sequence >= b.sequence for a, b in zip(self.messages, self.messages[1:])):
            raise ValueError("observation messages must be ordered")
        if self.last_customer_message is not None:
            if self.last_customer_message.actor != "customer":
                raise ValueError("last_customer_message must belong to the customer")
            if self.messages and self.last_customer_message.sequence >= self.messages[0].sequence:
                raise ValueError("last customer message must precede the completed response")
            if self.last_customer_message.id in {message.id for message in self.messages}:
                raise ValueError("customer and assistant message identities must be distinct")
        if self.kind == "greeting":
            if self.turn_count != 0 or self.last_customer_message is not None:
                raise ValueError("greeting precedes any customer turn")
        elif self.turn_count == 0 or not self.messages:
            raise ValueError("completed customer turn requires a turn and assistant response")
        return self


class BehavioralSignal(DomainModel):
    """A detector classification, not an evaluator verdict or state mutation.

    References name acknowledged evidence; membership and backward ordering
    require observation/journal context and are not guessed by this model.
    Payload is finite JSON, at most 8 KiB in canonical UTF-8 and 16 containers
    deep. Rule conditions cannot inspect arbitrary payload fields.
    """

    detector_id: BehaviorIdentifier
    detector_type: BehaviorIdentifier
    detector_version: BehaviorDetectorVersion
    signal: BehaviorIdentifier
    deduplication_key: DeduplicationKey
    observation_id: UUID4
    source_event_ids: tuple[EvidenceReference, ...] = Field(min_length=1)
    source_message_ids: tuple[EvidenceReference, ...] = ()
    payload: JsonObject = Field(default_factory=dict)

    @field_validator("payload")
    @classmethod
    def validate_payload(cls, value: JsonObject) -> JsonObject:
        pending: list[tuple[object, int]] = [(value, 1)]
        while pending:
            item, depth = pending.pop()
            if isinstance(item, (dict, list)):
                if depth > MAX_SIGNAL_PAYLOAD_DEPTH:
                    raise ValueError("signal payload exceeds maximum nesting depth")
                children = item.values() if isinstance(item, dict) else item
                pending.extend((child, depth + 1) for child in children)
        try:
            encoded = json.dumps(
                value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        except (ValueError, UnicodeEncodeError) as exc:
            raise ValueError("signal payload must be finite UTF-8 JSON") from exc
        if len(encoded) > MAX_SIGNAL_PAYLOAD_BYTES:
            raise ValueError("signal payload exceeds maximum encoded size")
        return value

    @model_validator(mode="after")
    def validate_references(self) -> BehavioralSignal:
        _require_unique(self.source_event_ids, "signal event references")
        _require_unique(self.source_message_ids, "signal message references")
        return self


# Validate this alias with TypeAdapter at the detector boundary. A protocol alone
# cannot enforce output bounds; overflow must fail, never truncate evidence.
BehavioralDetectorOutput = Annotated[tuple[BehavioralSignal, ...], Field(max_length=32)]


class BehaviorReplaySourceHashes(DomainModel):
    """SHA-256 of exact source trace and effective-config artifact bytes."""

    trace_sha256: Sha256Digest
    effective_config_sha256: Sha256Digest


class BehaviorReplayTransition(DomainModel):
    """Calculated transition for one distinct signal, including no-match signals."""

    signal: BehavioralSignal
    occurrence: int = Field(gt=0, strict=True)
    before: PersonaState
    after: PersonaState
    applied_rule_ids: tuple[BehaviorIdentifier, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def validate_transition(self) -> BehaviorReplayTransition:
        _require_unique(self.applied_rule_ids, "applied rule IDs")
        if self.before.turn_number != self.after.turn_number:
            raise ValueError("behavioral transitions cannot change the turn number")
        if self.before.abandoned and (self.after != self.before or self.applied_rule_ids):
            raise ValueError("abandoned behavioral state cannot change or fire rules")
        return self


class BehaviorReplayExternalStateChange(DomainModel):
    """Calculated application of fields changed by a recorded persona/send event.

    Counterfactual replay assumes the recorded external decision, not an
    independently regenerated persona or alternate conversation.
    """

    source_event_id: EvidenceReference
    transition_index: int = Field(ge=0, strict=True)
    before: PersonaState
    after: PersonaState

    @model_validator(mode="after")
    def validate_change(self) -> BehaviorReplayExternalStateChange:
        if self.before.abandoned and not self.after.abandoned:
            raise ValueError("recorded external changes cannot reverse abandonment")
        if self.after.turn_number < self.before.turn_number:
            raise ValueError("recorded external changes cannot reverse turn progress")
        return self


class BehaviorReplayStateFinding(DomainModel):
    kind: Literal["state_mismatch"] = "state_mismatch"
    transition_index: int = Field(ge=0, strict=True)
    boundary: Literal["before", "after"]
    expected: PersonaState
    calculated: PersonaState

    @model_validator(mode="after")
    def require_difference(self) -> BehaviorReplayStateFinding:
        if self.expected == self.calculated:
            raise ValueError("state mismatch requires different states")
        return self


class BehaviorReplayRulesFinding(DomainModel):
    kind: Literal["applied_rules_mismatch"] = "applied_rules_mismatch"
    transition_index: int = Field(ge=0, strict=True)
    expected: tuple[BehaviorIdentifier, ...] = Field(max_length=32)
    calculated: tuple[BehaviorIdentifier, ...] = Field(max_length=32)

    @model_validator(mode="after")
    def require_difference(self) -> BehaviorReplayRulesFinding:
        _require_unique(self.expected, "expected applied rule IDs")
        _require_unique(self.calculated, "calculated applied rule IDs")
        if self.expected == self.calculated:
            raise ValueError("applied rule mismatch requires different rule lists")
        return self


class BehaviorReplayOccurrenceFinding(DomainModel):
    kind: Literal["occurrence_mismatch"] = "occurrence_mismatch"
    transition_index: int = Field(ge=0, strict=True)
    expected: int = Field(gt=0, strict=True)
    calculated: int = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def require_difference(self) -> BehaviorReplayOccurrenceFinding:
        if self.expected == self.calculated:
            raise ValueError("occurrence mismatch requires different counts")
        return self


BehaviorReplayFinding = Annotated[
    BehaviorReplayStateFinding | BehaviorReplayRulesFinding | BehaviorReplayOccurrenceFinding,
    Field(discriminator="kind"),
]


class BehaviorReplayResult(DomainModel):
    """Versioned deterministic output, not a run outcome or replay executable.

    Prefix means only a fully committed prefix was processed; prefix_reason is
    required. Counterfactual outside-continuation IDs name recorded observations
    excluded after terminal abandonment, not a fabricated alternate conversation.
    Cross-trace completeness/reference verification belongs to the replay reader.
    """

    version: Literal["1"] = "1"
    source_config_version: Literal["1"] = "1"
    source_hashes: BehaviorReplaySourceHashes
    mode: ReplayMode
    initial_state: PersonaState
    rules: tuple[BehaviorRule, ...] = Field(max_length=32)
    calculated_transitions: tuple[BehaviorReplayTransition, ...] = ()
    completeness: Literal["complete", "prefix"]
    prefix_reason: Annotated[str, Field(min_length=1, max_length=2000, pattern=r"\S")] | None = None
    comparison_findings: tuple[BehaviorReplayFinding, ...] = ()
    outside_continuation_observation_ids: tuple[UUID4, ...] = ()
    external_state_changes: tuple[BehaviorReplayExternalStateChange, ...] = ()
    external_state_policy: Literal["recorded_changed_fields"] = "recorded_changed_fields"

    @model_validator(mode="after")
    def validate_result(self) -> BehaviorReplayResult:
        if (self.completeness == "prefix") != (self.prefix_reason is not None):
            raise ValueError("only prefix results require a prefix reason")
        rule_ids = tuple(rule.id for rule in self.rules)
        _require_unique(rule_ids, "replay rule IDs")
        outside_ids = self.outside_continuation_observation_ids
        if len(outside_ids) != len(set(outside_ids)):
            raise ValueError("outside-continuation observation IDs must be unique")
        if self.mode == "verification" and outside_ids:
            raise ValueError("verification cannot exclude counterfactual observations")
        external = self.external_state_changes
        _require_unique(
            tuple(change.source_event_id for change in external), "external state event references"
        )
        if any(change.transition_index > len(self.calculated_transitions) for change in external):
            raise ValueError("external state change references an absent transition boundary")
        if any(
            first.transition_index > second.transition_index
            for first, second in zip(external, external[1:])
        ):
            raise ValueError("external state changes must be ordered")
        keys: set[tuple[str, str]] = set()
        counters: dict[str, int] = {}
        previous = self.initial_state
        previous_observation_id: UUID4 | None = None
        external_index = 0

        def apply_external(index: int) -> None:
            nonlocal previous, external_index
            while (
                external_index < len(external)
                and external[external_index].transition_index == index
            ):
                change = external[external_index]
                if self.mode == "counterfactual" and previous.abandoned:
                    raise ValueError("counterfactual replay cannot apply later external decisions")
                if change.before != previous:
                    raise ValueError("replay external state must be continuous")
                previous = change.after
                external_index += 1

        for index, transition in enumerate(self.calculated_transitions):
            apply_external(index)
            signal = transition.signal
            if (
                self.mode == "counterfactual"
                and previous.abandoned
                and signal.observation_id != previous_observation_id
            ):
                raise ValueError("counterfactual replay cannot continue after abandonment")
            key = (signal.detector_id, signal.deduplication_key)
            if key in keys:
                raise ValueError("replay transitions must process each signal key once")
            keys.add(key)
            if signal.observation_id in outside_ids:
                raise ValueError("outside-continuation observations cannot have transitions")
            count = counters.get(signal.signal, 0) + 1
            if transition.occurrence != count:
                raise ValueError("replay signal occurrence counts must be contiguous")
            counters[signal.signal] = count
            before = transition.before
            if (
                before.frustration != previous.frustration
                or before.willingness_to_continue != previous.willingness_to_continue
                or before.abandoned != previous.abandoned
                or before.turn_number < previous.turn_number
            ):
                raise ValueError("replay behavioral state must be continuous")
            if not set(transition.applied_rule_ids).issubset(rule_ids):
                raise ValueError("transition references an undeclared rule")
            previous = transition.after
            previous_observation_id = signal.observation_id
        apply_external(len(self.calculated_transitions))
        if outside_ids and not previous.abandoned:
            raise ValueError("outside-continuation observations require terminal abandonment")
        for finding in self.comparison_findings:
            if finding.transition_index >= len(self.calculated_transitions):
                raise ValueError("comparison finding references an absent transition")
            transition = self.calculated_transitions[finding.transition_index]
            calculated: PersonaState | tuple[str, ...] | int
            if isinstance(finding, BehaviorReplayStateFinding):
                calculated = transition.before if finding.boundary == "before" else transition.after
            elif isinstance(finding, BehaviorReplayRulesFinding):
                calculated = transition.applied_rule_ids
            else:
                calculated = transition.occurrence
            if finding.calculated != calculated:
                raise ValueError("comparison finding must cite the calculated transition value")
        return self

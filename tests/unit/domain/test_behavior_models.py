"""Behavior contracts reject ambiguous evidence without executing detectors."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4, uuid5

import pytest
from pydantic import TypeAdapter, ValidationError

from syntheticcx.domain.behavior import (
    MAX_SIGNAL_PAYLOAD_BYTES,
    MAX_SIGNAL_PAYLOAD_DEPTH,
    BehavioralDetectorOutput,
    BehavioralObservation,
    BehavioralSignal,
    BehaviorReplayOccurrenceFinding,
    BehaviorReplayResult,
    BehaviorReplayRulesFinding,
    BehaviorReplayStateFinding,
    BehaviorReplayTransition,
)
from syntheticcx.domain.models import BehavioralDetectorMetadata, Message, PersonaState
from syntheticcx.errors import BehaviorError, SyntheticCXError

NOW = datetime(2026, 10, 10, tzinfo=UTC)
OBSERVATION_ID = uuid4()


def message(
    actor: str = "assistant", sequence: int = 1, identifier: str = "assistant-1"
) -> Message:
    return Message.model_validate(
        {
            "id": identifier,
            "actor": actor,
            "sequence": sequence,
            "content": "Customer-visible text",
            "observed_at": NOW,
        }
    )


def observation_payload() -> dict[str, Any]:
    return {
        "kind": "customer_turn_completed",
        "messages": [message()],
        "evidence_event_ids": ["event-response"],
        "state": PersonaState(),
        "turn_count": 1,
        "last_customer_message": message("customer", 0, "customer-1"),
    }


def signal_payload(key: str = "completed-turn-1") -> dict[str, Any]:
    return {
        "detector_id": "completed-turn",
        "detector_type": "customer_turn_completed",
        "detector_version": "1",
        "signal": "customer_turn_completed",
        "deduplication_key": key,
        "observation_id": OBSERVATION_ID,
        "source_event_ids": ["event-response"],
        "source_message_ids": ["assistant-1"],
    }


def transition_payload(key: str = "completed-turn-1") -> dict[str, Any]:
    return {
        "signal": signal_payload(key),
        "occurrence": 1,
        "before": PersonaState(),
        "after": PersonaState(),
    }


def replay_payload() -> dict[str, Any]:
    return {
        "source_hashes": {"trace_sha256": "a" * 64, "effective_config_sha256": "b" * 64},
        "mode": "verification",
        "initial_state": PersonaState(),
        "rules": [],
        "calculated_transitions": [transition_payload()],
        "completeness": "complete",
    }


def test_observation_identity_is_random_not_a_customer_text_fingerprint() -> None:
    first = BehavioralObservation.model_validate(observation_payload())
    second = BehavioralObservation.model_validate(observation_payload())
    assert first.observation_id.version == second.observation_id.version == 4
    assert first.observation_id != second.observation_id
    assert first.messages == second.messages
    with pytest.raises(ValidationError):
        BehavioralObservation.model_validate(
            {**observation_payload(), "observation_id": uuid5(UUID(int=0), "customer text")}
        )


@pytest.mark.parametrize("field", ["scenario", "criteria", "evaluator", "expected_facts"])
def test_observation_forbids_hidden_or_evaluator_context(field: str) -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        BehavioralObservation.model_validate({**observation_payload(), field: {}})


def test_empty_greeting_and_option_turn_have_distinct_boundaries() -> None:
    greeting = BehavioralObservation(
        kind="greeting",
        evidence_event_ids=("conversation-start",),
        state=PersonaState(),
        turn_count=0,
    )
    assert greeting.messages == () and greeting.last_customer_message is None
    payload = observation_payload()
    payload.pop("last_customer_message")
    option_turn = BehavioralObservation.model_validate(payload)
    assert option_turn.turn_count == 1 and option_turn.last_customer_message is None


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"turn_count": True}, "valid integer"),
        ({"turn_count": "1"}, "valid integer"),
        ({"turn_count": 0}, "requires a turn"),
        ({"messages": []}, "requires a turn"),
        ({"messages": [message("system")]}, "assistant"),
        ({"messages": [message(), message()]}, "unique"),
        ({"messages": [message(sequence=2), message(identifier="m2")]}, "ordered"),
        ({"evidence_event_ids": []}, "at least 1"),
        ({"evidence_event_ids": ["event", "event"]}, "unique"),
        ({"evidence_event_ids": [" "]}, "pattern"),
        ({"last_customer_message": message()}, "customer"),
        ({"last_customer_message": message("customer", 2)}, "precede"),
        ({"last_customer_message": message("customer", 0, "assistant-1")}, "distinct"),
        ({"kind": "greeting"}, "precedes"),
    ],
)
def test_observation_rejects_invalid_boundaries(change: dict[str, Any], error: str) -> None:
    with pytest.raises(ValidationError, match=error):
        BehavioralObservation.model_validate({**observation_payload(), **change})


@pytest.mark.parametrize("field", ["detector_id", "detector_type", "signal"])
@pytest.mark.parametrize("value", ["", "   ", "9name", "non ascii é", "a" * 81])
def test_signal_identifiers_are_meaningful_bounded_names(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        BehavioralSignal.model_validate({**signal_payload(), field: value})


@pytest.mark.parametrize(
    "change",
    [
        {"source_event_ids": []},
        {"source_event_ids": ["event", "event"]},
        {"source_message_ids": ["message", "message"]},
        {"source_message_ids": ["message with spaces"]},
        {"deduplication_key": " "},
        {"deduplication_key": "x" * 201},
        {"detector_version": " "},
        {"detector_version": "v" * 81},
        {"observation_id": "text-content-hash"},
    ],
)
def test_signal_requires_explicit_identity_and_unique_evidence(change: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        BehavioralSignal.model_validate({**signal_payload(), **change})


def test_payload_limit_counts_utf8_bytes_not_characters() -> None:
    # Canonical {"x":"..."} has eight bytes of object/string syntax.
    exact = BehavioralSignal.model_validate(
        {**signal_payload(), "payload": {"x": "a" * (MAX_SIGNAL_PAYLOAD_BYTES - 8)}}
    )
    assert exact.payload["x"]
    with pytest.raises(ValidationError, match="encoded size"):
        BehavioralSignal.model_validate(
            {**signal_payload(), "payload": {"x": "é" * (MAX_SIGNAL_PAYLOAD_BYTES // 2)}}
        )


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), "\ud800"])
def test_payload_rejects_nonfinite_and_invalid_utf8_values(value: object) -> None:
    with pytest.raises(ValidationError):
        BehavioralSignal.model_validate({**signal_payload(), "payload": {"nested": [value]}})


def test_payload_depth_is_bounded_even_for_small_json() -> None:
    nested: dict[str, Any] = {}
    for _ in range(MAX_SIGNAL_PAYLOAD_DEPTH - 1):
        nested = {"child": nested}
    BehavioralSignal.model_validate({**signal_payload(), "payload": nested})
    with pytest.raises(ValidationError, match="nesting depth"):
        BehavioralSignal.model_validate({**signal_payload(), "payload": {"child": nested}})


def test_detector_output_cap_rejects_overflow_instead_of_truncating() -> None:
    adapter = TypeAdapter(BehavioralDetectorOutput)
    signals = tuple(BehavioralSignal.model_validate(signal_payload(str(i))) for i in range(32))
    assert len(adapter.validate_python(signals)) == 32
    with pytest.raises(ValidationError):
        adapter.validate_python((*signals, signals[0]))


@pytest.mark.parametrize(
    "change",
    [
        {"output_signals": []},
        {"output_signals": ["repeat", "repeat"]},
        {"output_signals": [{"name": "repeat"}]},
        {"output_signals": [f"signal{i}" for i in range(33)]},
        {"mode": "semantic"},
        {"detector_version": " "},
        {"factory": "package.class"},
    ],
)
def test_metadata_declares_unique_names_and_exact_mode_without_factories(
    change: dict[str, Any],
) -> None:
    payload = {
        "detector_id": "detector",
        "detector_type": "custom_signal",
        "detector_version": "1.2.3+build-7",
        "mode": "deterministic",
        "output_signals": ["custom_signal"],
        "signal_schemas": {"custom_signal": {"type": "object"}},
    }
    with pytest.raises(ValidationError):
        BehavioralDetectorMetadata.model_validate({**payload, **change})


def test_behavior_error_has_its_own_nonretrying_category() -> None:
    error = BehaviorError("behavior.detect", "detector failed")
    assert isinstance(error, SyntheticCXError)
    assert error.category == "behavior" and not error.retryable
    assert str(error) == "behavior.detect: detector failed"


@pytest.mark.parametrize("change", [{"version": "2"}, {"source_config_version": "2"}])
def test_replay_versions_are_explicit_not_assumed_compatible(change: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        BehaviorReplayResult.model_validate({**replay_payload(), **change})


@pytest.mark.parametrize("digest", ["", "a" * 63, "a" * 65, "A" * 64, "z" * 64])
def test_replay_source_hashes_are_sha256_hex(digest: str) -> None:
    payload = replay_payload()
    payload["source_hashes"]["trace_sha256"] = digest
    with pytest.raises(ValidationError):
        BehaviorReplayResult.model_validate(payload)


@pytest.mark.parametrize(
    "change",
    [
        {"completeness": "prefix"},
        {"prefix_reason": "uncommitted tail"},
        {"completeness": "prefix", "prefix_reason": " "},
        {"outside_continuation_observation_ids": [uuid4()]},
    ],
)
def test_replay_cannot_claim_complete_or_counterfactual_coverage_ambiguously(
    change: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        BehaviorReplayResult.model_validate({**replay_payload(), **change})


def test_prefix_does_not_claim_full_run_verification() -> None:
    result = BehaviorReplayResult.model_validate(
        {**replay_payload(), "completeness": "prefix", "prefix_reason": "Uncommitted tail"}
    )
    assert result.completeness == "prefix" and not result.comparison_findings


@pytest.mark.parametrize(
    "change",
    [
        {"occurrence": True},
        {"occurrence": 0},
        {"after": PersonaState(turn_number=1)},
        {"applied_rule_ids": ["rule", "rule"]},
        {"before": PersonaState(abandoned=True), "after": PersonaState()},
        {
            "before": PersonaState(abandoned=True),
            "after": PersonaState(abandoned=True),
            "applied_rule_ids": ["rule"],
        },
    ],
)
def test_transition_preserves_terminal_state_and_owned_turn_count(change: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        BehaviorReplayTransition.model_validate({**transition_payload(), **change})


@pytest.mark.parametrize("case", ["duplicate", "counter", "state", "turn", "rule"])
def test_replay_rejects_duplicate_keys_counter_gaps_and_broken_continuity(case: str) -> None:
    first = transition_payload()
    second = transition_payload("completed-turn-2")
    second["occurrence"] = 2
    if case == "duplicate":
        second["signal"] = first["signal"]
    elif case == "counter":
        second["occurrence"] = 3
    elif case == "state":
        second["before"] = second["after"] = PersonaState(frustration=0.3)
    elif case == "turn":
        first["before"] = first["after"] = PersonaState(turn_number=1)
    else:
        second["applied_rule_ids"] = ["undeclared"]
    with pytest.raises(ValidationError):
        BehaviorReplayResult.model_validate(
            {**replay_payload(), "calculated_transitions": [first, second]}
        )


def test_replay_allows_dispatch_turn_advancement_without_behavioral_state_changes() -> None:
    first = transition_payload()
    second = transition_payload("completed-turn-2")
    second.update(
        occurrence=2, before=PersonaState(turn_number=1), after=PersonaState(turn_number=1)
    )
    result = BehaviorReplayResult.model_validate(
        {**replay_payload(), "calculated_transitions": [first, second]}
    )
    assert result.calculated_transitions[-1].occurrence == 2


def test_counterfactual_excludes_only_unprocessed_observations_after_abandonment() -> None:
    transition = transition_payload()
    transition["after"] = PersonaState(abandoned=True)
    outside_id = uuid4()
    payload = {
        **replay_payload(),
        "mode": "counterfactual",
        "calculated_transitions": [transition],
        "outside_continuation_observation_ids": [outside_id],
    }
    assert BehaviorReplayResult.model_validate(payload).outside_continuation_observation_ids
    for change in (
        {"calculated_transitions": [transition_payload()]},
        {"outside_continuation_observation_ids": [OBSERVATION_ID]},
        {"outside_continuation_observation_ids": [outside_id, outside_id]},
    ):
        with pytest.raises(ValidationError):
            BehaviorReplayResult.model_validate({**payload, **change})


@pytest.mark.parametrize("kind", ["state", "rules", "counter"])
def test_comparison_findings_require_real_differences(kind: str) -> None:
    with pytest.raises(ValidationError, match="different"):
        if kind == "state":
            BehaviorReplayStateFinding(
                transition_index=0,
                boundary="after",
                expected=PersonaState(),
                calculated=PersonaState(),
            )
        elif kind == "rules":
            BehaviorReplayRulesFinding(transition_index=0, expected=(), calculated=())
        else:
            BehaviorReplayOccurrenceFinding(transition_index=0, expected=1, calculated=1)


@pytest.mark.parametrize("case", ["index", "calculated"])
def test_comparison_findings_cite_existing_calculated_values(case: str) -> None:
    finding = {
        "kind": "occurrence_mismatch",
        "transition_index": 1 if case == "index" else 0,
        "expected": 3,
        "calculated": 1 if case == "index" else 2,
    }
    with pytest.raises(ValidationError):
        BehaviorReplayResult.model_validate({**replay_payload(), "comparison_findings": [finding]})


def test_counterfactual_cannot_silently_process_a_new_observation_after_abandonment() -> None:
    first = transition_payload()
    first["after"] = PersonaState(abandoned=True)
    second = transition_payload("completed-turn-2")
    second["signal"]["observation_id"] = uuid4()
    second.update(occurrence=2, before=first["after"], after=first["after"])
    with pytest.raises(ValidationError, match="cannot continue after abandonment"):
        BehaviorReplayResult.model_validate(
            {
                **replay_payload(),
                "mode": "counterfactual",
                "calculated_transitions": [first, second],
            }
        )


def test_replay_rule_ids_cannot_be_ambiguous() -> None:
    rule = {
        "id": "leave",
        "priority": 1,
        "when": {"signal": "customer_turn_completed"},
        "effects": {"abandon": True},
    }
    with pytest.raises(ValidationError, match="replay rule IDs must be unique"):
        BehaviorReplayResult.model_validate({**replay_payload(), "rules": [rule, rule]})


def test_metadata_accepts_exact_model_backed_identity_without_implementing_an_adapter() -> None:
    metadata = BehavioralDetectorMetadata(
        detector_id="trusted-detector",
        detector_type="future_classifier",
        detector_version="1.2.3+build-7",
        mode="model_backed",
        output_signals=("future_signal",),
        signal_schemas={"future_signal": {"type": "object"}},
    )
    assert metadata.output_signals == ("future_signal",)
    with pytest.raises(ValidationError, match="frozen"):
        metadata.mode = "deterministic"


def test_replay_interleaves_external_persona_and_send_changes_explicitly() -> None:
    initial = PersonaState()
    changed = PersonaState(frustration=0.3, turn_number=1)
    final = PersonaState(frustration=0.4, turn_number=1)
    transition = {**transition_payload(), "before": changed, "after": changed}
    result = BehaviorReplayResult.model_validate(
        {
            **replay_payload(),
            "calculated_transitions": [transition],
            "external_state_changes": [
                {
                    "source_event_id": "persona-send",
                    "transition_index": 0,
                    "before": initial,
                    "after": changed,
                },
                {
                    "source_event_id": "persona-after",
                    "transition_index": 1,
                    "before": changed,
                    "after": final,
                },
            ],
        }
    )
    assert result.external_state_policy == "recorded_changed_fields"
    assert result.external_state_changes[-1].after == final


@pytest.mark.parametrize(
    "change",
    [
        {"transition_index": True},
        {"transition_index": 2},
        {"before": PersonaState(frustration=0.4)},
    ],
)
def test_replay_external_records_require_bounded_contiguous_state(change: dict[str, Any]) -> None:
    external = {
        "source_event_id": "persona-event",
        "transition_index": 0,
        "before": PersonaState(),
        "after": PersonaState(),
    }
    with pytest.raises(ValidationError):
        BehaviorReplayResult.model_validate(
            {
                **replay_payload(),
                "external_state_changes": [{**external, **change}],
            }
        )


def test_replay_external_events_have_unique_ids_and_chronological_indices() -> None:
    external = {
        "source_event_id": "persona-event",
        "transition_index": 0,
        "before": PersonaState(),
        "after": PersonaState(),
    }
    for records in (
        [external, external],
        [{**external, "transition_index": 1}, {**external, "source_event_id": "other"}],
    ):
        with pytest.raises(ValidationError):
            BehaviorReplayResult.model_validate(
                {
                    **replay_payload(),
                    "external_state_changes": records,
                }
            )


def test_counterfactual_external_decisions_cannot_continue_after_abandonment() -> None:
    transition = {**transition_payload(), "after": PersonaState(abandoned=True)}
    with pytest.raises(ValidationError, match="later external decisions"):
        BehaviorReplayResult.model_validate(
            {
                **replay_payload(),
                "mode": "counterfactual",
                "calculated_transitions": [transition],
                "external_state_changes": [
                    {
                        "source_event_id": "later-persona",
                        "transition_index": 1,
                        "before": PersonaState(abandoned=True),
                        "after": PersonaState(abandoned=True),
                    }
                ],
            }
        )


@pytest.mark.parametrize("schemas", [{}, {"wrong_signal": {"type": "object"}}])
def test_metadata_requires_exact_advertised_payload_schema_names(schemas: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        BehavioralDetectorMetadata(
            detector_id="detector",
            detector_type="custom",
            detector_version="1",
            mode="deterministic",
            output_signals=("custom_signal",),
            signal_schemas=schemas,
        )


def test_metadata_does_not_invent_missing_payload_schemas() -> None:
    with pytest.raises(ValidationError):
        BehavioralDetectorMetadata.model_validate(
            {
                "detector_id": "detector",
                "detector_type": "custom",
                "detector_version": "1",
                "mode": "deterministic",
                "output_signals": ["custom_signal"],
            }
        )

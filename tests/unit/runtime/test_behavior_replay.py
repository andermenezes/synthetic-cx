"""Offline replay validates committed evidence, never reconstructs classifications."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
import yaml
from typer.testing import CliRunner

from syntheticcx.config.models import BehaviorRule
from syntheticcx.domain.models import PersonaState
from syntheticcx.runtime.behavior import apply_behavior
from syntheticcx.runtime.behavior_replay import (
    BehaviorReplayError,
    replay_behavior,
    write_behavior_replay,
)

NOW = "2026-10-10T12:00:00Z"


def rule(*, occurrence: int = 1, abandon: bool = False) -> dict[str, Any]:
    return {
        "id": "owned-effect",
        "priority": 10,
        "when": {"signal": "customer_turn_completed", "occurrence_at_least": occurrence},
        "effects": {"abandon": True} if abandon else {"frustration_delta": 0.2},
    }


def source(
    tmp_path: Path,
    *,
    turns: int = 3,
    rules: list[dict[str, Any]] | None = None,
    initial_frustration: float = 0,
    persona_willingness: float | None = None,
):
    directory = tmp_path / "run"
    directory.mkdir()
    selected = [rule()] if rules is None else rules
    metadata = {
        "detector_id": "completed-turn",
        "detector_type": "customer_turn_completed",
        "detector_version": "1",
        "mode": "deterministic",
        "output_signals": ["customer_turn_completed"],
        "signal_schemas": {
            "customer_turn_completed": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            }
        },
    }
    config = {
        "source_content_hash": "source-hash",
        "behavior_detectors": [metadata],
        "effective_config": {
            "persona": {"initial_state": PersonaState().model_dump(), "behavioral_rules": selected},
            "behavior": {
                "detectors": [{"id": "completed-turn", "type": "customer_turn_completed"}]
            },
        },
    }
    run_metadata = {
        "id": "run",
        "scenario_id": "fixture",
        "persona_id": "fixture",
        "started_at": NOW,
        "ended_at": NOW,
        "lifecycle_status": "finalized",
        "outcome": "passed",
        "termination_reason": "fixture",
        "metadata": {
            "configuration_version": "1",
            "framework_version": "0.1.0",
            "persona_model": {"provider": "openai", "model": "offline"},
            "evaluation_model": {"provider": "openai", "model": "offline"},
            "usage": {"unavailable_reason": "no calls"},
            "runtime_platform": "test",
            "owner": {
                "owner_uuid": "owner",
                "hostname": "test",
                "pid": 1,
                "process_start_evidence": "test",
            },
            "behavior_detectors": [metadata],
        },
    }
    events: list[dict[str, Any]] = []

    def append(kind: str, data: dict[str, Any]):
        event = {
            "id": f"event-{len(events)}",
            "sequence": len(events),
            "timestamp": NOW,
            "type": kind,
            "data": data,
        }
        events.append(event)
        return event

    def message(actor: str, turn: int):
        return append(
            "message_observed",
            {
                "message": {
                    "id": f"{actor}-{turn}",
                    "sequence": sum(item["type"] == "message_observed" for item in events),
                    "actor": actor,
                    "content": "Visible customer context",
                    "observed_at": NOW,
                }
            },
        )

    state = PersonaState()
    greeting = append(
        "conversation_started",
        {"greeting_observed": False, "grace_seconds": 0, "greeting_message_ids": []},
    )
    if initial_frustration:
        changed = state.model_copy(update={"frustration": initial_frustration})
        append(
            "persona_state_changed",
            {
                "trigger_ids": [greeting["id"]],
                "before": state.model_dump(),
                "after": changed.model_dump(),
                "applied_rule_ids": [],
            },
        )
        state = changed
    append(
        "behavior_observation_recorded",
        {
            "observation_id": str(uuid4()),
            "kind": "greeting",
            "evidence_event_ids": [greeting["id"]],
            "message_ids": [],
            "detector_ids": ["completed-turn"],
            "state": state.model_dump(),
            "turn_count": 0,
            "signal_count": 0,
        },
    )
    fired: set[str] = set()
    for turn in range(1, turns + 1):
        customer = message("customer", turn)
        before = state
        state = state.model_copy(update={"turn_number": turn})
        append(
            "persona_state_changed",
            {
                "trigger_ids": [customer["id"]],
                "before": before.model_dump(),
                "after": state.model_dump(),
                "applied_rule_ids": [],
            },
        )
        assistant = message("assistant", turn)
        observation_id = str(uuid4())
        append(
            "behavior_observation_recorded",
            {
                "observation_id": observation_id,
                "kind": "customer_turn_completed",
                "evidence_event_ids": [assistant["id"]],
                "message_ids": [assistant["data"]["message"]["id"]],
                "detector_ids": ["completed-turn"],
                "state": state.model_dump(),
                "turn_count": turn,
                "signal_count": 1,
            },
        )
        signal = append(
            "behavioral_signal_recorded",
            {
                "signal": {
                    "detector_id": "completed-turn",
                    "detector_type": "customer_turn_completed",
                    "detector_version": "1",
                    "signal": "customer_turn_completed",
                    "deduplication_key": f"completed-{turn}",
                    "observation_id": observation_id,
                    "source_event_ids": [assistant["id"]],
                    "source_message_ids": [assistant["data"]["message"]["id"]],
                    "payload": {},
                }
            },
        )
        change = apply_behavior(
            state,
            [BehaviorRule.model_validate(item) for item in selected],
            signal="customer_turn_completed",
            trigger_ids=[signal["id"]],
            occurrence=turn,
            fired_rule_ids=fired,
        )
        append(
            "behavioral_state_changed",
            {
                "observation_id": observation_id,
                "signal_event_id": signal["id"],
                "signal": "customer_turn_completed",
                "occurrence": turn,
                "before": change.before.model_dump(),
                "after": change.after.model_dump(),
                "applied_rule_ids": change.applied_rule_ids,
            },
        )
        state = change.after
        fired.update(change.applied_rule_ids)
        if turn == 1 and persona_willingness is not None:
            changed = state.model_copy(update={"willingness_to_continue": persona_willingness})
            append(
                "persona_state_changed",
                {
                    "trigger_ids": [events[-1]["id"]],
                    "before": state.model_dump(),
                    "after": changed.model_dump(),
                    "applied_rule_ids": [],
                },
            )
            state = changed
    publish(directory, events, config, run_metadata)
    return directory, events, config, run_metadata


def publish(directory, events, config, metadata):
    (directory / "trace.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    (directory / "config.yaml").write_text(yaml.safe_dump(config))
    (directory / "metadata.json").write_text(json.dumps(metadata))


def test_verification_exact_hashes_determinism_and_read_only(tmp_path):
    directory, *_ = source(tmp_path)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    first = replay_behavior(directory)
    second = replay_behavior(directory)
    assert first == second
    assert first.completeness == "complete" and not first.comparison_findings
    assert [item.occurrence for item in first.calculated_transitions] == [1, 2, 3]
    assert first.source_hashes.trace_sha256 == sha256(before["trace.jsonl"]).hexdigest()
    assert first.source_hashes.effective_config_sha256 == sha256(before["config.yaml"]).hexdigest()
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before
    write_behavior_replay(first, tmp_path / "result-a.json")
    write_behavior_replay(second, tmp_path / "result-b.json")
    assert (tmp_path / "result-a.json").read_bytes() == (tmp_path / "result-b.json").read_bytes()


@pytest.mark.parametrize(
    "corruption",
    [
        "forward_ref",
        "message_ref",
        "boundary_ref",
        "duplicate_key",
        "signal_link",
        "signal_order",
        "sequence",
        "state",
        "rule_id",
        "version",
        "metadata",
        "turn",
        "detector_order",
    ],
)
def test_corrupted_committed_evidence_is_not_verified(tmp_path, corruption):
    directory, events, config, metadata = source(tmp_path)
    signals = [item for item in events if item["type"] == "behavioral_signal_recorded"]
    changes = [item for item in events if item["type"] == "behavioral_state_changed"]
    boundaries = [item for item in events if item["type"] == "behavior_observation_recorded"]
    if corruption == "forward_ref":
        signals[0]["data"]["signal"]["source_event_ids"] = [changes[-1]["id"]]
    elif corruption == "message_ref":
        signals[0]["data"]["signal"]["source_message_ids"] = ["nonexistent"]
    elif corruption == "boundary_ref":
        signals[0]["data"]["signal"]["source_event_ids"] = [boundaries[1]["id"]]
    elif corruption == "duplicate_key":
        signals[1]["data"]["signal"]["deduplication_key"] = signals[0]["data"]["signal"][
            "deduplication_key"
        ]
    elif corruption == "signal_link":
        changes[0]["data"]["signal_event_id"] = signals[-1]["id"]
    elif corruption == "signal_order":
        a, b = events.index(signals[0]), events.index(changes[0])
        events[a], events[b] = events[b], events[a]
        for index, event in enumerate(events):
            event["sequence"] = index
    elif corruption == "sequence":
        events[1]["sequence"] = 99
    elif corruption == "state":
        changes[0]["data"]["before"]["frustration"] = 0.5
    elif corruption == "rule_id":
        changes[0]["data"]["applied_rule_ids"] = ["unknown-rule"]
    elif corruption == "version":
        signals[0]["data"]["signal"]["detector_version"] = "2"
    elif corruption == "metadata":
        metadata["metadata"]["behavior_detectors"][0] = {
            **metadata["metadata"]["behavior_detectors"][0],
            "mode": "model_backed",
        }
    elif corruption == "turn":
        boundaries[1]["data"]["turn_count"] = 2
    elif corruption == "detector_order":
        boundaries[1]["data"]["detector_ids"] = []
    publish(directory, events, config, metadata)
    with pytest.raises(BehaviorReplayError) as failure:
        replay_behavior(directory)
    assert failure.value.kind in {"invalid_source", "unsupported_source"}


def test_counter_corruption_is_an_explicit_mismatch(tmp_path):
    directory, events, config, metadata = source(tmp_path)
    next(item for item in events if item["type"] == "behavioral_state_changed")["data"][
        "occurrence"
    ] = 9
    publish(directory, events, config, metadata)
    result = replay_behavior(directory)
    assert any(item.kind == "occurrence_mismatch" for item in result.comparison_findings)


def test_changed_effect_is_valid_replay_mismatch(tmp_path):
    directory, events, config, metadata = source(tmp_path)
    config["effective_config"]["persona"]["behavioral_rules"][0]["effects"]["frustration_delta"] = (
        0.1
    )
    publish(directory, events, config, metadata)
    result = replay_behavior(directory)
    assert result.completeness == "complete"
    assert any(item.kind == "state_mismatch" for item in result.comparison_findings)


@pytest.mark.parametrize("tail", ["torn", "missing_transition", "missing_signal", "incomplete_run"])
def test_only_complete_batches_are_returned_as_verified_prefix(tmp_path, tail):
    directory, events, config, metadata = source(tmp_path)
    if tail == "missing_transition":
        events.pop()
    elif tail == "missing_signal":
        events = events[:-2]
    elif tail == "incomplete_run":
        metadata.update(
            lifecycle_status="incomplete", outcome=None, ended_at=None, termination_reason=None
        )
    publish(directory, events, config, metadata)
    if tail == "torn":
        with (directory / "trace.jsonl").open("ab") as stream:
            stream.write(b'{"unfinished":')
    result = replay_behavior(directory)
    assert result.completeness == "prefix" and result.prefix_reason
    assert len(result.calculated_transitions) == (2 if tail.startswith("missing") else 3)
    assert not result.comparison_findings


def test_malformed_newline_terminated_tail_is_invalid_not_prefix(tmp_path):
    directory, *_ = source(tmp_path)
    with (directory / "trace.jsonl").open("ab") as stream:
        stream.write(b'{"unfinished":\n')
    with pytest.raises(BehaviorReplayError, match="malformed committed event"):
        replay_behavior(directory)


def test_legacy_evidence_is_actionably_unsupported(tmp_path):
    directory, events, config, metadata = source(tmp_path)
    del config["behavior_detectors"]
    del metadata["metadata"]["behavior_detectors"]
    publish(directory, events, config, metadata)
    with pytest.raises(BehaviorReplayError) as failure:
        replay_behavior(directory)
    assert failure.value.kind == "unsupported_source"
    assert "new run" in failure.value.message


def test_earlier_counterfactual_abandonment_marks_outside_continuation(tmp_path):
    directory, events, *_ = source(tmp_path, turns=6, rules=[rule(occurrence=6, abandon=True)])
    override = tmp_path / "rules.yaml"
    override.write_text(yaml.safe_dump({"rules": [rule(occurrence=2, abandon=True)]}))
    result = replay_behavior(directory, rules=override)
    assert result.mode == "counterfactual" and result.completeness == "complete"
    assert len(result.calculated_transitions) == 2
    assert result.calculated_transitions[-1].after.abandoned
    expected = [
        item["data"]["observation_id"]
        for item in events
        if item["type"] == "behavior_observation_recorded" and item["data"]["turn_count"] > 2
    ]
    assert [str(item) for item in result.outside_continuation_observation_ids] == expected
    assert not result.comparison_findings


@pytest.mark.parametrize(
    "document",
    [
        {"unexpected": "value", "rules": []},
        {"rules": [], "persona": {}},
        {"rules": [rule(), rule()]},
        {"rules": [{**rule(), "when": {"signal": "unknown"}}]},
    ],
)
def test_counterfactual_rules_only_contract_is_strict(tmp_path, document):
    directory, *_ = source(tmp_path)
    override = tmp_path / "rules.yaml"
    override.write_text(yaml.safe_dump(document))
    with pytest.raises(BehaviorReplayError) as failure:
        replay_behavior(directory, rules=override)
    assert failure.value.kind == "invalid_rules"


def test_no_match_boundaries_and_run_isolation(tmp_path):
    directory, *_ = source(tmp_path, rules=[])
    result = replay_behavior(directory)
    assert len(result.calculated_transitions) == 3
    assert all(
        item.before == item.after and not item.applied_rule_ids
        for item in result.calculated_transitions
    )
    assert replay_behavior(directory) == result


def test_actual_cli_is_offline_and_no_clobber(tmp_path, monkeypatch):
    from syntheticcx import cli
    from syntheticcx.adapters.channel import dynamic, whatsapp
    from syntheticcx.application import run as application
    from syntheticcx.runtime import behavior_signals

    def forbidden(*args, **kwargs):
        raise AssertionError("offline replay must not construct live resources")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(cli, "execute_run", forbidden)
    for name in [
        "OpenAIProvider",
        "OpenAICompatibleProvider",
        "TinyTroupePersonaSimulator",
        "FilesystemStorage",
        "prepare_settings",
    ]:
        monkeypatch.setattr(application, name, forbidden)
    monkeypatch.setattr(whatsapp, "WhatsAppWebChannel", forbidden)
    monkeypatch.setattr(dynamic, "DynamicPlaywrightChannel", forbidden)
    monkeypatch.setattr(behavior_signals, "prepare_behavior", forbidden)
    monkeypatch.setattr(behavior_signals, "BehaviorDetectors", forbidden)
    directory, *_ = source(tmp_path)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    output = tmp_path / "replay.json"
    runner = CliRunner()
    result = runner.invoke(cli.app, ["replay-behavior", str(directory), "--output", str(output)])
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text())["mode"] == "verification"
    original = output.read_bytes()
    second = runner.invoke(cli.app, ["replay-behavior", str(directory), "--output", str(output)])
    assert second.exit_code == 3 and "output_exists" in second.output
    assert output.read_bytes() == original
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before
    inside = runner.invoke(
        cli.app, ["replay-behavior", str(directory), "--output", str(directory / "replay.json")]
    )
    assert inside.exit_code == 3 and not (directory / "replay.json").exists()


@pytest.mark.parametrize("case,exit_code", [("mismatch", 4), ("prefix", 5), ("unsupported", 2)])
def test_actual_cli_distinguishes_valid_mismatch_prefix_and_invalid_source(
    tmp_path, case, exit_code
):
    from syntheticcx.cli import app

    directory, events, config, metadata = source(tmp_path)
    if case == "mismatch":
        config["effective_config"]["persona"]["behavioral_rules"][0]["effects"][
            "frustration_delta"
        ] = 0.1
    elif case == "prefix":
        events.pop()
    else:
        del config["behavior_detectors"]
    publish(directory, events, config, metadata)
    output = tmp_path / "result.json"
    result = CliRunner().invoke(app, ["replay-behavior", str(directory), "--output", str(output)])
    assert result.exit_code == exit_code, result.output
    assert output.exists() == (case != "unsupported")


def test_actual_cli_counterfactual_rules_option(tmp_path):
    from syntheticcx.cli import app

    directory, *_ = source(tmp_path, turns=6, rules=[rule(occurrence=6, abandon=True)])
    override = tmp_path / "rules.yaml"
    override.write_text(yaml.safe_dump({"rules": [rule(occurrence=2, abandon=True)]}))
    output = tmp_path / "counterfactual.json"
    result = CliRunner().invoke(
        app, ["replay-behavior", str(directory), "--output", str(output), "--rules", str(override)]
    )
    assert result.exit_code == 0, result.output
    document = json.loads(output.read_text())
    assert document["mode"] == "counterfactual"
    assert len(document["calculated_transitions"]) == 2
    assert len(document["outside_continuation_observation_ids"]) == 4


def test_explicit_initialization_state_change_is_replayed_not_reset(tmp_path):
    directory, *_ = source(tmp_path, initial_frustration=0.1)
    result = replay_behavior(directory)
    assert not result.comparison_findings
    assert result.initial_state.frustration == 0
    assert result.external_state_changes[0].before.frustration == 0
    assert result.external_state_changes[0].after.frustration == 0.1
    assert result.calculated_transitions[0].before.frustration == 0.1


def test_recorded_persona_changed_fields_preserve_counterfactual_other_values(tmp_path):
    directory, *_ = source(tmp_path, turns=2, persona_willingness=0.4)
    assert not replay_behavior(directory).comparison_findings
    override = tmp_path / "rules.yaml"
    alternate = rule()
    alternate["effects"] = {"frustration_delta": 0.1}
    override.write_text(yaml.safe_dump({"rules": [alternate]}))
    result = replay_behavior(directory, rules=override)
    assert result.external_state_policy == "recorded_changed_fields"
    assert result.calculated_transitions[1].before.frustration == 0.1
    assert result.calculated_transitions[1].before.willingness_to_continue == 0.4
    assert result.calculated_transitions[1].after.frustration == 0.2


def test_terminal_recorded_persona_abandonment_is_explicit_external_evidence(tmp_path):
    directory, events, config, metadata = source(tmp_path, turns=1)
    before = events[-1]["data"]["after"]
    after = {**before, "abandoned": True, "willingness_to_continue": 0}
    events.append(
        {
            "id": "persona-terminal",
            "sequence": len(events),
            "timestamp": NOW,
            "type": "persona_state_changed",
            "data": {
                "trigger_ids": [events[-1]["id"]],
                "before": before,
                "after": after,
                "applied_rule_ids": [],
            },
        }
    )
    publish(directory, events, config, metadata)
    result = replay_behavior(directory)
    assert not result.comparison_findings
    assert result.external_state_changes[-1].after.abandoned
    assert result.external_state_changes[-1].transition_index == 1


def test_counterfactual_does_not_accept_corrupted_original_counters(tmp_path):
    directory, events, config, metadata = source(tmp_path)
    next(item for item in events if item["type"] == "behavioral_state_changed")["data"][
        "occurrence"
    ] = 9
    publish(directory, events, config, metadata)
    override = tmp_path / "rules.yaml"
    override.write_text(yaml.safe_dump({"rules": []}))
    with pytest.raises(BehaviorReplayError, match="occurrence count"):
        replay_behavior(directory, rules=override)


def test_fabricated_completed_boundary_cannot_reuse_previous_response(tmp_path):
    directory, events, config, metadata = source(tmp_path)
    boundaries = [item for item in events if item["type"] == "behavior_observation_recorded"]
    boundaries[2]["data"]["message_ids"] = boundaries[1]["data"]["message_ids"]
    boundaries[2]["data"]["evidence_event_ids"] = boundaries[1]["data"]["evidence_event_ids"]
    publish(directory, events, config, metadata)
    with pytest.raises(BehaviorReplayError, match="unrelated response evidence"):
        replay_behavior(directory)


def test_programmatic_output_never_clobbers_dangling_symlink(tmp_path):
    directory, *_ = source(tmp_path)
    output = tmp_path / "output.json"
    output.symlink_to(tmp_path / "missing-target")
    with pytest.raises(BehaviorReplayError) as failure:
        write_behavior_replay(replay_behavior(directory), output)
    assert failure.value.kind == "output_exists"
    assert output.is_symlink() and not (tmp_path / "missing-target").exists()


def test_disabled_detectors_preserve_explicit_no_signal_boundaries(tmp_path):
    directory, events, config, metadata = source(tmp_path, rules=[])
    config["behavior_detectors"] = []
    config["effective_config"]["behavior"]["detectors"] = []
    metadata["metadata"]["behavior_detectors"] = []
    events = [
        event
        for event in events
        if event["type"] not in {"behavioral_signal_recorded", "behavioral_state_changed"}
    ]
    for index, event in enumerate(events):
        event["sequence"] = index
        if event["type"] == "behavior_observation_recorded":
            event["data"]["detector_ids"] = []
            event["data"]["signal_count"] = 0
    publish(directory, events, config, metadata)
    result = replay_behavior(directory)
    assert result.completeness == "complete" and not result.comparison_findings
    assert not result.calculated_transitions
    assert len(result.external_state_changes) == 3


def test_remaining_signals_in_abandonment_batch_are_terminal_frozen(tmp_path):
    directory, events, config, metadata = source(tmp_path, turns=1, rules=[rule(abandon=True)])
    boundary = next(
        event
        for event in events
        if event["type"] == "behavior_observation_recorded"
        and event["data"]["kind"] == "customer_turn_completed"
    )
    boundary["data"]["signal_count"] = 2
    first_signal = next(event for event in events if event["type"] == "behavioral_signal_recorded")
    first_transition = events[-1]
    second_signal = json.loads(json.dumps(first_signal))
    second_signal["id"] = "second-signal"
    second_signal["data"]["signal"]["deduplication_key"] = "second-key"
    events.insert(events.index(first_transition), second_signal)
    events.append(
        {
            "id": "second-transition",
            "timestamp": NOW,
            "type": "behavioral_state_changed",
            "data": {
                **first_transition["data"],
                "signal_event_id": second_signal["id"],
                "occurrence": 2,
                "before": first_transition["data"]["after"],
                "after": first_transition["data"]["after"],
                "applied_rule_ids": [],
            },
        }
    )
    for index, event in enumerate(events):
        event["sequence"] = index
    publish(directory, events, config, metadata)
    result = replay_behavior(directory)
    assert not result.comparison_findings and len(result.calculated_transitions) == 2
    assert result.calculated_transitions[0].after.abandoned
    assert result.calculated_transitions[1].before == result.calculated_transitions[1].after
    assert not result.calculated_transitions[1].applied_rule_ids
    override = tmp_path / "rules.yaml"
    override.write_text(yaml.safe_dump({"rules": [rule(abandon=True)]}))
    assert len(replay_behavior(directory, rules=override).calculated_transitions) == 2
    # A transition from an otherwise acknowledged final batch is not a complete
    # batch: dropping the second transition excludes both signals.
    publish(directory, events[:-1], config, metadata)
    prefix = replay_behavior(directory)
    assert prefix.completeness == "prefix" and not prefix.calculated_transitions


def test_persisted_custom_payload_schema_is_validated_offline(tmp_path):
    directory, events, config, metadata = source(tmp_path, turns=1)
    schema = {
        "type": "object",
        "required": ["classification"],
        "properties": {"classification": {"$ref": "#/$defs/Classification"}},
        "$defs": {"Classification": {"type": "string", "enum": ["observed"]}},
        "additionalProperties": False,
    }
    configured_metadata = config["behavior_detectors"][0]
    configured_metadata["signal_schemas"]["customer_turn_completed"] = schema
    configured_metadata["detector_type"] = "custom_classifier"
    config["effective_config"]["behavior"]["detectors"][0]["type"] = "custom_classifier"
    metadata["metadata"]["behavior_detectors"] = [configured_metadata]
    signal = next(event for event in events if event["type"] == "behavioral_signal_recorded")[
        "data"
    ]["signal"]
    signal["detector_type"] = "custom_classifier"
    signal["payload"] = {"classification": "observed"}
    publish(directory, events, config, metadata)
    assert not replay_behavior(directory).comparison_findings
    signal["payload"] = {"classification": "invalid"}
    publish(directory, events, config, metadata)
    with pytest.raises(BehaviorReplayError, match="advertised schema"):
        replay_behavior(directory)


@pytest.mark.parametrize(
    "reference",
    [
        {"$ref": "https://untrusted.example/schema.json"},
        {"$dynamicRef": "https://untrusted.example/schema.json"},
        {"$ref": "file:///tmp/untrusted-replay-schema.json"},
        {"$ref": "../untrusted-replay-schema.json"},
    ],
)
def test_payload_schema_external_references_cannot_access_network_or_files(
    tmp_path, monkeypatch, reference
):
    directory, events, config, metadata = source(tmp_path, turns=1)
    schema = reference
    config["behavior_detectors"][0]["signal_schemas"]["customer_turn_completed"] = schema
    metadata["metadata"]["behavior_detectors"] = config["behavior_detectors"]

    def forbidden(*args, **kwargs):
        raise AssertionError("offline replay attempted a network request")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    publish(directory, events, config, metadata)
    permitted = {directory / name for name in ("trace.jsonl", "config.yaml", "metadata.json")}
    original_open = Path.open
    opened = []

    def guarded_open(path, *args, **kwargs):
        opened.append(path)
        assert path in permitted, "offline replay attempted arbitrary file access"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(BehaviorReplayError, match="advertised schema"):
        replay_behavior(directory)
    assert set(opened) == permitted


def test_missing_final_no_match_boundary_is_prefix_not_false_complete(tmp_path):
    directory, events, config, metadata = source(tmp_path, rules=[])
    last_boundary = [item for item in events if item["type"] == "behavior_observation_recorded"][-1]
    events = events[: events.index(last_boundary)]
    publish(directory, events, config, metadata)
    result = replay_behavior(directory)
    assert result.completeness == "prefix"
    assert "lacks a behavioral observation" in result.prefix_reason
    assert len(result.calculated_transitions) == 2


def test_persisted_payload_uuid_format_is_checked_without_custom_class(tmp_path):
    directory, events, config, metadata = source(tmp_path, turns=1)
    schema = {
        "type": "object",
        "required": ["identity"],
        "additionalProperties": False,
        "properties": {"identity": {"type": "string", "format": "uuid"}},
    }
    configured_metadata = config["behavior_detectors"][0]
    configured_metadata["detector_type"] = "custom_classifier"
    configured_metadata["signal_schemas"]["customer_turn_completed"] = schema
    config["effective_config"]["behavior"]["detectors"][0]["type"] = "custom_classifier"
    metadata["metadata"]["behavior_detectors"] = [configured_metadata]
    signal = next(event for event in events if event["type"] == "behavioral_signal_recorded")[
        "data"
    ]["signal"]
    signal["detector_type"] = "custom_classifier"
    signal["payload"] = {"identity": str(uuid4())}
    publish(directory, events, config, metadata)
    assert not replay_behavior(directory).comparison_findings
    signal["payload"] = {"identity": "not-a-uuid"}
    publish(directory, events, config, metadata)
    with pytest.raises(BehaviorReplayError, match="advertised schema"):
        replay_behavior(directory)

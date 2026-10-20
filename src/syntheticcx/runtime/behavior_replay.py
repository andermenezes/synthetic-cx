"""Read-only deterministic replay of committed behavioral evidence.

This module never imports application composition, detector registries, browser
adapters, or model providers. Redacted project snapshots are evidence, not runnable
configuration: only the behavioral state, rules and detector selection are read.
"""

from __future__ import annotations

import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Literal, NoReturn

import yaml
from jsonschema import Draft202012Validator
from pydantic import Field, TypeAdapter, ValidationError, model_validator
from referencing import Registry
from referencing.exceptions import NoSuchResource

from syntheticcx.config.models import BehaviorConfig, BehaviorRule, StrictModel
from syntheticcx.domain.artifacts import ConfigArtifact, MetadataArtifact
from syntheticcx.domain.behavior import (
    BehaviorReplayExternalStateChange,
    BehaviorReplayFinding,
    BehaviorReplayOccurrenceFinding,
    BehaviorReplayResult,
    BehaviorReplayRulesFinding,
    BehaviorReplaySourceHashes,
    BehaviorReplayStateFinding,
    BehaviorReplayTransition,
)
from syntheticcx.domain.events import (
    CommittedBehaviorBatch,
    PersonaStateChangedEvent,
    TypedTraceEvent,
    committed_behavior_batches,
)
from syntheticcx.domain.models import BehavioralDetectorMetadata, PersonaState
from syntheticcx.errors import ConfigurationError
from syntheticcx.runtime.behavior import apply_behavior


class BehaviorReplayError(ConfigurationError):
    """Safe diagnostic distinguishing unsupported evidence from invalid evidence."""

    kind: str

    def __init__(self, kind: str, message: str) -> None:
        super().__init__("replay behavior", message)
        object.__setattr__(self, "kind", kind)


class BehaviorReplayRules(StrictModel):
    """Rules-only counterfactual input; cannot replace the source observations."""

    rules: list[BehaviorRule] = Field(max_length=32)

    @model_validator(mode="after")
    def unique_rules(self) -> BehaviorReplayRules:
        if len({rule.id for rule in self.rules}) != len(self.rules):
            raise ValueError("rule IDs must be unique")
        return self


def _read_source(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        raise BehaviorReplayError(
            "invalid_source", "required source artifact is unreadable"
        ) from None


def _unique_json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON member")
        result[key] = value
    return result


def _json(source: bytes) -> object:
    return json.loads(source, object_pairs_hook=_unique_json_pairs)


def _load_rules(path: Path, advertised: set[str]) -> list[BehaviorRule]:
    try:
        source = path.read_bytes()
        document = BehaviorReplayRules.model_validate(yaml.safe_load(source))
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError, ValueError):
        raise BehaviorReplayError(
            "invalid_rules", "rules must be a rules-only mapping with valid unique rules"
        ) from None
    if any(rule.when.signal not in advertised for rule in document.rules):
        raise BehaviorReplayError("invalid_rules", "rule references an unrecorded detector signal")
    rules: list[BehaviorRule] = document.rules
    return rules


def _trace(source: bytes) -> tuple[list[TypedTraceEvent], str | None]:
    events: list[TypedTraceEvent] = []
    ids: set[str] = set()
    adapter = TypeAdapter(TypedTraceEvent)
    records = source.splitlines(keepends=True)
    tail_reason = None
    for index, record in enumerate(records):
        # A journal acknowledgement requires its complete newline-terminated record.
        if not record.endswith(b"\n"):
            if index != len(records) - 1:
                raise BehaviorReplayError("invalid_source", "trace contains an unterminated record")
            tail_reason = "Unterminated final trace record; only committed batches were replayed."
            break
        try:
            event = adapter.validate_python(_json(record))
        except (ValueError, UnicodeError, ValidationError):
            raise BehaviorReplayError(
                "invalid_source", "trace contains a malformed committed event"
            ) from None
        if event.sequence != len(events) or event.id in ids:
            raise BehaviorReplayError("invalid_source", "trace event order or identity is invalid")
        ids.add(event.id)
        events.append(event)
    return events, tail_reason


def _metadata(
    artifact: ConfigArtifact, metadata: MetadataArtifact
) -> tuple[tuple[BehavioralDetectorMetadata, ...], list[BehaviorRule], PersonaState]:
    effective = artifact.effective_config
    if (
        "behavior_detectors" not in artifact.model_fields_set
        or "behavior_detectors" not in metadata.metadata.model_fields_set
    ):
        raise BehaviorReplayError(
            "unsupported_source",
            "legacy run lacks behavioral detector evidence; "
            "record a new run with behavioral journaling",
        )
    if metadata.metadata.configuration_version != "1":
        raise BehaviorReplayError(
            "unsupported_source", "source configuration version is unsupported"
        )
    try:
        persona = effective["persona"]
        if not isinstance(persona, dict):
            raise ValueError("persona snapshot missing")
        initial = PersonaState.model_validate(persona["initial_state"])
        rules = TypeAdapter(list[BehaviorRule]).validate_python(persona["behavioral_rules"])
        selection = BehaviorConfig.model_validate(effective.get("behavior", {}))
    except (KeyError, ValueError, ValidationError):
        raise BehaviorReplayError(
            "invalid_source", "behavioral configuration snapshot is invalid"
        ) from None
    detector_metadata = artifact.behavior_detectors
    if detector_metadata != metadata.metadata.behavior_detectors:
        raise BehaviorReplayError("invalid_source", "detector metadata snapshots disagree")
    configured = [(item.id, item.type) for item in selection.detectors]
    recorded = [(item.detector_id, item.detector_type) for item in detector_metadata]
    names = [name for item in detector_metadata for name in item.output_signals]
    if configured != recorded or len(names) != len(set(names)):
        raise BehaviorReplayError(
            "invalid_source", "detector identities or signal ownership are invalid"
        )
    if len(rules) > 32 or len({rule.id for rule in rules}) != len(rules):
        raise BehaviorReplayError("invalid_source", "recorded behavioral rule IDs are invalid")
    if any(rule.when.signal not in names for rule in rules):
        raise BehaviorReplayError("invalid_source", "recorded rule references an undeclared signal")
    return detector_metadata, rules, initial


def _deny_schema_retrieval(uri: str) -> NoReturn:
    """Offline source evidence cannot fetch a remote JSON Schema reference."""
    raise NoSuchResource(ref=uri)


def _validate_batches(
    batches: tuple[CommittedBehaviorBatch, ...],
    events: list[TypedTraceEvent],
    detectors: tuple[BehavioralDetectorMetadata, ...],
    rules: list[BehaviorRule],
    initial: PersonaState,
) -> None:
    """Validate recorded continuity separately from deterministic engine comparison."""
    by_id = {item.detector_id: item for item in detectors}
    validators: dict[tuple[str, str], Draft202012Validator] = {}
    try:
        for detector in detectors:
            for signal_name, schema in detector.signal_schemas.items():
                Draft202012Validator.check_schema(schema)
                validators[(detector.detector_id, signal_name)] = Draft202012Validator(
                    schema,
                    registry=Registry(retrieve=_deny_schema_retrieval),
                    format_checker=Draft202012Validator.FORMAT_CHECKER,
                )
    except Exception:
        raise BehaviorReplayError(
            "invalid_source", "advertised detector payload schema is invalid"
        ) from None
    order = {item.detector_id: index for index, item in enumerate(detectors)}
    rule_ids = {item.id for item in rules}
    observations = {batch.observation.id: batch for batch in batches}
    transitions = {event.id: event for batch in batches for event in batch.transitions}
    source_events = {event.id: event for event in events}
    message_events = {
        event.data.message.id: event for event in events if event.type == "message_observed"
    }
    current = initial
    event_ids: set[str] = set()
    message_ids: set[str] = set()
    customer_turns = 0
    greeted = False
    previous_turn = 0
    last_customer_sequence = -1
    previous_boundary_sequence = -1
    conversation_start_ids: list[str] = []
    for event in events:
        if event.type == "behavior_observation_recorded" and event.id not in observations:
            break  # The whole interrupted final batch is outside the verified prefix.
        if event.type == "message_observed":
            if event.data.message.sequence != len(message_ids):
                raise BehaviorReplayError(
                    "invalid_source", "conversation message sequence is invalid"
                )
            message_ids.add(event.data.message.id)
        if event.type == "message_observed" and event.data.message.actor == "customer":
            customer_turns += 1
            last_customer_sequence = event.sequence
            if current.abandoned:
                raise BehaviorReplayError(
                    "invalid_source", "customer dispatch follows terminal abandonment"
                )
        elif event.type == "conversation_started":
            conversation_start_ids.append(event.id)
        elif event.type == "repeated_information":
            fact = event.data
            original = message_events.get(fact.original_message_id)
            request = message_events.get(fact.request_message_id)
            if (
                original is None
                or request is None
                or original.id not in event_ids
                or request.id not in event_ids
                or original.data.message.actor != "customer"
                or request.data.message.actor != "assistant"
                or original.sequence >= request.sequence
            ):
                raise BehaviorReplayError(
                    "invalid_source", "formal repetition references are invalid"
                )
        elif event.type == "persona_state_changed":
            change = event.data
            if change.before != current or any(
                ref not in event_ids and ref not in message_ids for ref in change.trigger_ids
            ):
                raise BehaviorReplayError(
                    "invalid_source", "recorded persona state continuity is invalid"
                )
            if change.applied_rule_ids:
                raise BehaviorReplayError(
                    "unsupported_source", "legacy behavioral transition evidence is unsupported"
                )
            if change.after.turn_number != customer_turns:
                raise BehaviorReplayError(
                    "invalid_source", "persona turn number disagrees with committed actions"
                )
            if change.before.abandoned and change.after != change.before:
                raise BehaviorReplayError("invalid_source", "terminal persona state changed")
            current = change.after
        elif event.id in observations:
            batch = observations[event.id]
            boundary = batch.observation.data
            if boundary.state != current or boundary.state.turn_number != boundary.turn_count:
                raise BehaviorReplayError(
                    "invalid_source", "recorded observation state is discontinuous"
                )
            if boundary.turn_count != customer_turns:
                raise BehaviorReplayError(
                    "invalid_source", "observation turn count disagrees with committed actions"
                )
            if boundary.kind == "greeting":
                if greeted or previous_turn or customer_turns:
                    raise BehaviorReplayError(
                        "invalid_source", "greeting observation is out of order"
                    )
                if len(conversation_start_ids) != 1 or (
                    conversation_start_ids[0] not in boundary.evidence_event_ids
                ):
                    raise BehaviorReplayError(
                        "invalid_source", "greeting lacks conversation-start evidence"
                    )
                greeted = True
            elif boundary.kind == "assistant_update":
                if not greeted or boundary.turn_count != previous_turn:
                    raise BehaviorReplayError(
                        "invalid_source", "assistant-update observation is out of order"
                    )
            elif not greeted or boundary.turn_count != previous_turn + 1:
                raise BehaviorReplayError(
                    "invalid_source", "completed-turn observation is out of order"
                )
            for message_id in boundary.message_ids:
                response_event = message_events[message_id]
                if (
                    response_event.id not in boundary.evidence_event_ids
                    or response_event.sequence
                    <= max(previous_boundary_sequence, last_customer_sequence)
                ):
                    raise BehaviorReplayError(
                        "invalid_source", "observation reuses unrelated response evidence"
                    )
            if any(
                source_events[ref].type
                not in {"message_observed", "conversation_started", "repeated_information"}
                for ref in boundary.evidence_event_ids
            ):
                raise BehaviorReplayError(
                    "invalid_source", "observation evidence does not describe a response"
                )
            previous_boundary_sequence = event.sequence
            previous_turn = boundary.turn_count
            if boundary.detector_ids != [item.detector_id for item in detectors]:
                raise BehaviorReplayError(
                    "invalid_source", "observation detector order differs from configuration"
                )
            counts: Counter[str] = Counter()
            previous_detector = -1
            for signal_event in batch.signals:
                signal = signal_event.data.signal
                declared = by_id.get(signal.detector_id)
                if declared is None or (
                    signal.detector_type != declared.detector_type
                    or signal.detector_version != declared.detector_version
                    or signal.signal not in declared.output_signals
                ):
                    raise BehaviorReplayError(
                        "invalid_source", "signal detector identity is not advertised"
                    )
                try:
                    validators[(signal.detector_id, signal.signal)].validate(signal.payload)
                except Exception:
                    raise BehaviorReplayError(
                        "invalid_source",
                        "recorded signal payload does not validate against its advertised schema",
                    ) from None
                if (
                    signal.detector_type == "customer_turn_completed"
                    and signal.detector_version == "1"
                ):
                    if (
                        signal.signal != "customer_turn_completed"
                        or signal.source_event_ids != tuple(boundary.evidence_event_ids)
                        or signal.source_message_ids != tuple(boundary.message_ids)
                    ):
                        raise BehaviorReplayError(
                            "invalid_source", "completed-turn signal does not cite its response"
                        )
                if (
                    signal.detector_type == "repeated_information"
                    and signal.detector_version == "1"
                ):
                    if len(signal.source_event_ids) != 1:
                        raise BehaviorReplayError(
                            "invalid_source", "repetition signal must cite its committed fact"
                        )
                    fact_event = source_events[signal.source_event_ids[0]]
                    if fact_event.type != "repeated_information" or (
                        signal.signal != "repeated_information"
                        or signal.source_message_ids
                        != (fact_event.data.original_message_id, fact_event.data.request_message_id)
                    ):
                        raise BehaviorReplayError(
                            "invalid_source", "repetition signal does not cite its committed fact"
                        )
                position = order[signal.detector_id]
                if position < previous_detector:
                    raise BehaviorReplayError(
                        "invalid_source", "signal detector ordering is invalid"
                    )
                previous_detector = position
                counts[signal.detector_id] += 1
                if counts[signal.detector_id] > 32:
                    raise BehaviorReplayError(
                        "invalid_source", "detector signal batch exceeds its bound"
                    )
                if (
                    boundary.kind != "customer_turn_completed"
                    and signal.signal == "customer_turn_completed"
                ):
                    raise BehaviorReplayError(
                        "invalid_source",
                        "supplemental observations cannot complete a customer turn",
                    )
            if any(
                not set(change.data.applied_rule_ids).issubset(rule_ids)
                for change in batch.transitions
            ):
                raise BehaviorReplayError(
                    "invalid_source", "transition references an undeclared rule"
                )
        elif event.id in transitions:
            change = transitions[event.id].data
            if change.before != current:
                raise BehaviorReplayError(
                    "invalid_source", "recorded behavioral state continuity is invalid"
                )
            if change.before.turn_number != change.after.turn_number or (
                change.before.abandoned
                and (change.before != change.after or change.applied_rule_ids)
            ):
                raise BehaviorReplayError(
                    "invalid_source", "recorded transition violates terminal or turn state"
                )
            current = change.after
        event_ids.add(event.id)


def replay_behavior(run_directory: Path, *, rules: Path | None = None) -> BehaviorReplayResult:
    """Verify source behavior or calculate rules-only counterfactual behavior.

    Invalid/unsupported source raises a safe BehaviorReplayError. Valid engine
    mismatches are findings in the returned result. A prefix is never a full-run
    verification, even when every available calculated transition matches.
    """
    directory = Path(run_directory)
    config_bytes = _read_source(directory / "config.yaml")
    trace_bytes = _read_source(directory / "trace.jsonl")
    metadata_bytes = _read_source(directory / "metadata.json")
    try:
        artifact = ConfigArtifact.model_validate(yaml.safe_load(config_bytes))
        metadata = MetadataArtifact.model_validate(_json(metadata_bytes))
    except (ValueError, UnicodeError, yaml.YAMLError, ValidationError):
        raise BehaviorReplayError(
            "invalid_source", "source metadata or configuration artifact is malformed"
        ) from None
    detectors, recorded_rules, initial = _metadata(artifact, metadata)
    events, tail_reason = _trace(trace_bytes)
    try:
        batches, batch_reason = committed_behavior_batches(events)
    except ValueError:
        raise BehaviorReplayError(
            "invalid_source", "behavioral batch references or ordering are invalid"
        ) from None
    _validate_batches(batches, events, detectors, recorded_rules, initial)
    if (
        not batches
        and metadata.lifecycle_status == "finalized"
        and not batch_reason
        and not tail_reason
    ):
        raise BehaviorReplayError(
            "unsupported_source",
            "source has no committed behavioral observations; record a new run",
        )
    advertised = {name for item in detectors for name in item.output_signals}
    effective_rules = recorded_rules if rules is None else _load_rules(Path(rules), advertised)
    if rules is not None:
        recorded_counts: Counter[str] = Counter()
        for batch in batches:
            for signal_event, transition_event in zip(
                batch.signals, batch.transitions, strict=True
            ):
                recorded_counts[signal_event.data.signal.signal] += 1
                if (
                    transition_event.data.occurrence
                    != recorded_counts[signal_event.data.signal.signal]
                ):
                    raise BehaviorReplayError(
                        "invalid_source", "recorded signal occurrence count is invalid"
                    )
    calculated: list[BehaviorReplayTransition] = []
    findings: list[BehaviorReplayFinding] = []
    external_changes: list[BehaviorReplayExternalStateChange] = []
    outside = []
    state = initial
    counts: Counter[str] = Counter()
    fired: set[str] = set()
    completed_ids = {batch.observation.id for batch in batches}
    cutoff = next(
        (
            event.sequence
            for event in events
            if event.type == "behavior_observation_recorded" and event.id not in completed_ids
        ),
        len(events),
    )
    external_events = [
        event
        for event in events
        if isinstance(event, PersonaStateChangedEvent) and event.sequence < cutoff
    ]
    external_index = 0

    def apply_external_until(sequence: int) -> None:
        nonlocal state, external_index
        while external_index < len(external_events):
            event = external_events[external_index]
            if event.sequence >= sequence:
                break
            external_index += 1
            if rules is not None and state.abandoned:
                continue
            change = event.data
            updates = {
                field: getattr(change.after, field)
                for field in ("frustration", "willingness_to_continue", "abandoned", "turn_number")
                if getattr(change.before, field) != getattr(change.after, field)
                and (not state.abandoned or field == "turn_number")
            }
            after = PersonaState.model_validate(state.model_copy(update=updates).model_dump())
            external_changes.append(
                BehaviorReplayExternalStateChange(
                    source_event_id=event.id,
                    transition_index=len(calculated),
                    before=state,
                    after=after,
                )
            )
            state = after

    for batch in batches:
        apply_external_until(batch.observation.sequence)
        boundary = batch.observation.data
        if rules is not None and state.abandoned:
            outside.append(boundary.observation_id)
            continue
        for signal_event, recorded_event in zip(batch.signals, batch.transitions, strict=True):
            signal = signal_event.data.signal
            counts[signal.signal] += 1
            result = apply_behavior(
                state,
                effective_rules,
                signal=signal.signal,
                trigger_ids=[signal_event.id],
                occurrence=counts[signal.signal],
                fired_rule_ids=fired,
            )
            transition = BehaviorReplayTransition(
                signal=signal,
                occurrence=counts[signal.signal],
                before=result.before,
                after=result.after,
                applied_rule_ids=tuple(result.applied_rule_ids),
            )
            index = len(calculated)
            calculated.append(transition)
            if rules is None:
                expected = recorded_event.data
                sides: tuple[Literal["before", "after"], ...] = ("before", "after")
                for side in sides:
                    before_or_after = getattr(transition, side)
                    recorded_state = getattr(expected, side)
                    if recorded_state != before_or_after:
                        findings.append(
                            BehaviorReplayStateFinding(
                                transition_index=index,
                                boundary=side,
                                expected=recorded_state,
                                calculated=before_or_after,
                            )
                        )
                if tuple(expected.applied_rule_ids) != transition.applied_rule_ids:
                    findings.append(
                        BehaviorReplayRulesFinding(
                            transition_index=index,
                            expected=tuple(expected.applied_rule_ids),
                            calculated=transition.applied_rule_ids,
                        )
                    )
                if expected.occurrence != transition.occurrence:
                    findings.append(
                        BehaviorReplayOccurrenceFinding(
                            transition_index=index,
                            expected=expected.occurrence,
                            calculated=transition.occurrence,
                        )
                    )
            state = transition.after
            fired.update(transition.applied_rule_ids)
    apply_external_until(cutoff)
    reasons = [reason for reason in (tail_reason, batch_reason) if reason]
    terminal = [
        event
        for event in events
        if event.type == "run_lifecycle_changed" and event.data.current == "finalized"
    ]
    if terminal:
        final = terminal[0]
        if (
            len(terminal) != 1
            or final is not events[-1]
            or (
                metadata.lifecycle_status == "finalized"
                and (
                    final.data.outcome != metadata.outcome
                    or final.data.termination_reason != metadata.termination_reason
                )
            )
        ):
            raise BehaviorReplayError(
                "invalid_source", "terminal trace evidence disagrees with source metadata"
            )
    else:
        reasons.append("Terminal trace evidence is missing; only a committed prefix was replayed.")
    covered_messages = {
        message_id for batch in batches for message_id in batch.observation.data.message_ids
    }
    if batch_reason is None and any(
        event.type == "message_observed"
        and event.data.message.actor == "assistant"
        and event.data.message.id not in covered_messages
        for event in events
    ):
        reasons.append("Committed assistant response lacks a behavioral observation boundary.")
    if metadata.lifecycle_status != "finalized":
        reasons.append(
            "Source run is not finalized; only its committed behavioral prefix was replayed."
        )
    return BehaviorReplayResult(
        source_hashes=BehaviorReplaySourceHashes(
            trace_sha256=sha256(trace_bytes).hexdigest(),
            effective_config_sha256=sha256(config_bytes).hexdigest(),
        ),
        mode="verification" if rules is None else "counterfactual",
        initial_state=initial,
        rules=tuple(effective_rules),
        calculated_transitions=tuple(calculated),
        external_state_changes=tuple(external_changes),
        completeness="prefix" if reasons else "complete",
        prefix_reason=" ".join(reasons) if reasons else None,
        comparison_findings=tuple(findings),
        outside_continuation_observation_ids=tuple(outside),
    )


def write_behavior_replay(result: BehaviorReplayResult, output: Path) -> None:
    """Exclusively create deterministic replay JSON; never overwrite a file/link."""
    encoded = (
        json.dumps(
            result.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    try:
        with Path(output).open("xb") as stream:
            stream.write(encoded)
    except FileExistsError:
        raise BehaviorReplayError(
            "output_exists", "output already exists; choose a new output path"
        ) from None
    except OSError:
        raise BehaviorReplayError("output_error", "replay output could not be written") from None

"""Pure projections of committed evidence into persisted artifact views.

These functions never read previously materialized files and never mutate the
committed event stream. Referential integrity and criterion coverage are
checked before any caller persists a projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from pydantic import TypeAdapter

from syntheticcx.config.models import Criterion
from syntheticcx.domain.artifacts import ConversationArtifact, EvaluationArtifact
from syntheticcx.domain.events import (
    EvaluationCompletedEvent,
    MessageObservedEvent,
    PersonaStateChangedEvent,
    TypedTraceEvent,
)
from syntheticcx.domain.models import (
    ArtifactRecord,
    EvaluationResult,
    Message,
    PersonaState,
    PersonaStateTransition,
    Report,
)
from syntheticcx.errors import PersistenceError
from syntheticcx.privacy import Redactor

_EVENT = TypeAdapter(TypedTraceEvent)


@dataclass(frozen=True)
class Projections:
    """Validated views derived solely from committed events."""

    conversation: ConversationArtifact | None
    evaluation: EvaluationArtifact | None
    report: Report | None


def build_projections(
    run_id: str,
    events: list[TypedTraceEvent],
    criteria: list[Criterion | tuple[str, str, bool]],
    *,
    artifact_location: str | None = None,
    artifacts: list[ArtifactRecord] | None = None,
    materialize_conversation: bool,
    materialize_evaluation: bool,
    materialize_report: bool,
) -> Projections:
    """Derive artifact views and validate references before finalization.

    Criteria are complete sanitized snapshots from the effective configuration.
    Message order follows committed event order; report metrics are derived as
    approved in the plan checkpoint.
    """
    snapshots = [_criterion_snapshot(criterion) for criterion in criteria]
    messages: list[Message] = []
    message_ids: set[str] = set()
    event_ids = {event.id for event in events}
    results: list[EvaluationResult] = []
    result_criterion_ids: list[str] = []
    turn_count = 0
    repetition_count = 0

    for event in events:
        if isinstance(event, MessageObservedEvent):
            message = event.data.message
            if message.id in message_ids:
                raise PersistenceError(
                    "materialize evidence", "committed messages contain a duplicate ID"
                )
            message_ids.add(message.id)
            messages.append(message.model_copy(deep=True))
            if message.actor == "customer":
                turn_count += 1
        elif isinstance(event, PersonaStateChangedEvent):
            if any("repeated_information" in trigger_id for trigger_id in event.data.trigger_ids):
                repetition_count += 1
        elif isinstance(event, EvaluationCompletedEvent):
            result = event.data.result.model_copy(deep=True)
            _validate_references(result, event_ids, message_ids)
            if result.criterion_id in result_criterion_ids:
                raise PersistenceError(
                    "materialize evidence", "one criterion produced multiple results"
                )
            result_criterion_ids.append(result.criterion_id)
            results.append(result)

    _validate_criterion_coverage(snapshots, result_criterion_ids)

    conversation = ConversationArtifact(messages) if materialize_conversation else None
    evaluation = (
        EvaluationArtifact(
            criteria=[criterion.model_copy(deep=True) for criterion in snapshots],
            results=results,
        )
        if materialize_evaluation
        else None
    )
    report = _build_report(
        run_id,
        events,
        results,
        turn_count,
        repetition_count,
        materialize_report,
        materialize_conversation,
        materialize_evaluation,
        snapshots,
        artifact_location or f".runs/{run_id}",
        artifacts or [],
    )
    return Projections(conversation, evaluation, report)


def _validate_references(
    result: EvaluationResult, event_ids: set[str], message_ids: set[str]
) -> None:
    unknown_events = set(result.evidence_event_ids) - event_ids
    unknown_messages = set(result.evidence_message_ids) - message_ids
    if unknown_events or unknown_messages:
        raise PersistenceError(
            "materialize evidence", "an evaluation references uncommitted evidence"
        )


def _criterion_snapshot(criterion: Criterion | tuple[str, str, bool]) -> Criterion:
    """Accept legacy compact inputs only for pure direct callers.

    Filesystem finalization always supplies complete validated Criterion values.
    """
    if isinstance(criterion, Criterion):
        return cast(Criterion, Criterion.model_validate(criterion.model_dump()))
    identifier, evaluator_type, required = criterion
    return Criterion(
        id=identifier,
        evaluator_type=evaluator_type,
        result_kind="boolean",
        evidence_policy="required",
        required=required,
    )


def _validate_criterion_coverage(
    criteria: list[Criterion], result_criterion_ids: list[str]
) -> None:
    known_ids = {criterion.id for criterion in criteria}
    required_ids = {criterion.id for criterion in criteria if criterion.required}
    unknown = sorted(set(result_criterion_ids) - known_ids)
    if unknown:
        raise PersistenceError(
            "materialize evidence", "an evaluation references an unknown criterion"
        )
    missing = sorted(required_ids - set(result_criterion_ids))
    if missing:
        raise PersistenceError(
            "materialize evidence", "a required criterion has no committed evaluation"
        )


def _build_report(
    run_id: str,
    events: list[TypedTraceEvent],
    results: list[EvaluationResult],
    turn_count: int,
    repetition_count: int,
    materialize_report: bool,
    materialize_conversation: bool,
    materialize_evaluation: bool,
    criteria: list[Criterion],
    artifact_location: str,
    artifacts: list[ArtifactRecord],
) -> Report | None:
    if not materialize_report:
        return None
    timestamps = [event.timestamp for event in events]
    duration = (max(timestamps) - min(timestamps)).total_seconds() if len(timestamps) > 1 else 0.0
    goal_criterion_ids = {
        criterion.id
        for criterion in criteria
        if criterion.evaluator_type == "semantic_goal_completion"
    }
    goal = next((result for result in results if result.criterion_id in goal_criterion_ids), None)
    omissions: list[str] = []
    if not materialize_conversation:
        omissions.append("conversation")
    if not materialize_evaluation:
        omissions.append("evaluation")
    return Report(
        run_id=run_id,
        outcome=None,
        findings=[result.model_copy(deep=True) for result in results],
        termination_reason=None,
        goal_result=goal.model_copy(deep=True) if goal is not None else None,
        turn_count=turn_count,
        duration_seconds=duration,
        repetition_count=repetition_count,
        errors=[],
        artifact_location=artifact_location,
        artifacts=[artifact.model_copy(deep=True) for artifact in artifacts],
        omissions=omissions,
    )


def committed_events_from_lines(lines: list[str]) -> list[TypedTraceEvent]:
    """Parse committed JSONL records; a malformed record is a persistence error."""
    try:
        return [_EVENT.validate_json(line) for line in lines]
    except Exception:
        raise PersistenceError("materialize evidence", "committed trace is not readable") from None


def omitted_artifact_records(
    *,
    materialize_conversation: bool,
    materialize_evaluation: bool,
    materialize_report: bool,
) -> list[ArtifactRecord]:
    """Describe disabled views explicitly; never silently omit required artifacts."""
    records: list[ArtifactRecord] = []
    if not materialize_conversation:
        records.append(
            ArtifactRecord(media_type="application/json", status="omitted", reason="disabled")
        )
    if not materialize_evaluation:
        records.append(
            ArtifactRecord(media_type="application/json", status="omitted", reason="disabled")
        )
    if not materialize_report:
        records.append(
            ArtifactRecord(media_type="application/json", status="omitted", reason="disabled")
        )
    return records


def sanitize_transition_proof(
    transition: PersonaStateTransition, redactor: Redactor
) -> PersonaState:
    """Revalidate a transition through the redaction boundary."""
    return redactor.redact_model(transition).after


__all__ = [
    "Projections",
    "build_projections",
    "committed_events_from_lines",
    "omitted_artifact_records",
]

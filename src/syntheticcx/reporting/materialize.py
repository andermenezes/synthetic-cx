"""Pure projections of committed evidence into persisted artifact views.

These functions never read previously materialized files and never mutate the
committed event stream. Referential integrity and criterion coverage are
checked before any caller persists a projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from pydantic import TypeAdapter

from syntheticcx.config.models import Criterion
from syntheticcx.domain.artifacts import ConversationArtifact, EvaluationArtifact
from syntheticcx.domain.events import (
    ArtifactRecordedEvent,
    BehavioralObservationRecordedEvent,
    EvaluationCompletedEvent,
    MessageObservedEvent,
    PersonaFeedbackRecordedEvent,
    RepeatedInformationEvent,
    RunLifecycleEvent,
    TypedTraceEvent,
    committed_behavior_batches,
    committed_errors,
)
from syntheticcx.domain.models import (
    ArtifactRecord,
    BehavioralTransitionSummary,
    ErrorDetail,
    EvaluationResult,
    Message,
    Report,
)
from syntheticcx.errors import EvaluationError, PersistenceError
from syntheticcx.evaluation.validation import validate_result

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
    criteria: list[Criterion],
    *,
    artifact_location: str | None = None,
    artifacts: list[ArtifactRecord] | None = None,
    materialize_conversation: bool,
    materialize_evaluation: bool,
    materialize_report: bool,
    allow_partial: bool = False,
) -> Projections:
    """Derive artifact views and validate references before finalization.

    Criteria are complete sanitized snapshots from the effective configuration.
    Message order follows committed event order. allow_partial permits a
    read-only diagnostic view before finalization, retaining missing required
    evaluation omissions without inventing verdicts or writing to a failed store.
    """
    snapshots = [_criterion_snapshot(criterion) for criterion in criteria]
    configured = {criterion.id: criterion for criterion in snapshots}
    if len(configured) != len(snapshots):
        raise PersistenceError("materialize evidence", "configured criteria contain duplicate IDs")
    messages: list[Message] = []
    message_ids: set[str] = set()
    prior_events: list[TypedTraceEvent] = []
    results: list[EvaluationResult] = []
    final_lifecycle: RunLifecycleEvent | None = None
    result_criterion_ids: list[str] = []
    turn_count = 0
    repetition_count = 0
    feedback = None

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
        elif isinstance(event, PersonaFeedbackRecordedEvent):
            if feedback is not None:
                raise PersistenceError("materialize evidence", "multiple feedback events")
            feedback = event.data
            if feedback.response and any(
                ref not in message_ids
                for topic in feedback.response.topics
                for ref in topic.evidence_message_ids
            ):
                raise PersistenceError(
                    "materialize evidence", "feedback references uncommitted messages"
                )
        elif isinstance(event, RepeatedInformationEvent):
            repetition_count += 1
        elif isinstance(event, EvaluationCompletedEvent):
            result = event.data.result.model_copy(deep=True)
            criterion = configured.get(result.criterion_id)
            if criterion is None:
                raise PersistenceError(
                    "materialize evidence", "an evaluation references an unknown criterion"
                )
            snapshot = event.data.criterion
            if (
                snapshot.id != criterion.id
                or snapshot.evaluator_type != criterion.evaluator_type
                or snapshot.result_kind != criterion.result_kind
                or snapshot.required != criterion.required
                or snapshot.threshold != criterion.threshold
                or snapshot.evidence_policy != criterion.evidence_policy
                or snapshot.parameters != criterion.parameters
            ):
                raise PersistenceError(
                    "materialize evidence", "evaluation criterion snapshot is inconsistent"
                )
            try:
                validate_result(result, criterion, prior_events)
            except EvaluationError:
                raise PersistenceError(
                    "materialize evidence", "an evaluation contains invalid or uncommitted evidence"
                ) from None
            if result.criterion_id in result_criterion_ids:
                raise PersistenceError(
                    "materialize evidence", "one criterion produced multiple results"
                )
            result_criterion_ids.append(result.criterion_id)
            results.append(result)
        elif isinstance(event, RunLifecycleEvent) and event.data.current == "finalized":
            if final_lifecycle is not None:
                raise PersistenceError("materialize evidence", "multiple final lifecycle events")
            final_lifecycle = event
        prior_events.append(event)

    _validate_criterion_coverage(
        snapshots,
        result_criterion_ids,
        require_complete=not allow_partial or final_lifecycle is not None,
    )

    conversation = ConversationArtifact(messages) if materialize_conversation else None
    evaluation = (
        EvaluationArtifact(
            criteria=[criterion.model_copy(deep=True) for criterion in snapshots],
            results=results,
        )
        if materialize_evaluation
        else None
    )
    reconciled_artifacts = reconcile_artifacts(events, artifacts or [])
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
        reconciled_artifacts,
        final_lifecycle,
        committed_errors(events),
        *_behavior_evidence(events, final_lifecycle is not None),
    )
    return Projections(conversation, evaluation, report)


def _behavior_evidence(
    events: list[TypedTraceEvent], finalized: bool
) -> tuple[
    int,
    int,
    list[BehavioralTransitionSummary],
    Literal["unavailable", "complete", "prefix"],
    str | None,
]:
    """Claim effects only for batches whose entire transition list committed."""
    try:
        batches, reason = committed_behavior_batches(events)
        counters: dict[str, int] = {}
        for batch in batches:
            current = batch.observation.data.state
            for transition in batch.transitions:
                data = transition.data
                occurrence = counters.get(data.signal, 0) + 1
                if data.before != current or data.occurrence != occurrence:
                    raise ValueError("behavioral state or occurrence evidence is inconsistent")
                counters[data.signal] = occurrence
                current = data.after
    except ValueError:
        raise PersistenceError(
            "materialize evidence", "committed behavioral evidence is inconsistent"
        ) from None
    if not any(isinstance(event, BehavioralObservationRecordedEvent) for event in events):
        return 0, 0, [], "unavailable", None
    if reason is None and not finalized:
        reason = "source run has no committed final lifecycle"
    transitions = [
        BehavioralTransitionSummary.model_validate(
            {
                **transition.data.model_dump(mode="json"),
                "detector_id": signal.data.signal.detector_id,
                "detector_type": signal.data.signal.detector_type,
                "detector_version": signal.data.signal.detector_version,
                "source_event_ids": list(signal.data.signal.source_event_ids),
                "source_message_ids": list(signal.data.signal.source_message_ids),
            }
        )
        for batch in batches
        for signal, transition in zip(batch.signals, batch.transitions, strict=True)
    ]
    return len(batches), len(transitions), transitions, "prefix" if reason else "complete", reason


def reconcile_artifacts(
    events: list[TypedTraceEvent], supplied: list[ArtifactRecord]
) -> list[ArtifactRecord]:
    """Make committed artifact events visible without duplicating caller projections."""
    records: list[ArtifactRecord] = []
    for record in supplied + [
        event.data for event in events if isinstance(event, ArtifactRecordedEvent)
    ]:
        if record.path is not None and any(
            existing.path == record.path and existing != record for existing in records
        ):
            raise PersistenceError("materialize evidence", "conflicting artifact records")
        if record not in records:
            records.append(record.model_copy(deep=True))
    return records


def _criterion_snapshot(criterion: Criterion) -> Criterion:
    """Detach a complete validated criterion for a persisted projection."""
    return cast(Criterion, Criterion.model_validate(criterion.model_dump()))


def _validate_criterion_coverage(
    criteria: list[Criterion],
    result_criterion_ids: list[str],
    *,
    require_complete: bool = True,
) -> None:
    known_ids = {criterion.id for criterion in criteria}
    required_ids = {criterion.id for criterion in criteria if criterion.required}
    unknown = sorted(set(result_criterion_ids) - known_ids)
    if unknown:
        raise PersistenceError(
            "materialize evidence", "an evaluation references an unknown criterion"
        )
    missing = sorted(required_ids - set(result_criterion_ids))
    if missing and require_complete:
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
    final_lifecycle: RunLifecycleEvent | None,
    failures: list[ErrorDetail],
    behavior_observation_count: int,
    behavior_signal_count: int,
    behavior_transitions: list[BehavioralTransitionSummary],
    behavior_completeness: Literal["unavailable", "complete", "prefix"],
    behavior_prefix_reason: str | None,
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
    evaluated_ids = {result.criterion_id for result in results}
    if any(criterion.required and criterion.id not in evaluated_ids for criterion in criteria):
        omissions.append("evaluation:incomplete")
    omissions.extend(
        f"screenshot:{artifact.reason or artifact.status}"
        for artifact in artifacts
        if artifact.media_type == "image/png" and artifact.status != "captured"
    )
    if behavior_prefix_reason is not None:
        omissions.append("behavior:incomplete_prefix")
    return Report(
        run_id=run_id,
        outcome=final_lifecycle.data.outcome if final_lifecycle else None,
        findings=[result.model_copy(deep=True) for result in results],
        termination_reason=final_lifecycle.data.termination_reason if final_lifecycle else None,
        goal_result=goal.model_copy(deep=True) if goal is not None else None,
        turn_count=turn_count,
        duration_seconds=duration,
        repetition_count=repetition_count,
        errors=failures,
        artifact_location=artifact_location,
        artifacts=[artifact.model_copy(deep=True) for artifact in artifacts],
        omissions=omissions,
        persona_feedback=next(
            (
                event.data.model_copy(deep=True)
                for event in events
                if isinstance(event, PersonaFeedbackRecordedEvent)
            ),
            None,
        ),
        conversation_duration_seconds=next(
            (
                event.data.conversation_duration_seconds
                for event in events
                if isinstance(event, RunLifecycleEvent)
                and event.data.conversation_duration_seconds is not None
            ),
            None,
        ),
        behavior_observation_count=behavior_observation_count,
        behavior_signal_count=behavior_signal_count,
        behavior_transitions=behavior_transitions,
        behavior_completeness=behavior_completeness,
        behavior_prefix_reason=behavior_prefix_reason,
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


__all__ = [
    "Projections",
    "build_projections",
    "committed_events_from_lines",
    "omitted_artifact_records",
]

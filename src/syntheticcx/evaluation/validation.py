"""Criterion and citation integrity shared by evaluation and trace projections."""

from __future__ import annotations

from pydantic import ValidationError

from syntheticcx.config.models import Criterion
from syntheticcx.domain.events import (
    MessageObservedEvent,
    RepeatedInformationEvent,
    TypedTraceEvent,
)
from syntheticcx.domain.models import EvaluationResult
from syntheticcx.errors import EvaluationError


def validate_result(
    result: EvaluationResult, criterion: Criterion, committed: list[TypedTraceEvent]
) -> None:
    """Reject claims not supported by the criterion and prior committed evidence.

    ``committed`` excludes the evaluation event under construction. Every message
    citation must be paired with its own committed message-observation event.
    """
    try:
        EvaluationResult.model_validate(result.model_dump(mode="json"))
    except (ValidationError, ValueError):
        raise EvaluationError("validate evaluation", "evaluation result is invalid") from None
    if (
        result.criterion_id != criterion.id
        or result.evaluator_id != criterion.evaluator_type
        or result.required != criterion.required
        or not result.explanation.strip()
        or (result.severity is not None and not result.severity.strip())
    ):
        raise EvaluationError("validate evaluation", "evaluation contract is inconsistent")

    if result.status in {"passed", "failed"}:
        if criterion.result_kind == "scored":
            if result.score is None or criterion.threshold is None:
                raise EvaluationError("validate evaluation", "scored verdict is invalid")
            if (result.score >= criterion.threshold) != (result.status == "passed"):
                raise EvaluationError(
                    "validate evaluation", "scored verdict conflicts with threshold"
                )
        elif result.score is not None:
            raise EvaluationError("validate evaluation", "boolean verdict cannot have a score")
    elif result.score is not None:
        raise EvaluationError("validate evaluation", "non-decision verdict cannot have a score")

    if criterion.evidence_policy == "required" and result.status in {"passed", "failed"}:
        if not result.evidence_event_ids and not result.evidence_message_ids:
            raise EvaluationError("validate evaluation", "required evidence is missing")

    if len(set(result.evidence_event_ids)) != len(result.evidence_event_ids) or len(
        set(result.evidence_message_ids)
    ) != len(result.evidence_message_ids):
        raise EvaluationError("validate evaluation", "duplicate evidence citation")

    events = {event.id: event for event in committed}
    if len(events) != len(committed):
        raise EvaluationError("validate evaluation", "committed event identity is invalid")
    messages = {
        event.data.message.id: event
        for event in committed
        if isinstance(event, MessageObservedEvent)
    }
    if len(messages) != sum(isinstance(event, MessageObservedEvent) for event in committed):
        raise EvaluationError("validate evaluation", "committed message identity is invalid")
    if any(identifier not in events for identifier in result.evidence_event_ids):
        raise EvaluationError("validate evaluation", "event evidence is not committed")
    for identifier in result.evidence_message_ids:
        message_event = messages.get(identifier)
        if message_event is None or message_event.id not in result.evidence_event_ids:
            raise EvaluationError("validate evaluation", "message evidence is not paired")
    for identifier in result.evidence_event_ids:
        event = events[identifier]
        if (
            isinstance(event, MessageObservedEvent)
            and event.data.message.id not in result.evidence_message_ids
        ):
            raise EvaluationError("validate evaluation", "message evidence is not paired")

    for identifier in result.evidence_event_ids:
        event = events[identifier]
        if not isinstance(event, RepeatedInformationEvent):
            continue
        original = messages.get(event.data.original_message_id)
        request = messages.get(event.data.request_message_id)
        if (
            original is None
            or request is None
            or original.data.message.actor != "customer"
            or request.data.message.actor != "assistant"
            or not original.sequence < request.sequence < event.sequence
            or original.id not in result.evidence_event_ids
            or request.id not in result.evidence_event_ids
            or original.data.message.id not in result.evidence_message_ids
            or request.data.message.id not in result.evidence_message_ids
        ):
            raise EvaluationError("validate evaluation", "repetition evidence is invalid")

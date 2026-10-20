"""Deterministic findings derived exclusively from committed conversation evidence."""

from __future__ import annotations

from syntheticcx.domain.artifacts import EvaluationContext
from syntheticcx.domain.events import (
    MessageObservedEvent,
    RepeatedInformationEvent,
    RunLifecycleEvent,
    TypedTraceEvent,
)
from syntheticcx.domain.models import EvaluationResult
from syntheticcx.errors import EvaluationError

# A journey timeout still closes the observation window for count-based checks;
# the timeout itself is flagged separately by JourneyTimeoutEvaluator.
_COMPLETE_REASONS = frozenset(
    {"goal_believed_complete", "max_turns", "abandonment", "journey_timeout"}
)


def _termination(events: list[TypedTraceEvent]) -> RunLifecycleEvent | None:
    """Only a durable conversation stop proves the observation window has ended."""
    return next(
        (
            event
            for event in reversed(events)
            if isinstance(event, RunLifecycleEvent)
            and event.data.current in {"in_progress", "incomplete", "finalized"}
            and event.data.termination_reason is not None
        ),
        None,
    )


def _complete(terminal: RunLifecycleEvent | None) -> bool:
    return terminal is not None and terminal.data.termination_reason in _COMPLETE_REASONS


def _status(failed: bool, terminal: RunLifecycleEvent | None) -> str:
    """Fail on an exceeded count even without a terminal record; absence stays unavailable."""
    return "failed" if failed else "passed" if _complete(terminal) else "unavailable"


def _require_type(context: EvaluationContext, expected: str) -> None:
    if context.criterion.evaluator_type != expected:
        raise EvaluationError("evaluate criterion", "criterion does not match evaluator")


class TurnCountEvaluator:
    """Check the number of durably observed, successfully sent customer messages."""

    async def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        _require_type(context, "turn_count")
        limit = context.criterion.parameters["max_turns"]
        submitted = [
            event
            for event in context.sanitized_trace
            if isinstance(event, MessageObservedEvent) and event.data.message.actor == "customer"
        ]
        terminal = _termination(context.sanitized_trace)
        count = len(submitted)
        status = _status(count > limit, terminal)
        return EvaluationResult(
            evaluator_id="turn_count",
            criterion_id=context.criterion.id,
            required=context.criterion.required,
            status=status,
            explanation={
                "passed": f"Observed {count} submitted customer turns; limit is {limit}.",
                "failed": f"Observed {count} submitted customer turns; limit is {limit}.",
                "unavailable": (
                    "The conversation did not finish with a complete turn-count observation."
                ),
            }[status],
            evidence_event_ids=[event.id for event in submitted]
            + ([terminal.id] if terminal is not None and _complete(terminal) else []),
            evidence_message_ids=[event.data.message.id for event in submitted],
            metadata={"turn_count": count, "max_turns": limit},
        )


class JourneyTimeoutEvaluator:
    """Distinguish a documented journey deadline from absent terminal evidence."""

    async def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        _require_type(context, "journey_timeout")
        terminal = _termination(context.sanitized_trace)
        reason = terminal.data.termination_reason if terminal is not None else None
        status = _status(reason == "journey_timeout", terminal)
        return EvaluationResult(
            evaluator_id="journey_timeout",
            criterion_id=context.criterion.id,
            required=context.criterion.required,
            status=status,
            explanation={
                "passed": "The completed journey did not exceed its deadline.",
                "failed": "The journey ended at its configured timeout.",
                "unavailable": "A complete journey termination was not observed.",
            }[status],
            evidence_event_ids=[terminal.id] if terminal is not None else [],
            metadata={"termination_reason": reason},
        )


class RepeatedInformationEvaluator:
    """Count rule-detected repeated requests with their original and later messages."""

    async def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        _require_type(context, "repeated_information")
        limit = context.criterion.parameters["max_repeats"]
        repetitions = [
            event
            for event in context.sanitized_trace
            if isinstance(event, RepeatedInformationEvent)
        ]
        messages = {
            event.data.message.id: event
            for event in context.sanitized_trace
            if isinstance(event, MessageObservedEvent)
        }
        references: list[str] = []
        evidence_events: list[str] = []
        for repetition in repetitions:
            original = messages.get(repetition.data.original_message_id)
            request = messages.get(repetition.data.request_message_id)
            if (
                original is None
                or request is None
                or original.data.message.actor != "customer"
                or request.data.message.actor != "assistant"
                or not original.sequence < request.sequence < repetition.sequence
            ):
                raise EvaluationError("evaluate repetition", "repetition evidence is invalid")
            references.extend((original.data.message.id, request.data.message.id))
            evidence_events.extend((original.id, request.id, repetition.id))
        terminal = _termination(context.sanitized_trace)
        count = len(repetitions)
        status = _status(count > limit, terminal)
        if _complete(terminal):
            if terminal is not None:
                evidence_events.append(terminal.id)
        return EvaluationResult(
            evaluator_id="repeated_information",
            criterion_id=context.criterion.id,
            required=context.criterion.required,
            status=status,
            explanation={
                "passed": f"Detected {count} repeated information requests; allowance is {limit}.",
                "failed": f"Detected {count} repeated information requests; allowance is {limit}.",
                "unavailable": (
                    "The conversation ended before repetition could be assessed completely."
                ),
            }[status],
            evidence_event_ids=list(dict.fromkeys(evidence_events)),
            evidence_message_ids=list(dict.fromkeys(references)),
            severity="warning" if status == "failed" else None,
            metadata={"repetition_count": count, "max_repeats": limit, "detection_method": "rules"},
        )

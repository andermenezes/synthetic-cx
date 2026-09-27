"""Deterministic evaluators use committed evidence, never inferred conversation state."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from syntheticcx.config.loader import load_config
from syntheticcx.config.models import Criterion
from syntheticcx.domain.artifacts import EvaluationContext
from syntheticcx.domain.events import (
    MessageObservedEvent,
    RepeatedInformationEvent,
    RepeatedInformationObserved,
    RunLifecycleChange,
    RunLifecycleEvent,
    TypedTraceEvent,
)
from syntheticcx.domain.models import Message, Scenario
from syntheticcx.errors import EvaluationError
from syntheticcx.evaluation.deterministic import (
    JourneyTimeoutEvaluator,
    RepeatedInformationEvaluator,
    TurnCountEvaluator,
)

NOW = datetime(2026, 9, 27, tzinfo=UTC)
SCENARIO = Scenario.model_validate(
    load_config(Path("examples/telecom/scenario.yaml")).scenario.model_dump()
)


def message(index: int, actor: str, text: str) -> MessageObservedEvent:
    return MessageObservedEvent.model_validate(
        {
            "id": f"event-{index}",
            "sequence": index,
            "timestamp": NOW,
            "type": "message_observed",
            "data": {
                "message": Message(
                    id=f"message-{index}",
                    sequence=index,
                    actor=actor,
                    content=text,
                    observed_at=NOW,
                ).model_dump()
            },
        }
    )


def terminal(index: int, reason: str) -> RunLifecycleEvent:
    return RunLifecycleEvent(
        id=f"event-{index}",
        sequence=index,
        timestamp=NOW,
        type="run_lifecycle_changed",
        data=RunLifecycleChange(current="incomplete", termination_reason=reason),
    )


def repetition(index: int, original: str, request: str) -> RepeatedInformationEvent:
    return RepeatedInformationEvent(
        id=f"event-{index}",
        sequence=index,
        timestamp=NOW,
        type="repeated_information",
        data=RepeatedInformationObserved(
            information_id="account_number",
            original_message_id=original,
            request_message_id=request,
        ),
    )


def context(
    kind: str, events: list[TypedTraceEvent], *, parameters: dict[str, object] | None = None
) -> EvaluationContext:
    return EvaluationContext(
        criterion=Criterion.model_validate(
            {
                "id": "finding",
                "evaluator_type": kind,
                "result_kind": "boolean",
                "evidence_policy": "required",
                "parameters": parameters or {},
            }
        ),
        scenario=SCENARIO,
        sanitized_trace=events,
    )


def evaluate(evaluator, evaluation: EvaluationContext):
    return asyncio.run(evaluator.evaluate(evaluation))


def test_turn_count_excludes_greeting_and_cites_only_submitted_customers() -> None:
    events = [
        message(0, "assistant", "Welcome!"),
        message(1, "customer", "Why did my bill increase?"),
        message(2, "assistant", "A charge was added."),
        terminal(3, "goal_believed_complete"),
    ]
    result = evaluate(
        TurnCountEvaluator(), context("turn_count", events, parameters={"max_turns": 1})
    )
    assert result.status == "passed"
    assert result.metadata["turn_count"] == 1
    assert result.evidence_message_ids == ["message-1"]
    assert result.evidence_event_ids == ["event-1", "event-3"]


def test_turn_count_exceeded_even_with_partial_trace_but_no_pass_without_terminal() -> None:
    events = [message(0, "customer", "First"), message(1, "customer", "Second")]
    evaluator = TurnCountEvaluator()
    assert (
        evaluate(evaluator, context("turn_count", events, parameters={"max_turns": 1})).status
        == "failed"
    )
    assert (
        evaluate(evaluator, context("turn_count", events, parameters={"max_turns": 2})).status
        == "unavailable"
    )


@pytest.mark.parametrize(
    ("reason", "status"),
    [
        ("goal_believed_complete", "passed"),
        ("max_turns", "passed"),
        ("journey_timeout", "failed"),
        ("channel_error", "unavailable"),
        ("cancelled", "unavailable"),
    ],
)
def test_timeout_requires_a_committed_meaningful_termination(reason: str, status: str) -> None:
    result = evaluate(JourneyTimeoutEvaluator(), context("journey_timeout", [terminal(0, reason)]))
    assert result.status == status
    assert result.evidence_event_ids == ["event-0"]


def test_missing_termination_cannot_prove_no_timeout() -> None:
    result = evaluate(JourneyTimeoutEvaluator(), context("journey_timeout", []))
    assert result.status == "unavailable"
    assert result.evidence_event_ids == []


def test_repetition_cites_original_customer_and_later_assistant_request() -> None:
    events = [
        message(0, "customer", "My account number is 123."),
        message(1, "assistant", "What is your account number?"),
        repetition(2, "message-0", "message-1"),
        terminal(3, "abandonment"),
    ]
    result = evaluate(RepeatedInformationEvaluator(), context("repeated_information", events))
    assert result.status == "failed"
    assert result.evidence_event_ids == ["event-0", "event-1", "event-2", "event-3"]
    assert result.evidence_message_ids == ["message-0", "message-1"]
    assert result.metadata["detection_method"] == "rules"
    allowed = evaluate(
        RepeatedInformationEvaluator(),
        context("repeated_information", events, parameters={"max_repeats": 1}),
    )
    assert allowed.status == "passed"


def test_zero_repetitions_need_a_complete_observation_window() -> None:
    evaluator = RepeatedInformationEvaluator()
    assert evaluate(evaluator, context("repeated_information", [])).status == "unavailable"
    result = evaluate(evaluator, context("repeated_information", [terminal(0, "max_turns")]))
    assert result.status == "passed"
    assert result.evidence_event_ids == ["event-0"]


def test_in_progress_conversation_stop_is_complete_evidence_for_evaluation() -> None:
    stop = terminal(0, "goal_believed_complete").model_copy(
        update={
            "data": RunLifecycleChange(
                current="in_progress", termination_reason="goal_believed_complete"
            )
        }
    )
    result = evaluate(RepeatedInformationEvaluator(), context("repeated_information", [stop]))
    assert result.status == "passed"
    assert result.evidence_event_ids == ["event-0"]


@pytest.mark.parametrize(
    ("original", "requested_message"),
    [("absent", "message-1"), ("message-1", "message-0")],
)
def test_invalid_repetition_links_raise_safe_evaluation_error(
    original: str, requested_message: str
) -> None:
    events = [
        message(0, "customer", "Account number"),
        message(1, "assistant", "Account number?"),
        repetition(2, original, requested_message),
    ]
    with pytest.raises(EvaluationError, match="repetition evidence is invalid"):
        evaluate(RepeatedInformationEvaluator(), context("repeated_information", events))


def test_evaluator_rejects_the_wrong_criterion() -> None:
    with pytest.raises(EvaluationError, match="criterion does not match"):
        evaluate(TurnCountEvaluator(), context("journey_timeout", []))

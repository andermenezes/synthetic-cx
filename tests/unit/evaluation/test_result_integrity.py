"""Findings are tied to their configured criterion and prior committed evidence."""

from datetime import UTC, datetime

import pytest

from syntheticcx.config.models import Criterion, PrivacyConfig, ScenarioConfig
from syntheticcx.domain.events import (
    MessageObserved,
    MessageObservedEvent,
    RepeatedInformationEvent,
    RepeatedInformationObserved,
)
from syntheticcx.domain.models import EvaluationResult, Message
from syntheticcx.errors import EvaluationError
from syntheticcx.evaluation.validation import validate_result
from syntheticcx.privacy import Redactor

NOW = datetime(2026, 9, 27, tzinfo=UTC)


def criterion(*, scored: bool = False, evidence_policy: str = "required") -> Criterion:
    return Criterion.model_validate(
        {
            "id": "goal",
            "evaluator_type": "semantic_goal_completion",
            "result_kind": "scored" if scored else "boolean",
            "threshold": 0.8 if scored else None,
            "evidence_policy": evidence_policy,
            "parameters": {"expected_facts": ["The plan costs 50 more."]},
        }
    )


def message(index: int, actor: str) -> MessageObservedEvent:
    return MessageObservedEvent.model_validate(
        {
            "id": f"event-{index}",
            "sequence": index,
            "timestamp": NOW,
            "type": "message_observed",
            "data": MessageObserved(
                message=Message.model_validate(
                    {
                        "id": f"message-{index}",
                        "sequence": index,
                        "actor": actor,
                        "content": "The new plan costs 50 more.",
                        "observed_at": NOW,
                    }
                )
            ).model_dump(),
        }
    )


def result(**changes: object) -> EvaluationResult:
    values = {
        "evaluator_id": "semantic_goal_completion",
        "criterion_id": "goal",
        "required": True,
        "status": "passed",
        "explanation": "The plan increase is explained.",
        "evidence_event_ids": ["event-1"],
        "evidence_message_ids": ["message-1"],
        **changes,
    }
    return EvaluationResult.model_validate(values)


def test_valid_paired_citation_and_boolean_decision() -> None:
    validate_result(result(), criterion(), [message(0, "customer"), message(1, "assistant")])


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"criterion_id": "other"}, "inconsistent"),
        ({"evaluator_id": "other"}, "inconsistent"),
        ({"required": False}, "inconsistent"),
        ({"explanation": " "}, "inconsistent"),
        ({"severity": " "}, "inconsistent"),
        ({"evidence_event_ids": []}, "not paired"),
        ({"evidence_message_ids": []}, "not paired"),
        ({"evidence_message_ids": ["message-0"]}, "not paired"),
        ({"evidence_event_ids": ["event-2"]}, "not committed"),
        ({"evidence_event_ids": ["event-1", "event-1"]}, "duplicate evidence"),
        ({"evidence_event_ids": [], "evidence_message_ids": []}, "required evidence"),
        ({"score": 0.9}, "boolean verdict"),
    ],
)
def test_rejects_invalid_or_uncommitted_evidence(changes: dict[str, object], expected: str) -> None:
    with pytest.raises(EvaluationError, match=expected):
        validate_result(
            result(**changes), criterion(), [message(0, "customer"), message(1, "assistant")]
        )


def test_required_missing_evidence_is_allowed_only_for_unavailable_or_error() -> None:
    for status in ("unavailable", "error"):
        validate_result(
            result(status=status, evidence_event_ids=[], evidence_message_ids=[]),
            criterion(),
            [],
        )
    validate_result(
        result(evidence_event_ids=[], evidence_message_ids=[]),
        criterion(evidence_policy="optional"),
        [],
    )


@pytest.mark.parametrize(
    ("score", "status", "valid"),
    [
        (0.8, "passed", True),
        (0.79, "failed", True),
        (0.79, "passed", False),
        (None, "passed", False),
        (0.8, "unavailable", False),
    ],
)
def test_scored_status_is_derived_from_threshold(
    score: float | None, status: str, valid: bool
) -> None:
    finding = result(score=score, status=status)
    events = [message(1, "assistant")]
    if valid:
        validate_result(finding, criterion(scored=True), events)
    else:
        with pytest.raises(EvaluationError):
            validate_result(finding, criterion(scored=True), events)


def test_repetition_evidence_must_pair_the_original_and_later_request() -> None:
    repeated = RepeatedInformationEvent(
        id="event-2",
        sequence=2,
        timestamp=NOW,
        type="repeated_information",
        data=RepeatedInformationObserved(
            information_id="account_number",
            original_message_id="message-0",
            request_message_id="message-1",
        ),
    )
    linked = result(
        evidence_event_ids=["event-0", "event-1", "event-2"],
        evidence_message_ids=["message-0", "message-1"],
    )
    events = [message(0, "customer"), message(1, "assistant"), repeated]
    validate_result(linked, criterion(), events)
    with pytest.raises(EvaluationError, match="message evidence is not paired"):
        validate_result(
            linked.model_copy(update={"evidence_message_ids": ["message-1"]}),
            criterion(),
            events,
        )


def test_hidden_expected_fact_snapshot_redacts_sensitive_context_values() -> None:
    scenario = ScenarioConfig(
        id="goal",
        description="Explain the bill",
        goal="Understand the charge",
        starting_context={"account_number": "private-marker"},
        success_criteria=[
            criterion().model_copy(
                update={"parameters": {"expected_facts": ["Account private-marker costs 50 more."]}}
            )
        ],
    )
    safe = Redactor(PrivacyConfig(redaction_keys=["account_number"])).redact_criterion(
        scenario.success_criteria[0], scenario
    )
    assert "private-marker" not in safe.model_dump_json()
    assert "<redacted:account_number>" in safe.parameters["expected_facts"][0]

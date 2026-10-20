"""Required criteria and infrastructure determine outcome, not stop signals."""

import pytest

from syntheticcx.config.models import Criterion
from syntheticcx.domain.models import EvaluationResult
from syntheticcx.domain.outcome import aggregate_outcome
from syntheticcx.errors import EvaluationError


def criterion(identifier: str, *, required: bool = True) -> Criterion:
    return Criterion(
        id=identifier,
        evaluator_type="journey_timeout",
        result_kind="boolean",
        evidence_policy="optional",
        required=required,
    )


def finding(identifier: str, status: str, *, required: bool = True) -> EvaluationResult:
    return EvaluationResult.model_validate(
        {
            "evaluator_id": "journey_timeout",
            "criterion_id": identifier,
            "required": required,
            "status": status,
            "explanation": "Observed evidence.",
        }
    )


@pytest.mark.parametrize(
    ("required_status", "optional_status", "reason", "infra", "expected"),
    [
        ("passed", "error", "goal_believed_complete", False, ("passed", 0)),
        ("passed", "failed", "max_turns", False, ("passed", 0)),
        ("failed", "passed", "goal_believed_complete", False, ("failed", 1)),
        ("unavailable", "passed", "abandonment", False, ("error", 3)),
        ("error", "passed", "max_turns", False, ("error", 3)),
        ("passed", "passed", "channel_error", False, ("error", 3)),
        ("passed", "passed", "journey_timeout", False, ("error", 3)),
        ("failed", "passed", "max_turns", True, ("error", 3)),
        ("passed", "passed", "cancelled", False, ("cancelled", 130)),
        ("failed", "passed", "cancelled", True, ("cancelled", 130)),
    ],
)
def test_outcome_table(
    required_status: str,
    optional_status: str,
    reason: str,
    infra: bool,
    expected: tuple[str, int],
) -> None:
    decision = aggregate_outcome(
        [criterion("required"), criterion("optional", required=False)],
        [
            finding("required", required_status),
            finding("optional", optional_status, required=False),
        ],
        termination_reason=reason,
        infrastructure_error=infra,
    )
    assert (decision.outcome, decision.exit_code) == expected


def test_missing_required_result_cannot_pass() -> None:
    decision = aggregate_outcome([criterion("required")], [], termination_reason="max_turns")
    assert (decision.outcome, decision.exit_code) == ("error", 3)


@pytest.mark.parametrize("ids", [["unknown"], ["required", "required"]])
def test_duplicate_or_unknown_results_are_rejected(ids: list[str]) -> None:
    with pytest.raises(EvaluationError, match="coverage"):
        aggregate_outcome(
            [criterion("required")],
            [finding(identifier, "passed") for identifier in ids],
            termination_reason="max_turns",
        )

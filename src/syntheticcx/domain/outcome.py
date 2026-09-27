"""Pure final-outcome policy over configured required criterion findings."""

from __future__ import annotations

from dataclasses import dataclass

from syntheticcx.config.models import Criterion
from syntheticcx.domain.models import EvaluationResult, Outcome
from syntheticcx.errors import EvaluationError

_INFRASTRUCTURE_REASONS = frozenset(
    {"channel_error", "model_error", "persistence_error", "journey_timeout"}
)


@dataclass(frozen=True)
class OutcomeDecision:
    outcome: Outcome
    exit_code: int


def aggregate_outcome(
    criteria: list[Criterion],
    results: list[EvaluationResult],
    *,
    termination_reason: str,
    infrastructure_error: bool = False,
) -> OutcomeDecision:
    """A stop signal never passes by itself; optional diagnostics never gate."""
    configured = {criterion.id: criterion for criterion in criteria}
    if len(configured) != len(criteria):
        raise EvaluationError("aggregate outcome", "criterion IDs are duplicated")
    by_id: dict[str, EvaluationResult] = {}
    for result in results:
        criterion = configured.get(result.criterion_id)
        if (
            criterion is None
            or result.criterion_id in by_id
            or result.required != criterion.required
        ):
            raise EvaluationError("aggregate outcome", "evaluation coverage is invalid")
        by_id[result.criterion_id] = result

    if termination_reason == "cancelled":
        return OutcomeDecision("cancelled", 130)
    if infrastructure_error or termination_reason in _INFRASTRUCTURE_REASONS:
        return OutcomeDecision("error", 3)

    required = [by_id.get(criterion.id) for criterion in criteria if criterion.required]
    if any(result is None or result.status in {"error", "unavailable"} for result in required):
        return OutcomeDecision("error", 3)
    if any(result is not None and result.status == "failed" for result in required):
        return OutcomeDecision("failed", 1)
    return OutcomeDecision("passed", 0)

"""Pure, auditable persona state transitions for configured events."""

from __future__ import annotations

from syntheticcx.config.models import BehaviorRule
from syntheticcx.domain.models import PersonaState, PersonaStateTransition


def apply_behavior(
    state: PersonaState,
    rules: list[BehaviorRule],
    *,
    event: str,
    trigger_ids: list[str],
) -> PersonaStateTransition:
    """Accumulate matching deltas before clamping, in priority/ID order.

    A threshold is checked after all deltas have been applied. Already-abandoned
    state is terminal and cannot be revived by later events.
    """
    matching = sorted(
        (rule for rule in rules if rule.event == event), key=lambda rule: (rule.priority, rule.id)
    )
    frustration = _clamp(state.frustration + sum(rule.frustration_delta for rule in matching))
    willingness = _clamp(
        state.willingness_to_continue + sum(rule.willingness_to_continue_delta for rule in matching)
    )
    abandoned = state.abandoned or any(
        rule.abandonment_threshold is not None and willingness <= rule.abandonment_threshold
        for rule in matching
    )
    after = state.model_copy(
        update={
            "frustration": frustration,
            "willingness_to_continue": willingness,
            "abandoned": abandoned,
        }
    )
    return PersonaStateTransition(
        trigger_ids=list(trigger_ids),
        before=state.model_copy(deep=True),
        after=PersonaState.model_validate(after.model_dump()),
        applied_rule_ids=[rule.id for rule in matching],
    )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))

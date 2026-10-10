"""Pure, auditable persona state transitions for configured signals."""

from __future__ import annotations

from collections.abc import Set

from syntheticcx.config.models import BehaviorRule
from syntheticcx.domain.models import PersonaState, PersonaStateTransition


def apply_behavior(
    state: PersonaState,
    rules: list[BehaviorRule],
    *,
    signal: str,
    trigger_ids: list[str],
    occurrence: int = 1,
    fired_rule_ids: Set[str] = frozenset(),
) -> PersonaStateTransition:
    """Accumulate matching deltas before clamping, in priority/ID order.

    A threshold is checked after all deltas have been applied. Already-abandoned
    state is terminal and cannot be revived by later events.
    """
    if occurrence < 1:
        raise ValueError("signal occurrence must be positive")
    matching = sorted(
        (
            rule
            for rule in rules
            if not state.abandoned
            and rule.when.signal == signal
            and occurrence >= rule.when.occurrence_at_least
            and (rule.fire != "once" or rule.id not in fired_rule_ids)
            and _state_matches(state, rule)
        ),
        key=lambda rule: (rule.priority, rule.id),
    )
    frustration = _clamp(
        state.frustration + sum(rule.effects.frustration_delta for rule in matching)
    )
    willingness = _clamp(
        state.willingness_to_continue
        + sum(rule.effects.willingness_to_continue_delta for rule in matching)
    )
    abandoned = state.abandoned or any(
        rule.effects.abandon is True
        or (
            rule.effects.abandonment_threshold is not None
            and willingness <= rule.effects.abandonment_threshold
        )
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


def _state_matches(state: PersonaState, rule: BehaviorRule) -> bool:
    conditions = rule.when.state
    if conditions is None:
        return True
    for name in ("frustration", "willingness_to_continue"):
        condition = getattr(conditions, name)
        if condition is None:
            continue
        value = getattr(state, name)
        if condition.gte is not None and value < condition.gte:
            return False
        if condition.lte is not None and value > condition.lte:
            return False
    return True


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))

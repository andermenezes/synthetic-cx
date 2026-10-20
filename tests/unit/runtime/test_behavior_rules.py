"""Deterministic condition and terminal-state boundaries for version-1 behavioral rules."""

import pytest

from syntheticcx.config.models import BehaviorRule
from syntheticcx.domain.models import PersonaState
from syntheticcx.runtime.behavior import apply_behavior


def rule(identifier: str, **values: object) -> BehaviorRule:
    return BehaviorRule.model_validate(
        {
            "id": identifier,
            "priority": 10,
            "when": {"signal": "repeated_information"},
            "effects": {"frustration_delta": 0.2},
            **values,
        }
    )


@pytest.mark.parametrize("occurrence,expected", [(1, 0.1), (2, 0.3), (3, 0.3)])
def test_occurrence_threshold_is_inclusive(occurrence: int, expected: float) -> None:
    transition = apply_behavior(
        PersonaState(frustration=0.1),
        [rule("repeat", when={"signal": "repeated_information", "occurrence_at_least": 2})],
        signal="repeated_information",
        occurrence=occurrence,
        trigger_ids=["request"],
    )
    assert transition.after.frustration == pytest.approx(expected)
    assert transition.applied_rule_ids == ([] if occurrence == 1 else ["repeat"])


def test_state_conditions_use_pre_signal_state_for_all_matching_rules() -> None:
    state = PersonaState(frustration=0.2, willingness_to_continue=0.6)
    conditions = {
        "signal": "repeated_information",
        "state": {
            "frustration": {"gte": 0.2, "lte": 0.2},
            "willingness_to_continue": {"gte": 0.5},
        },
    }
    transition = apply_behavior(
        state,
        [
            rule("first", priority=0, effects={"frustration_delta": 0.5}),
            rule("second", when=conditions, effects={"willingness_to_continue_delta": -0.2}),
        ],
        signal="repeated_information",
        trigger_ids=["request"],
    )
    assert transition.applied_rule_ids == ["first", "second"]
    assert transition.after.frustration == pytest.approx(0.7)
    assert transition.after.willingness_to_continue == pytest.approx(0.4)
    assert state.frustration == 0.2


def test_once_rules_do_not_reapply_but_every_match_rules_do() -> None:
    rules = [rule("once", fire="once"), rule("each")]
    first = apply_behavior(
        PersonaState(), rules, signal="repeated_information", trigger_ids=["request-1"]
    )
    second = apply_behavior(
        first.after,
        rules,
        signal="repeated_information",
        trigger_ids=["request-2"],
        occurrence=2,
        fired_rule_ids=set(first.applied_rule_ids),
    )
    assert first.applied_rule_ids == ["each", "once"]
    assert second.applied_rule_ids == ["each"]
    assert second.after.frustration == pytest.approx(0.6)


def test_explicit_abandonment_is_terminal_even_with_recovery_effects() -> None:
    rules = [
        rule("leave", effects={"abandon": True}),
        rule("recover", effects={"frustration_delta": -0.8, "willingness_to_continue_delta": 0.8}),
    ]
    first = apply_behavior(
        PersonaState(frustration=0.9, willingness_to_continue=0.1),
        rules,
        signal="repeated_information",
        trigger_ids=["request-1"],
    )
    assert first.after.abandoned
    second = apply_behavior(
        first.after,
        rules,
        signal="repeated_information",
        occurrence=2,
        trigger_ids=["request-2"],
    )
    assert second.after == first.after
    assert second.applied_rule_ids == []

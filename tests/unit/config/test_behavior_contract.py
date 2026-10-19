"""Structural behavioral configuration contracts without user-facing version fields."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from syntheticcx.config.loader import (
    load_config_text,
    load_config_with_source_hash,
    load_discovery_draft,
)
from syntheticcx.config.models import (
    BehaviorCondition,
    BehaviorConfig,
    BehaviorDetectorConfig,
    BehaviorEffects,
    BehaviorRule,
    BehaviorStateCondition,
    BehaviorStateConditions,
    InterfaceChannelGuide,
    PersonaConfig,
    ProjectConfig,
)
from syntheticcx.config.settings import prepare_settings
from syntheticcx.errors import ConfigurationError


def rule(**changes: Any) -> dict[str, Any]:
    return {
        "id": "repetition-frustration",
        "priority": 10,
        "when": {"signal": "repeated_information"},
        "effects": {"frustration_delta": 0.2},
        **changes,
    }


def detector(**changes: Any) -> dict[str, Any]:
    return {"id": "custom-detector", "type": "extension.custom.v1", **changes}


@pytest.fixture
def document() -> dict[str, Any]:
    return {
        "scenario": {
            "id": "billing",
            "description": "Explain an invoice.",
            "goal": "Understand the invoice.",
            "success_criteria": [
                {
                    "id": "turn-limit",
                    "evaluator_type": "turn_count",
                    "result_kind": "boolean",
                    "evidence_policy": "none",
                    "parameters": {"max_turns": 5},
                }
            ],
        },
        "persona": {
            "id": "customer",
            "communication": {"tone": "polite", "style": "concise"},
            "initial_state": {},
        },
        "channel": {
            "type": "webchat_playwright_dynamic",
            "url": "https://chat.example/support",
            "ownership": {
                "root_locator": {"strategy": "test_id", "value": "chat"},
                "proof": "chat_dialog",
            },
            "assistant_message_locator": {"strategy": "test_id", "value": "assistant"},
            "message_identity": {"strategy": "dom_node"},
            "response_completion": {"strategy": "quiet_interval", "quiet_interval_seconds": 0.1},
        },
        "execution": {
            "max_turns": 5,
            "journey_timeout_seconds": 120,
            "greeting_grace_seconds": 2,
            "browser_action_timeout_seconds": 10,
            "response_timeout_seconds": 20,
            "model_timeout_seconds": 30,
            "evaluator_timeout_seconds": 30,
            "persistence_timeout_seconds": 5,
            "close_timeout_seconds": 5,
        },
        "model": {"provider": "openai", "model": "test-model"},
    }


def guide_document(channel: dict[str, Any]) -> dict[str, Any]:
    return {
        "channel": channel,
        "validation": {
            "status": "verified",
            "verified_at": "2026-10-10T12:00:00Z",
            "target_origin": "https://chat.example",
            "observed_behaviors": ["open", "select_option", "receive", "completion"],
        },
    }


@pytest.mark.parametrize("identifier", ["A", "a" * 80, "Custom.signal.v1-name_2"])
def test_behavior_identifiers_accept_bounded_extensible_names(identifier: str) -> None:
    parsed = BehaviorRule.model_validate(rule(id=identifier, when={"signal": identifier}))
    assert parsed.id == identifier
    config = BehaviorDetectorConfig.model_validate(detector(id=identifier, type=identifier))
    assert config.id == config.type == identifier


@pytest.mark.parametrize(
    "identifier",
    [
        "",
        " ",
        "a" * 81,
        "1first",
        "_first",
        "-first",
        "évent",
        "a b",
        "a:signal",
        "a/signal",
        "a\n",
    ],
)
@pytest.mark.parametrize("target", ["rule_id", "signal", "detector_id", "detector_type"])
def test_behavior_identifiers_reject_invalid_names(identifier: str, target: str) -> None:
    with pytest.raises(ValidationError):
        if target == "rule_id":
            BehaviorRule.model_validate(rule(id=identifier))
        elif target == "signal":
            BehaviorCondition(signal=identifier)
        else:
            field = "id" if target == "detector_id" else "type"
            BehaviorDetectorConfig.model_validate(detector(**{field: identifier}))


@pytest.mark.parametrize("field", ["priority", "occurrence_at_least"])
@pytest.mark.parametrize("value", [True, False, "1", 1.0, 1.5, None])
def test_rule_integer_fields_are_strict(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        if field == "priority":
            BehaviorRule.model_validate(rule(priority=value))
        else:
            BehaviorCondition(signal="custom_signal", occurrence_at_least=value)


@pytest.mark.parametrize("value", [0, -1])
def test_occurrence_requires_positive_integer(value: int) -> None:
    with pytest.raises(ValidationError):
        BehaviorCondition(signal="custom_signal", occurrence_at_least=value)


@pytest.mark.parametrize("priority", [-10, 0, 10])
def test_rule_defaults_and_signed_priority(priority: int) -> None:
    parsed = BehaviorRule.model_validate(rule(priority=priority))
    assert parsed.priority == priority
    assert parsed.when.occurrence_at_least == 1
    assert parsed.when.state is None
    assert parsed.fire == "every_match"
    assert parsed.description is None
    assert parsed.effects.willingness_to_continue_delta == 0


@pytest.mark.parametrize("fire", ["every_match", "once"])
def test_rule_fire_modes(fire: str) -> None:
    assert BehaviorRule.model_validate(rule(fire=fire)).fire == fire


def test_unknown_fire_mode_is_rejected() -> None:
    with pytest.raises(ValidationError):
        BehaviorRule.model_validate(rule(fire="sometimes"))


@pytest.mark.parametrize(
    "bounds", [{"gte": 0}, {"lte": 1}, {"gte": 0, "lte": 1}, {"gte": 0.5, "lte": 0.5}]
)
def test_state_comparisons_accept_inclusive_bounds(bounds: dict[str, float]) -> None:
    parsed = BehaviorStateCondition.model_validate(bounds)
    for field, value in bounds.items():
        assert getattr(parsed, field) == value


@pytest.mark.parametrize("field", ["gte", "lte"])
@pytest.mark.parametrize("value", [-0.01, 1.01, float("inf"), float("-inf"), float("nan")])
def test_state_comparisons_reject_out_of_range_or_nonfinite_bounds(
    field: str, value: float
) -> None:
    with pytest.raises(ValidationError):
        BehaviorStateCondition.model_validate({field: value})


@pytest.mark.parametrize("bounds", [{}, {"gte": None, "lte": None}, {"gte": 0.8, "lte": 0.2}])
def test_state_comparisons_reject_empty_or_contradictory_bounds(bounds: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        BehaviorStateCondition.model_validate(bounds)


@pytest.mark.parametrize("conditions", [{}, {"frustration": None, "willingness_to_continue": None}])
def test_state_conditions_require_a_comparison(conditions: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        BehaviorStateConditions.model_validate(conditions)


def test_state_conditions_support_both_owned_state_fields() -> None:
    parsed = BehaviorCondition.model_validate(
        {
            "signal": "custom_signal",
            "occurrence_at_least": 2,
            "state": {"frustration": {"gte": 0.2}, "willingness_to_continue": {"lte": 0.8}},
        }
    )
    assert parsed.state is not None
    assert parsed.state.frustration == BehaviorStateCondition(gte=0.2)
    assert parsed.state.willingness_to_continue == BehaviorStateCondition(lte=0.8)


@pytest.mark.parametrize("field", ["frustration_delta", "willingness_to_continue_delta"])
@pytest.mark.parametrize("value", [-1, 1])
def test_effects_accept_delta_boundaries(field: str, value: float) -> None:
    assert getattr(BehaviorEffects.model_validate({field: value}), field) == value


@pytest.mark.parametrize("field", ["frustration_delta", "willingness_to_continue_delta"])
@pytest.mark.parametrize("value", [-1.01, 1.01, float("inf"), float("-inf"), float("nan")])
def test_effects_reject_invalid_deltas(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        BehaviorEffects.model_validate({field: value})


@pytest.mark.parametrize("threshold", [0, 1])
def test_threshold_is_an_effect_even_at_zero(threshold: float) -> None:
    assert BehaviorEffects(abandonment_threshold=threshold).abandonment_threshold == threshold


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("inf"), float("-inf"), float("nan")])
def test_threshold_rejects_invalid_values(threshold: float) -> None:
    with pytest.raises(ValidationError):
        BehaviorEffects(abandonment_threshold=threshold)


@pytest.mark.parametrize(
    "effects",
    [
        {},
        {"frustration_delta": 0},
        {"willingness_to_continue_delta": -0.0},
        {
            "frustration_delta": 0,
            "willingness_to_continue_delta": 0,
            "abandonment_threshold": None,
            "abandon": None,
        },
    ],
)
def test_empty_and_noop_effects_are_rejected(effects: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        BehaviorEffects.model_validate(effects)


@pytest.mark.parametrize("value", [False, 0, 1, "true", "false"])
def test_abandon_requires_explicit_true(value: Any) -> None:
    with pytest.raises(ValidationError):
        BehaviorEffects.model_validate({"abandon": value, "frustration_delta": 0.1})


def test_explicit_abandonment_is_an_effect() -> None:
    assert BehaviorEffects(abandon=True).abandon is True


@pytest.mark.parametrize("field", ["behavior_description", "description"])
@pytest.mark.parametrize("value", [None, "A", "x" * 2000])
def test_optional_behavior_prose_accepts_boundaries(
    field: str, value: str | None, document: dict[str, Any]
) -> None:
    if field == "behavior_description":
        parsed = PersonaConfig.model_validate({**document["persona"], field: value})
    else:
        parsed = BehaviorRule.model_validate(rule(description=value))
    assert getattr(parsed, field) == value


@pytest.mark.parametrize("field", ["behavior_description", "description"])
@pytest.mark.parametrize("value", ["", " \n\t", "x" * 2001, 42])
def test_optional_behavior_prose_rejects_invalid_values(
    field: str, value: Any, document: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError):
        if field == "behavior_description":
            PersonaConfig.model_validate({**document["persona"], field: value})
        else:
            BehaviorRule.model_validate(rule(description=value))


def test_detector_schema_does_not_expose_implementation_versions() -> None:
    assert "version" not in BehaviorDetectorConfig.model_fields
    assert "version" not in BehaviorDetectorConfig.model_json_schema()["properties"]


def test_detector_implementation_version_is_owned_by_registration() -> None:
    config = BehaviorDetectorConfig.model_validate(detector())
    assert not hasattr(config, "version")


def test_detector_parameters_accept_nested_json_objects() -> None:
    parsed = BehaviorDetectorConfig.model_validate(
        detector(parameters={"nested": {"values": [1, True, None]}})
    )
    assert parsed.parameters["nested"] == {"values": [1, True, None]}
    with pytest.raises(ValidationError):
        BehaviorDetectorConfig.model_validate(detector(parameters=[]))
    with pytest.raises(ValidationError):
        BehaviorDetectorConfig.model_validate(detector(parameters={"value": object()}))


@pytest.mark.parametrize("count", [0, 8])
def test_detector_count_accepts_cap(count: int) -> None:
    parsed = BehaviorConfig.model_validate(
        {"detectors": [detector(id=f"detector-{index}") for index in range(count)]}
    )
    assert len(parsed.detectors) == count


def test_detector_count_and_duplicate_ids_fail_closed() -> None:
    with pytest.raises(ValidationError):
        BehaviorConfig.model_validate(
            {"detectors": [detector(id=f"detector-{index}") for index in range(9)]}
        )
    with pytest.raises(ValidationError):
        BehaviorConfig.model_validate({"detectors": [detector(), detector(type="other_type")]})


@pytest.mark.parametrize("count", [0, 32])
def test_rule_count_accepts_cap(count: int, document: dict[str, Any]) -> None:
    persona = {
        **document["persona"],
        "behavioral_rules": [rule(id=f"rule-{index}") for index in range(count)],
    }
    assert len(PersonaConfig.model_validate(persona).behavioral_rules) == count


def test_rule_count_and_duplicate_ids_fail_closed(document: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        PersonaConfig.model_validate(
            {
                **document["persona"],
                "behavioral_rules": [rule(id=f"rule-{index}") for index in range(33)],
            }
        )
    with pytest.raises(ValidationError):
        PersonaConfig.model_validate(
            {**document["persona"], "behavioral_rules": [rule(), rule(priority=20)]}
        )


@pytest.mark.parametrize(
    "model,value",
    [
        (BehaviorCondition, {"signal": "custom_signal", "expression": "anything"}),
        (BehaviorEffects, {"abandon": True, "turn_number": 1}),
        (BehaviorStateCondition, {"gte": 0, "gt": 0}),
        (BehaviorStateConditions, {"frustration": {"gte": 0}, "abandoned": True}),
        (BehaviorRule, rule(event="repeated_information")),
        (BehaviorDetectorConfig, detector(factory="anything")),
        (BehaviorConfig, {"detectors": [], "registry": {}}),
    ],
)
def test_behavior_sections_forbid_unknown_fields(model: Any, value: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(value)


def test_behavior_models_remain_frozen() -> None:
    parsed = BehaviorRule.model_validate(rule())
    with pytest.raises(ValidationError):
        parsed.priority = 20


def test_version_two_loads_extensible_structural_behavior(document: dict[str, Any]) -> None:
    document["persona"].update(
        behavior_description="Requests concise explanations.",
        behavioral_rules=[
            rule(fire="once", when={"signal": "custom.signal", "occurrence_at_least": 2})
        ],
    )
    document["behavior"] = {"detectors": [detector(parameters={"limit": 2})]}
    config = load_config_text(yaml.safe_dump(document))
    assert config.persona.behavior_description == document["persona"]["behavior_description"]
    assert config.behavior.detectors[0].parameters == {"limit": 2}
    parsed_rule = config.persona.behavioral_rules[0]
    assert parsed_rule.when.signal == "custom.signal"
    assert parsed_rule.when.occurrence_at_least == 2
    assert parsed_rule.fire == "once"
    assert parsed_rule.effects.frustration_delta == 0.2


def test_omitted_detector_selection_supports_both_builtin_rules(document: dict[str, Any]) -> None:
    document["persona"]["behavioral_rules"] = [
        rule(),
        rule(id="completed", when={"signal": "customer_turn_completed"}),
    ]
    config = ProjectConfig.model_validate(document)
    prepared = prepare_settings(config, {"OPENAI_API_KEY": "test-credential"})
    assert prepared.persona.credential.get_secret_value() == "test-credential"


def test_legacy_flat_rule_shape_is_rejected(document: dict[str, Any]) -> None:
    document["persona"]["behavioral_rules"] = [
        {"id": "legacy", "priority": 10, "event": "repeated_information", "frustration_delta": 0.2}
    ]
    with pytest.raises(ConfigurationError):
        load_config_text(yaml.safe_dump(document))


def test_journey_configuration_does_not_expose_a_version_field(
    document: dict[str, Any],
) -> None:
    assert "version" not in ProjectConfig.model_fields
    assert "version" not in ProjectConfig.model_json_schema()["properties"]
    with pytest.raises(ConfigurationError):
        document["version"] = "1"
        load_config_text(yaml.safe_dump(document))


def test_journey_one_references_guide_one_without_changing_guide_contract(
    document: dict[str, Any], tmp_path: Path
) -> None:
    guide = guide_document(document.pop("channel"))
    document["channel_guide"] = "guide.yaml"
    guide_path = tmp_path / "guide.yaml"
    guide_path.write_text(yaml.safe_dump(guide))
    journey_path = tmp_path / "journey.yaml"
    journey_path.write_text(yaml.safe_dump(document))
    loaded = load_config_with_source_hash(journey_path)
    assert loaded.config.version == "1"
    assert loaded.guide_source_hash is not None
    assert "version" not in InterfaceChannelGuide.model_validate(guide).model_dump()
    changed = deepcopy(guide)
    changed["version"] = "1"
    with pytest.raises(ValidationError):
        InterfaceChannelGuide.model_validate(changed)


@pytest.mark.parametrize(
    "detectors,rules",
    [
        ([detector(type="unregistered_type")], []),
        ([detector(type="repeated_information", parameters={"unsupported": True})], []),
        (
            [
                detector(id="first", type="repeated_information"),
                detector(id="second", type="repeated_information"),
            ],
            [],
        ),
        ([], [rule()]),
        ([detector(type="customer_turn_completed")], [rule()]),
        (
            [detector(type="repeated_information")],
            [rule(when={"signal": "unregistered_signal"})],
        ),
    ],
)
def test_builtin_preflight_rejects_invalid_behavior_before_credentials(
    detectors: list[dict[str, Any]], rules: list[dict[str, Any]], document: dict[str, Any]
) -> None:
    document["behavior"] = {"detectors": detectors}
    document["persona"]["behavioral_rules"] = rules
    config = ProjectConfig.model_validate(document)
    with pytest.raises(ConfigurationError) as caught:
        prepare_settings(config, {})
    assert caught.value.operation == "prepare behavior"


def test_explicitly_disabled_behavior_prepares_without_rules(document: dict[str, Any]) -> None:
    document["behavior"] = {"detectors": []}
    config = ProjectConfig.model_validate(document)
    prepared = prepare_settings(config, {"OPENAI_API_KEY": "test-credential"})
    assert prepared.persona.credential.get_secret_value() == "test-credential"
    assert config.behavior.detectors == []


def test_version_one_draft_accepts_behavior_and_no_channel(
    document: dict[str, Any], tmp_path: Path
) -> None:
    document.pop("channel")
    document["persona"]["behavioral_rules"] = [rule()]
    source = tmp_path / "draft.yaml"
    source.write_text(yaml.safe_dump(document))
    draft = load_discovery_draft(source)
    assert draft.version == "1"
    assert draft.persona.behavioral_rules[0] == BehaviorRule.model_validate(rule())

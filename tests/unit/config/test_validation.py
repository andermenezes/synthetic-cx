"""Configuration boundary regressions without adapter construction."""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator

from syntheticcx.config.loader import load_config_text
from syntheticcx.config.models import ProjectConfig
from syntheticcx.config.sanitize import sanitized_effective_config
from syntheticcx.config.schema import export_project_schema
from syntheticcx.config.settings import prepare_settings, resolve_model_settings
from syntheticcx.domain.models import CustomerScenarioContext, Scenario
from syntheticcx.errors import ConfigurationError


@pytest.fixture
def document() -> dict[str, Any]:
    return yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())


@pytest.mark.parametrize(
    ("path", "value", "location"),
    [
        (("version",), "3", "version"),
        (("includes",), "other.yaml", "includes"),
        (("execution", "max_turns"), 0, "execution.max_turns"),
        (("execution", "model_timeout_seconds"), float("inf"), "execution.model_timeout"),
        (("persona", "traits"), {"patience": 1.1}, "persona.traits.patience"),
        (("persona", "description"), " \n\t", "persona.description"),
        (("persona", "description"), "x" * 2001, "persona.description"),
        (("persona", "description"), 42, "persona.description"),
        (("persona", "communication", "writing_description"), "", "writing_description"),
        (("persona", "communication", "writing_description"), "  ", "writing_description"),
        (("persona", "communication", "writing_description"), "x" * 2001, "writing_description"),
        (("persona", "communication", "message_length"), "secret-value", "message_length"),
        (
            ("persona", "communication", "grammar_error_probability"),
            -0.1,
            "grammar_error_probability",
        ),
        (
            ("persona", "communication", "grammar_error_probability"),
            1.1,
            "grammar_error_probability",
        ),
        (
            ("persona", "communication", "grammar_error_probability"),
            float("nan"),
            "grammar_error_probability",
        ),
        (
            ("persona", "communication", "grammar_error_probability"),
            float("inf"),
            "grammar_error_probability",
        ),
        (("channel", "type"), "unknown", "channel.type"),
        (
            ("channel", "bootstrap_send_strategy"),
            {"type": "click"},
            "channel.bootstrap_send_strategy",
        ),
        (("model", "api_key"), "secret-value", "model.openai.api_key"),
        (("model", "provider"), "secret-value", "model"),
        (("privacy", "synthetic_id_patterns"), ["["], "privacy.synthetic_id_patterns"),
        (("storage", "store_screenshots"), True, "configuration"),
        (("telemetry", "remote_export"), True, "telemetry"),
    ],
)
def test_invalid_configuration_reports_field_without_values(
    document: dict[str, Any], path: tuple[str, ...], value: Any, location: str
) -> None:
    target = document
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(ConfigurationError) as caught:
        load_config_text(yaml.safe_dump(document))
    assert location in str(caught.value)
    assert "secret-value" not in str(caught.value)


def compatible(url: str) -> dict[str, Any]:
    return {
        "provider": "openai_compatible",
        "model": "local-model",
        "base_url": url,
        "credential_environment_variable": "LOCAL_KEY",
        "capabilities": dict.fromkeys(
            ["chat_completions", "structured_output", "embeddings", "token_usage", "model_version"],
            True,
        ),
    }


def test_legacy_persona_has_optional_writing_defaults(document: dict[str, Any]) -> None:
    config = load_config_text(yaml.safe_dump(document))
    assert config.persona.description is None
    assert config.persona.communication.writing_description is None
    assert config.persona.communication.message_length is None
    assert config.persona.communication.grammar_error_probability == 0


@pytest.mark.parametrize("length", ["short", "medium", "long", None])
@pytest.mark.parametrize("probability", [0, 0.15, 1])
def test_persona_writing_fields_round_trip(
    document: dict[str, Any], length: str | None, probability: float
) -> None:
    document["persona"]["description"] = "  A busy customer with limited technical knowledge.  "
    document["persona"]["communication"].update(
        writing_description="Uses WhatsApp abbreviations and occasionally omits punctuation.",
        message_length=length,
        grammar_error_probability=probability,
    )
    config = load_config_text(yaml.safe_dump(document))
    restored = load_config_text(yaml.safe_dump(config.model_dump(mode="json")))
    assert restored == config
    assert config.persona.description == document["persona"]["description"]
    assert config.persona.communication.message_length == length
    assert config.persona.communication.grammar_error_probability == probability
    validator = Draft202012Validator(json.loads(Path("schemas/project.schema.json").read_text()))
    assert validator.is_valid(document)


def test_sensitive_persona_prose_is_redacted_in_configuration_echoes(
    document: dict[str, Any],
) -> None:
    document["persona"]["description"] = 'Sensitive persona "narrative"'
    document["persona"]["communication"]["writing_description"] = "Sensitive writing voice"
    document["scenario"]["description"] = "Sensitive writing voice"
    document["privacy"]["redaction_keys"] = ["description", "writing_description"]
    config = ProjectConfig.model_validate(document)
    snapshot = sanitized_effective_config(config)
    assert "Sensitive" not in json.dumps(snapshot)
    assert config.persona.description == 'Sensitive persona "narrative"'


@pytest.mark.parametrize("url", ["http://localhost:8000", "http://127.0.0.2", "http://[::1]"])
def test_loopback_endpoints_and_explicit_evaluator_credentials(
    document: dict[str, Any], url: str
) -> None:
    document["evaluation_model"] = compatible(url)
    config = load_config_text(yaml.safe_dump(document))
    environment = {"OPENAI_API_KEY": "ambient", "LOCAL_KEY": "explicit"}
    assert resolve_model_settings(config, environment).credential.get_secret_value() == "ambient"
    evaluator = resolve_model_settings(config, environment, evaluation=True)
    assert evaluator.credential.get_secret_value() == "explicit"
    with pytest.raises(ConfigurationError, match="LOCAL_KEY"):
        resolve_model_settings(config, {"OPENAI_API_KEY": "ambient"}, evaluation=True)


@pytest.mark.parametrize("url", ["http://example.com", "https://user:secret@example.com"])
def test_rejects_unsafe_endpoint_configuration(document: dict[str, Any], url: str) -> None:
    document["model"] = compatible(url)
    with pytest.raises(ConfigurationError):
        load_config_text(yaml.safe_dump(document))


def test_snapshot_redacts_nested_fields_patterns_and_resolved_secrets(
    document: dict[str, Any],
) -> None:
    document["scenario"]["knowledge"]["nested"] = {"account_number": "account-secret"}
    document["scenario"]["description"] = "credential-secret SYNTH-123"
    document["privacy"]["synthetic_id_patterns"] = [r"SYNTH-\d+"]
    config = ProjectConfig.model_validate(document)
    snapshot = sanitized_effective_config(config, resolved_secrets=["credential-secret"])
    serialized = json.dumps(snapshot)
    for value in ["account-secret", "credential-secret", "SYNTH-123", "SYNTHETIC-001"]:
        assert value not in serialized
    assert snapshot["storage"]["flush_each_event"] is True
    assert config.scenario.description == "credential-secret SYNTH-123"


def test_checked_in_schema_matches_generated_contract(tmp_path: Path) -> None:
    generated = export_project_schema(tmp_path / "project.schema.json")
    assert generated.read_text() == Path("schemas/project.schema.json").read_text()


@pytest.mark.parametrize(
    ("evaluator_type", "parameters", "result_kind", "location"),
    [
        ("turn_count", {}, "boolean", "parameters.max_turns"),
        ("turn_count", {"max_turns": True}, "boolean", "parameters.max_turns"),
        ("turn_count", {"max_turns": 1, "extra": 2}, "boolean", "parameters.extra"),
        ("turn_count", {"max_turns": 1}, "scored", "result_kind"),
        ("journey_timeout", {"extra": 1}, "boolean", "parameters.extra"),
        ("repeated_information", {"max_repeats": -1}, "boolean", "parameters.max_repeats"),
        ("repeated_information", {"max_repeats": False}, "boolean", "parameters.max_repeats"),
        ("semantic_goal_completion", {}, "boolean", "parameters.expected_facts"),
        (
            "semantic_goal_completion",
            {"expected_facts": [" "]},
            "boolean",
            "parameters.expected_facts",
        ),
        (
            "semantic_goal_completion",
            {"expected_facts": ["invoice charge"], "extra": 1},
            "boolean",
            "parameters.extra",
        ),
    ],
)
def test_evaluator_parameters_fail_before_external_activity(
    document: dict[str, Any],
    evaluator_type: str,
    parameters: dict[str, Any],
    result_kind: str,
    location: str,
) -> None:
    criterion = document["scenario"]["success_criteria"][0]
    criterion.update(evaluator_type=evaluator_type, parameters=parameters, result_kind=result_kind)
    if result_kind == "scored":
        criterion["threshold"] = 0.8
    with pytest.raises(ConfigurationError) as caught:
        load_config_text(yaml.safe_dump(document))
    assert location in str(caught.value)


def test_turn_count_bound_uses_root_execution_limit(document: dict[str, Any]) -> None:
    document["scenario"]["success_criteria"][0].update(
        evaluator_type="turn_count", parameters={"max_turns": 6}
    )
    with pytest.raises(ConfigurationError, match="success_criteria.parameters.max_turns"):
        load_config_text(yaml.safe_dump(document))
    document["scenario"]["success_criteria"][0]["parameters"]["max_turns"] = 5
    assert load_config_text(yaml.safe_dump(document)).scenario.success_criteria[0].parameters == {
        "max_turns": 5
    }


def test_evaluator_parameter_defaults_and_scored_semantic_contract(
    document: dict[str, Any],
) -> None:
    criterion = document["scenario"]["success_criteria"][0]
    criterion.update(evaluator_type="repeated_information", parameters={})
    assert load_config_text(yaml.safe_dump(document)).scenario.success_criteria[0].parameters == {
        "max_repeats": 0
    }
    criterion.update(
        evaluator_type="semantic_goal_completion",
        result_kind="scored",
        threshold=0.8,
        parameters={"expected_facts": ["one-time charge"]},
    )
    assert load_config_text(yaml.safe_dump(document)).scenario.success_criteria[0].threshold == 0.8


def test_expected_facts_remain_outside_persona_context(document: dict[str, Any]) -> None:
    marker = "EVALUATOR-ONLY-7A-MARKER"
    document["scenario"]["success_criteria"][0]["parameters"]["expected_facts"] = [marker]
    scenario = Scenario.model_validate(
        load_config_text(yaml.safe_dump(document)).scenario.model_dump()
    )
    context = CustomerScenarioContext.from_scenario(scenario)
    assert marker not in context.model_dump_json()
    assert marker in scenario.success_criteria[0].parameters["expected_facts"]


def test_project_schema_enforces_evaluator_specific_parameters(document: dict[str, Any]) -> None:
    validator = Draft202012Validator(json.loads(Path("schemas/project.schema.json").read_text()))
    assert validator.is_valid(document)
    criterion = document["scenario"]["success_criteria"][0]
    criterion["parameters"] = {"max_turns": 2}
    assert not validator.is_valid(document)
    criterion.update(evaluator_type="turn_count", result_kind="boolean")
    assert validator.is_valid(document)
    criterion["parameters"] = {"max_turns": 2, "unknown": 1}
    assert not validator.is_valid(document)
    criterion.update(
        evaluator_type="semantic_goal_completion", parameters={"expected_facts": ["charge"]}
    )
    criterion.update(result_kind="scored")
    assert not validator.is_valid(document)
    criterion["threshold"] = 0.8
    assert validator.is_valid(document)


@pytest.mark.parametrize("field", ["success_criteria", "behavioral_rules"])
def test_duplicate_ids_are_rejected(document: dict[str, Any], field: str) -> None:
    section = "scenario" if field == "success_criteria" else "persona"
    document[section][field] *= 2
    with pytest.raises(ConfigurationError, match="IDs must be unique"):
        load_config_text(yaml.safe_dump(document))


@pytest.mark.parametrize("capability", ["chat_completions", "structured_output"])
def test_preparation_rejects_missing_required_capabilities(
    document: dict[str, Any], capability: str
) -> None:
    document["model"] = compatible("http://localhost")
    document["model"]["capabilities"][capability] = False
    config = load_config_text(yaml.safe_dump(document))
    with pytest.raises(ConfigurationError, match=capability):
        prepare_settings(config, {"LOCAL_KEY": "explicit"})


def test_persona_needs_structured_actions_even_without_semantic_evaluation(
    document: dict[str, Any],
) -> None:
    document["scenario"]["success_criteria"] = [
        {
            "id": "turns",
            "evaluator_type": "turn_count",
            "result_kind": "boolean",
            "evidence_policy": "none",
            "parameters": {"max_turns": 5},
        }
    ]
    document["model"] = compatible("http://localhost")
    document["model"]["capabilities"]["structured_output"] = False
    config = load_config_text(yaml.safe_dump(document))
    with pytest.raises(ConfigurationError, match="model.capabilities.structured_output"):
        prepare_settings(config, {"LOCAL_KEY": "explicit"})


def test_unused_evaluation_model_does_not_require_structured_output(
    document: dict[str, Any],
) -> None:
    document["scenario"]["success_criteria"] = [
        {
            "id": "turns",
            "evaluator_type": "turn_count",
            "result_kind": "boolean",
            "evidence_policy": "none",
            "parameters": {"max_turns": 5},
        }
    ]
    document["evaluation_model"] = compatible("http://localhost")
    document["evaluation_model"]["capabilities"]["structured_output"] = False
    config = load_config_text(yaml.safe_dump(document))
    assert prepare_settings(config, {"OPENAI_API_KEY": "persona", "LOCAL_KEY": "evaluation"})


def test_preparation_requires_telemetry_extra(
    document: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from importlib.metadata import PackageNotFoundError

    def missing_distribution(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr("syntheticcx.config.settings.version", missing_distribution)
    document["telemetry"].update(remote_export=True, otlp_endpoint="https://collector.example")
    config = load_config_text(yaml.safe_dump(document))
    with pytest.raises(ConfigurationError, match="telemetry extra"):
        prepare_settings(config, {"OPENAI_API_KEY": "secret"})


def test_preparation_inherits_evaluation_credentials(document: dict[str, Any]) -> None:
    settings = prepare_settings(
        ProjectConfig.model_validate(document), {"OPENAI_API_KEY": "secret"}
    )
    assert settings.persona == settings.evaluation
    assert "secret" not in settings.model_dump_json()

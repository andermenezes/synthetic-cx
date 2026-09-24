"""Configuration boundary regressions without adapter construction."""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from syntheticcx.config.loader import load_config_text
from syntheticcx.config.models import ProjectConfig
from syntheticcx.config.sanitize import sanitized_effective_config
from syntheticcx.config.schema import export_project_schema
from syntheticcx.config.settings import prepare_settings, resolve_model_settings
from syntheticcx.errors import ConfigurationError


@pytest.fixture
def document() -> dict[str, Any]:
    return yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())


@pytest.mark.parametrize(
    ("path", "value", "location"),
    [
        (("version",), "2", "version"),
        (("includes",), "other.yaml", "includes"),
        (("execution", "max_turns"), 0, "execution.max_turns"),
        (("execution", "model_timeout_seconds"), float("inf"), "execution.model_timeout"),
        (("persona", "traits"), {"patience": 1.1}, "persona.traits.patience"),
        (("channel", "type"), "unknown", "channel.type"),
        (("channel", "send_strategy"), {"type": "click"}, "channel.send_strategy"),
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

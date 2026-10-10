"""Configuration validation tests."""

from pathlib import Path

import pytest

from syntheticcx.config.loader import load_config, load_config_text
from syntheticcx.config.sanitize import sanitized_effective_config
from syntheticcx.config.schema import export_project_schema
from syntheticcx.config.settings import resolve_model_settings
from syntheticcx.errors import ConfigurationError


def test_loads_the_canonical_example() -> None:
    config = load_config(Path("examples/telecom/scenario.yaml"))

    assert config.effective_evaluation_model.model == "gpt-5.4-mini"


def test_rejects_a_non_mapping_document() -> None:
    with pytest.raises(ConfigurationError, match="root document must be a mapping"):
        load_config_text("- item")


def test_rejects_embedded_credentials() -> None:
    source = Path("examples/telecom/scenario.yaml").read_text(encoding="utf-8")
    source = source.replace("  model: gpt-5.4-mini", "  model: gpt-5.4-mini\n  api_key: leaked")

    with pytest.raises(ConfigurationError, match="Extra inputs are not permitted"):
        load_config_text(source)


def test_resolves_the_explicit_credential_only() -> None:
    config = load_config(Path("examples/telecom/scenario.yaml"))

    settings = resolve_model_settings(config, {"OPENAI_API_KEY": "test-secret"})

    assert settings.credential.get_secret_value() == "test-secret"


def test_missing_credential_is_an_actionable_error() -> None:
    config = load_config(Path("examples/telecom/scenario.yaml"))

    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        resolve_model_settings(config, {})


def test_sanitized_output_excludes_resolved_credentials() -> None:
    config = load_config(Path("examples/telecom/scenario.yaml"))

    assert "test-secret" not in str(sanitized_effective_config(config))


def test_schema_export_is_deterministic(tmp_path: Path) -> None:
    destination = tmp_path / "project.schema.json"

    first = export_project_schema(destination).read_text(encoding="utf-8")
    second = export_project_schema(destination).read_text(encoding="utf-8")

    assert first == second

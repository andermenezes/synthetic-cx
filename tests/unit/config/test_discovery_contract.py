"""Discovery draft and guide boundaries remain independent of browser/model I/O."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from syntheticcx.cli import app
from syntheticcx.config.loader import load_config_with_source_hash, load_discovery_draft
from syntheticcx.errors import ConfigurationError


def example() -> dict[str, Any]:
    return yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())  # type: ignore[no-any-return]


def guide(channel: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": "1",
        "channel": channel,
        "validation": {
            "status": "verified",
            "verified_at": "2026-09-28T12:00:00Z",
            "target_origin": "http://127.0.0.1:8000",
            "observed_behaviors": ["open", "send", "receive", "completion"],
        },
    }


def test_draft_is_strict_and_valid_without_channel(tmp_path: Path) -> None:
    document = example()
    document.pop("channel")
    path = tmp_path / "draft.yaml"
    path.write_text(yaml.safe_dump(document))
    assert load_discovery_draft(path).scenario.id == document["scenario"]["id"]
    document["includes"] = "other.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ConfigurationError, match="includes"):
        load_discovery_draft(path)
    document.pop("includes")
    document["channel_guide"] = "guide.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ConfigurationError, match="channel_guide"):
        load_discovery_draft(path)


def test_referenced_guide_materializes_existing_run_config(tmp_path: Path) -> None:
    document = example()
    channel = document.pop("channel")
    document["channel_guide"] = "guides/chat.yaml"
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(document))
    location = tmp_path / "guides" / "chat.yaml"
    location.parent.mkdir()
    content = yaml.safe_dump(guide(channel))
    location.write_text(content)

    loaded = load_config_with_source_hash(path)
    assert str(loaded.config.channel.url) == channel["url"]
    assert loaded.config.channel.input_locator.value == "chat-input"
    assert loaded.source_content_hash == sha256(path.read_bytes()).hexdigest()
    assert loaded.guide_source_hash == sha256(location.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"version": "2"}, "version"),
        ({"unexpected": "value"}, "unexpected"),
        ({"validation": {"status": "draft"}}, "validation.status"),
        ({"validation": {"target_origin": "https://other.example"}}, "target_origin"),
        ({"validation": {"observed_behaviors": ["open", "send"]}}, "observed_behaviors"),
        ({"channel": {"input_locator": {"strategy": "css", "value": "//input"}}}, "XPath"),
    ],
)
def test_guide_rejects_unverified_or_invalid_controls(
    tmp_path: Path, change: dict[str, Any], error: str
) -> None:
    document = example()
    saved_guide = guide(document.pop("channel"))
    document["channel_guide"] = "guide.yaml"
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(document))
    for key, value in change.items():
        if isinstance(value, dict):
            saved_guide[key].update(value)
        else:
            saved_guide[key] = value
    (tmp_path / "guide.yaml").write_text(yaml.safe_dump(saved_guide))
    with pytest.raises(ConfigurationError, match=error):
        load_config_with_source_hash(path)


def test_conflict_missing_file_and_absolute_reference_fail_before_io(tmp_path: Path) -> None:
    document = example()
    document["channel_guide"] = "absent.yaml"
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ConfigurationError, match="channel_guide"):
        load_config_with_source_hash(path)
    document.pop("channel")
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ConfigurationError, match="channel_guide must name a local file"):
        load_config_with_source_hash(path)
    document["channel_guide"] = str(tmp_path / "guide.yaml")
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ConfigurationError, match="relative path"):
        load_config_with_source_hash(path)


def test_new_schemas_accept_only_expected_document_shapes() -> None:
    document = example()
    channel = document.pop("channel")
    draft_schema = Draft202012Validator(
        json.loads(Path("schemas/discovery-draft.schema.json").read_text())
    )
    guide_schema = Draft202012Validator(
        json.loads(Path("schemas/channel-guide.schema.json").read_text())
    )
    project_schema = Draft202012Validator(
        json.loads(Path("schemas/project.schema.json").read_text())
    )
    assert draft_schema.is_valid(document)
    document["channel_guide"] = "chat.yaml"
    assert project_schema.is_valid(document)
    assert guide_schema.is_valid(guide(channel))
    document["channel"] = channel
    assert not project_schema.is_valid(document)
    assert not draft_schema.is_valid(document)


def test_discover_cli_never_publishes_after_failed_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = example()
    document.pop("channel")
    path = tmp_path / "draft.yaml"
    path.write_text(yaml.safe_dump(document))
    output = tmp_path / "chat.yaml"
    runner = CliRunner()
    from syntheticcx import cli as cli_module
    from syntheticcx.discovery.explore import DiscoveryCandidate

    async def candidate(*_args: Any, **_kwargs: Any) -> DiscoveryCandidate:
        return DiscoveryCandidate(channel=example()["channel"], open_steps=1)

    async def failed_verification(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("controlled verification failure")

    monkeypatch.setattr(cli_module, "discover_candidate", candidate)
    monkeypatch.setattr(cli_module, "verify_candidate", failed_verification)
    valid = runner.invoke(
        app, ["discover", str(path), "--url", "https://example.com", "--output", str(output)]
    )
    assert valid.exit_code == 3
    assert "verification failed" in valid.output
    assert not output.exists()
    invalid = runner.invoke(
        app, ["discover", str(path), "--url", "file:///secret", "--output", str(output)]
    )
    assert invalid.exit_code == 2
    assert not output.exists()

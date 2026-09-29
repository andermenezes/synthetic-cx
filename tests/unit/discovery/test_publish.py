"""Guide publication does not silently overwrite or misreport partial writes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from syntheticcx import cli
from syntheticcx.config.models import ChannelGuide
from syntheticcx.discovery import publish
from syntheticcx.errors import ConfigurationError


def inputs(tmp_path: Path) -> tuple[Path, ChannelGuide]:
    source: dict[str, Any] = yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())
    channel = source.pop("channel")
    path = tmp_path / "draft.yaml"
    path.write_text(yaml.safe_dump(source))
    return path, ChannelGuide.model_validate(
        {
            "version": "1",
            "channel": channel,
            "validation": {
                "status": "verified",
                "verified_at": "2026-09-28T12:00:00Z",
                "target_origin": "http://127.0.0.1:8000",
                "observed_behaviors": ["open", "send", "receive", "completion"],
            },
        }
    )


def test_partial_scenario_write_preserves_source_and_allows_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, guide = inputs(tmp_path)
    guide_path = tmp_path / "guide.yaml"
    original = source.read_bytes()
    writer = publish._write_new

    def fail_completed(path: Path, data: bytes) -> None:
        if path.name == "draft.completed.yaml":
            raise ConfigurationError("publish discovery", "could not create output file")
        writer(path, data)

    monkeypatch.setattr(publish, "_write_new", fail_completed)
    with pytest.raises(ConfigurationError):
        publish.publish_guide(source, guide_path, guide, in_place=False)
    assert source.read_bytes() == original
    assert not guide_path.exists()
    assert not (tmp_path / "draft.completed.yaml").exists()
    monkeypatch.setattr(publish, "_write_new", writer)
    assert publish.publish_guide(source, guide_path, guide, in_place=False).exists()


def test_interactive_mode_requires_an_explicit_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.typer, "prompt", lambda *_args, **_kwargs: "i")
    assert cli._choose_in_place(True)
    assert not cli._choose_in_place(False)
    monkeypatch.setattr(cli.typer, "prompt", lambda *_args, **_kwargs: "c")
    assert not cli._choose_in_place(True)
    monkeypatch.setattr(cli.typer, "prompt", lambda *_args, **_kwargs: "other")
    with pytest.raises(cli.typer.Exit) as caught:
        cli._choose_in_place(True)
    assert caught.value.exit_code == 2


def test_explicit_discovery_language_is_saved_in_completed_scenario(tmp_path: Path) -> None:
    source, guide = inputs(tmp_path)
    original = source.read_bytes()
    completed = publish.publish_guide(
        source, tmp_path / "guide.yaml", guide, in_place=False, language="pt-BR"
    )
    document = yaml.safe_load(completed.read_text())
    assert document["persona"]["communication"]["language"] == "pt-BR"
    assert source.read_bytes() == original

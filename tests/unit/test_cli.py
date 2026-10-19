from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from syntheticcx import __version__, cli
from syntheticcx.cli import app
from syntheticcx.errors import ChannelError


def test_create_profile_rejects_unsupported_channel_without_creating_directory(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "profile"
    result = CliRunner().invoke(
        app, ["create-profile", "--channel", "webchat", "--browser-profile", str(profile)]
    )
    assert result.exit_code == 2
    assert not profile.exists()


def test_create_profile_requires_interactive_authentication(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    result = CliRunner().invoke(
        app, ["create-profile", "--channel", "whatsapp_web", "--browser-profile", str(profile)]
    )
    assert result.exit_code == 2
    assert not profile.exists()


def test_version_is_available() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_help_advertises_run_command() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "SyntheticCX" in result.stdout
    assert "run" in result.stdout.lower()


def test_discover_credential_url_fails_before_model_browser_or_publication(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(cli, "load_discovery_draft", lambda _path: object())
    monkeypatch.setattr(
        cli, "discover_guide", lambda *_args, **_kwargs: pytest.fail("discovery started")
    )
    result = CliRunner().invoke(
        app,
        [
            "discover",
            "draft.yaml",
            "--url",
            "https://user:CANARY@chat.example",
            "--output",
            "guide.yaml",
        ],
    )
    assert result.exit_code == 2
    assert "URL credentials" in result.output
    assert "CANARY" not in result.output


def test_discover_help_advertises_headed_mode() -> None:
    result = CliRunner().invoke(app, ["discover", "--help"])

    assert result.exit_code == 0
    help_text = " ".join(result.stdout.split())
    assert "--headed" in help_text
    assert "--chat-hint" in help_text
    assert "--max-discovery-atte" in help_text
    assert "--verbose" in help_text
    assert "--language" in help_text
    assert "never retries" in help_text


def test_run_help_advertises_headed_mode() -> None:
    result = CliRunner().invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--headed" in result.stdout
    assert "--language" in result.stdout


def test_run_invalid_language_fails_before_loading_scenario() -> None:
    result = CliRunner().invoke(app, ["run", "missing.yaml", "--language", "pt_BR"])
    assert result.exit_code == 2
    assert "language must be en-US or pt-BR" in result.output


def test_discover_headed_forwards_mode_to_single_learning_replay_orchestrator(
    monkeypatch: Any,
) -> None:
    received: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        cli,
        "load_discovery_draft",
        lambda _path: SimpleNamespace(
            execution=object(),
            persona=SimpleNamespace(communication=SimpleNamespace(language="en-US")),
        ),
    )

    async def discover(*_args: Any, headed: bool, **_kwargs: Any) -> None:
        received.append(("learning_replay", headed))
        raise ChannelError("channel.start", "target navigation returned HTTP 403")

    monkeypatch.setattr(cli, "discover_guide", discover)
    result = CliRunner().invoke(
        app,
        [
            "discover",
            "draft.yaml",
            "--url",
            "https://example.test/",
            "--output",
            "guide.yaml",
            "--headed",
        ],
    )
    assert result.exit_code == 3
    assert received == [("learning_replay", True)]


def test_discover_language_passes_locale_without_inventing_explicit_probe_override(
    monkeypatch: Any,
) -> None:
    received: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        cli,
        "load_discovery_draft",
        lambda _path: SimpleNamespace(
            execution=object(),
            persona=SimpleNamespace(communication=SimpleNamespace(language="en-US")),
        ),
    )

    async def discover(
        *_args: Any, probe_message: str | None, language: str, **_kwargs: Any
    ) -> None:
        received.append((language, probe_message))
        raise ChannelError("discover probe", "test stopped before sending")

    monkeypatch.setattr(cli, "discover_guide", discover)
    result = CliRunner().invoke(
        app,
        [
            "discover",
            "draft.yaml",
            "--url",
            "https://example.test/",
            "--output",
            "guide.yaml",
            "--language",
            "pt-BR",
        ],
    )
    assert result.exit_code == 3
    assert received == [("pt-BR", None)]


def test_invalid_discovery_language_stops_before_browser(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        cli,
        "load_discovery_draft",
        lambda _path: SimpleNamespace(execution=object()),
    )
    result = CliRunner().invoke(
        app,
        [
            "discover",
            "draft.yaml",
            "--url",
            "https://example.test/",
            "--output",
            "guide.yaml",
            "--language",
            "pt_BR",
        ],
    )
    assert result.exit_code == 2
    assert "language must be en-US or pt-BR" in result.output

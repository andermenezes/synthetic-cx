from pathlib import Path
from types import SimpleNamespace
from typing import Any

from typer.testing import CliRunner

from syntheticcx import __version__, cli
from syntheticcx.cli import app
from syntheticcx.errors import ChannelError


def test_version_is_available() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_help_advertises_run_command() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "SyntheticCX" in result.stdout
    assert "run" in result.stdout.lower()


def test_discover_help_advertises_headed_mode() -> None:
    result = CliRunner().invoke(app, ["discover", "--help"])

    assert result.exit_code == 0
    help_text = " ".join(result.stdout.split())
    assert "--headed" in help_text
    assert "--chat-hint" in help_text
    assert "--max-discovery-atte" in help_text
    assert "--verbose" in help_text
    assert "does not retry" in help_text


def test_verbose_discovery_reports_safe_stage_and_owned_failure(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        cli,
        "load_discovery_draft",
        lambda _path: SimpleNamespace(execution=object()),
    )

    async def fail(*_args: Any, **_kwargs: Any) -> None:
        raise ChannelError("discover probe", "native channel verification timed out")

    monkeypatch.setattr(cli, "discover_candidate", fail)
    result = CliRunner().invoke(
        app,
        [
            "discover",
            str(Path("draft.yaml")),
            "--url",
            "https://example.test/chat",
            "--output",
            "guide.yaml",
            "--verbose",
        ],
    )

    assert result.exit_code == 3
    assert "[discover] validating draft and command options" in result.output
    assert "[discover] beginning discovery and control proposal" in result.output
    assert "failed: channel operation 'discover probe'" in result.output
    assert "native channel verification timed out" in result.output

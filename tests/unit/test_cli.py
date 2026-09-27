from typer.testing import CliRunner

from syntheticcx import __version__
from syntheticcx.cli import app


def test_version_is_available() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_help_advertises_run_command() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "SyntheticCX" in result.stdout
    assert "run" in result.stdout.lower()

"""Synchronous command boundary for one bounded, evidence-backed run."""

import asyncio
from pathlib import Path

import typer

from syntheticcx import __version__
from syntheticcx.application.run import execute_run
from syntheticcx.errors import ConfigurationError, SyntheticCXError
from syntheticcx.reporting.console import render_partial, render_report

app = typer.Typer(
    add_completion=False,
    help="SyntheticCX: simulate customers and test conversational journeys.",
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    """Print the installed package version when requested."""
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show the installed SyntheticCX version and exit.",
    ),
) -> None:
    """Expose package help and version."""


@app.command()
def run(scenario: Path = typer.Argument(..., help="Validated version-1 scenario YAML.")) -> None:
    """Run one scenario and persona through a configured web chat."""

    def created(run_id: str, directory: Path) -> None:
        typer.echo(f"Run ID: {run_id} | Evidence: {directory}")
        typer.echo("Running conversation and evaluation...")

    try:
        result = asyncio.run(execute_run(scenario, on_created=created))
    except ConfigurationError as error:
        typer.echo(f"Configuration error: {error}", err=True)
        raise typer.Exit(2) from None
    except KeyboardInterrupt:
        typer.echo("Cancelled after resource cleanup.", err=True)
        raise typer.Exit(130) from None
    except SyntheticCXError as error:
        typer.echo(f"{error.category} operation failed ({error.operation}).", err=True)
        raise typer.Exit(3) from None
    except Exception:
        typer.echo("Execution failed before a final report was available.", err=True)
        raise typer.Exit(3) from None
    if result.report is not None:
        typer.echo(render_report(result.report, report_omitted=result.report_omitted))
    else:
        reason = {
            "cancelled": "cancelled after resource cleanup",
            "persistence_error": "storage failure; final report unavailable",
        }.get(result.partial_reason or "", "final report unavailable")
        typer.echo(render_partial(result.run_id, result.artifact_location, reason))
    raise typer.Exit(result.exit_code)

"""Synchronous command boundary for one bounded, evidence-backed run."""

import asyncio
import sys
from pathlib import Path

import typer
from pydantic import AnyHttpUrl, TypeAdapter, ValidationError

from syntheticcx import __version__
from syntheticcx.application.run import execute_run
from syntheticcx.config.loader import load_discovery_draft
from syntheticcx.discovery.publish import publish_guide
from syntheticcx.discovery.run import discover_candidate
from syntheticcx.discovery.verify import verify_candidate
from syntheticcx.errors import ConfigurationError, SyntheticCXError
from syntheticcx.reporting.console import render_partial, render_report

app = typer.Typer(
    add_completion=False,
    help="SyntheticCX: simulate customers and test conversational journeys.",
    no_args_is_help=True,
)


def _choose_in_place(interactive: bool) -> bool:
    """Never alter the source without an interactive, explicit choice."""
    if not interactive:
        return False
    try:
        choice = str(
            typer.prompt("Save completed scenario as [c]opy or edit [i]n place?", default="c")
        )
    except (EOFError, KeyboardInterrupt):
        typer.echo("Discovery cancelled; no guide was written.", err=True)
        raise typer.Exit(130) from None
    if choice.lower() not in {"c", "i"}:
        typer.echo("Configuration error: choose c or i; no guide was written.", err=True)
        raise typer.Exit(2)
    return choice.lower() == "i"


@app.command()
def discover(
    scenario: Path = typer.Argument(..., help="Scenario draft without channel controls."),
    url: str = typer.Option(..., "--url", help="Customer-facing chat URL to explore."),
    output: Path = typer.Option(..., "--output", help="Path for the verified channel guide."),
    probe_message: str = typer.Option(
        "Synthetic test: please reply with a greeting.",
        "--probe-message",
        help="Non-destructive message for native channel validation.",
    ),
    discovery_model_config: Path | None = typer.Option(
        None, "--discovery-model-config", help="Optional discovery-only model configuration."
    ),
    headed: bool = typer.Option(
        False,
        "--headed",
        help="Show both MCP exploration and fresh native verification browsers.",
    ),
    chat_hint: str | None = typer.Option(
        None,
        "--chat-hint",
        help="Visible chat-button or hover-tooltip text to look for.",
    ),
    max_discovery_attempts: int = typer.Option(
        3,
        "--max-discovery-attempts",
        min=1,
        max=5,
        help="Model decisions per exploration snapshot (1–5; does not retry verification).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Show safe discovery progress and owned failure details.",
    ),
) -> None:
    """Discover and verify web-chat controls, then save a reusable guide."""

    def progress(message: str) -> None:
        typer.echo(f"[discover] {message}", err=True)

    try:
        if verbose:
            progress("validating draft and command options")
        draft = load_discovery_draft(scenario)
        TypeAdapter(AnyHttpUrl).validate_python(url)
        if not probe_message.strip():
            raise ConfigurationError("discover", "probe_message must not be empty")
        if chat_hint is not None and (not chat_hint.strip() or len(chat_hint) > 256):
            raise ConfigurationError("discover", "chat_hint must contain 1 to 256 characters")
        if output == scenario:
            raise ConfigurationError("discover", "guide output must differ from scenario input")
        if discovery_model_config is not None and not discovery_model_config.is_file():
            raise ConfigurationError("discover", "discovery model configuration must be a file")
    except ValidationError:
        typer.echo("Configuration error: discover URL must be HTTP or HTTPS.", err=True)
        raise typer.Exit(2) from None
    except ConfigurationError as error:
        typer.echo(f"Configuration error: {error}", err=True)
        raise typer.Exit(2) from None
    try:
        if verbose:
            progress("beginning discovery and control proposal")
        candidate = asyncio.run(
            discover_candidate(
                scenario,
                url,
                discovery_model_config=discovery_model_config,
                headed=headed,
                chat_hint=chat_hint,
                max_discovery_attempts=max_discovery_attempts,
                on_progress=progress if verbose else None,
            )
        )
        if verbose:
            progress("discovery candidate ready; beginning native verification")
        guide = asyncio.run(
            verify_candidate(
                candidate,
                draft.execution,
                probe_message,
                headed=headed,
                on_progress=progress if verbose else None,
            )
        )
    except ConfigurationError as error:
        typer.echo(f"Configuration error: {error}", err=True)
        raise typer.Exit(2) from None
    except KeyboardInterrupt:
        typer.echo("Discovery cancelled after resource cleanup.", err=True)
        raise typer.Exit(130) from None
    except SyntheticCXError as error:
        if verbose:
            progress(f"failed: {error.category} operation '{error.operation}': {error.message}")
        else:
            typer.echo("Discovery or native verification failed; no guide was written.", err=True)
        raise typer.Exit(3) from None
    except Exception:
        if verbose:
            progress("failed with an unexpected error; details suppressed for safety")
        else:
            typer.echo("Discovery or native verification failed; no guide was written.", err=True)
        raise typer.Exit(3) from None
    in_place = _choose_in_place(sys.stdin.isatty())
    try:
        if verbose:
            progress("publishing verified guide and completed scenario")
        completed = publish_guide(scenario, output, guide, in_place=in_place)
    except ConfigurationError as error:
        typer.echo(f"Configuration error: {error}; check output paths for partial files.", err=True)
        if output.exists():
            typer.echo(f"Guide path: {output}", err=True)
        raise typer.Exit(2) from None
    typer.echo(f"Verified channel guide: {output}")
    typer.echo(f"Completed scenario: {completed}")
    if verbose:
        progress("discovery completed successfully")


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
def run(
    scenario: Path = typer.Argument(..., help="Validated version-1 scenario YAML."),
    headed: bool = typer.Option(False, "--headed", help="Show the native browser during the run."),
) -> None:
    """Run one scenario and persona through a configured web chat."""

    def created(run_id: str, directory: Path) -> None:
        typer.echo(f"Run ID: {run_id} | Evidence: {directory}")
        typer.echo("Running conversation and evaluation...")

    try:
        result = asyncio.run(
            execute_run(scenario, on_created=created, headed=True)
            if headed
            else execute_run(scenario, on_created=created)
        )
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

"""Synchronous command boundary for one bounded, evidence-backed run."""

import asyncio
import json
import sys
from pathlib import Path
from typing import TypedDict

import typer
from pydantic import AnyHttpUrl, TypeAdapter, ValidationError

from syntheticcx import __version__
from syntheticcx.application.run import execute_run
from syntheticcx.config.loader import load_discovery_draft
from syntheticcx.discovery.hints import launcher_id
from syntheticcx.discovery.publish import publish_guide
from syntheticcx.discovery.run import discover_guide
from syntheticcx.errors import ConfigurationError, SyntheticCXError
from syntheticcx.language import require_language
from syntheticcx.reporting.console import render_partial, render_report
from syntheticcx.runtime.behavior_replay import (
    BehaviorReplayError,
    replay_behavior,
    write_behavior_replay,
)


class _DiscoveryChannelOptions(TypedDict, total=False):
    channel_type: str
    browser_profile: Path | None
    recipient: str | None


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


@app.command("create-profile")
def create_profile(
    channel: str = typer.Option(..., "--channel", help="Session channel: whatsapp_web."),
    browser_profile: Path | None = typer.Option(
        None, "--browser-profile", help="Dedicated profile directory to create or reopen."
    ),
) -> None:
    """Create a dedicated browser profile and log in manually."""
    if channel != "whatsapp_web":
        typer.echo("Configuration error: supported profile channels: whatsapp_web.", err=True)
        raise typer.Exit(2)
    if not sys.stdin.isatty():
        typer.echo(
            "Configuration error: profile creation requires an interactive terminal.", err=True
        )
        raise typer.Exit(2)
    profile = browser_profile or Path.home() / ".syntheticcx" / "whatsapp-profile"

    def confirm() -> None:
        input("Log in to WhatsApp Web, then press Enter here to save and close: ")

    try:
        from syntheticcx.application.profile import create_browser_profile

        saved = create_browser_profile(profile, confirm=confirm)
    except ConfigurationError as error:
        typer.echo(f"Configuration error: {error}", err=True)
        raise typer.Exit(2) from None
    except (KeyboardInterrupt, EOFError):
        typer.echo("Profile setup cancelled after browser cleanup.", err=True)
        raise typer.Exit(130) from None
    except Exception:
        typer.echo(
            "Profile setup failed. Ensure Chromium is installed and the profile is not in use.",
            err=True,
        )
        raise typer.Exit(3) from None
    typer.echo(f"Browser profile saved: {saved}")
    typer.echo("Login was manual; authentication has not been automatically verified.")
    typer.echo(f'Use --browser-profile "{saved}" for discovery and runs.')


@app.command()
def discover(
    scenario: Path = typer.Argument(..., help="Scenario draft without channel controls."),
    url: str = typer.Option(..., "--url", help="Customer-facing chat URL to explore."),
    output: Path = typer.Option(..., "--output", help="Path for the verified channel guide."),
    probe_message: str | None = typer.Option(
        None,
        "--probe-message",
        help="Explicit text probe override, kept in memory for learning and exact replay.",
    ),
    language: str | None = typer.Option(
        None, "--language", help="Customer-facing language: en-US or pt-BR."
    ),
    discovery_model_config: Path | None = typer.Option(
        None, "--discovery-model-config", help="Optional discovery-only model configuration."
    ),
    headed: bool = typer.Option(
        False,
        "--headed",
        help="Show native learning and fresh recipe-replay browsers.",
    ),
    chat_hint: str | None = typer.Option(
        None,
        "--chat-hint",
        help="Visible chat-button/tooltip text, or id=<DOM-ID> for an explicit launcher button.",
    ),
    max_discovery_attempts: int = typer.Option(
        3,
        "--max-discovery-attempts",
        min=1,
        max=5,
        help="Corrections per discovery decision (1–5); never retries conversational actions.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Show safe discovery progress and owned failure details.",
    ),
    channel: str = typer.Option(
        "webchat", "--channel", help="Entry channel: webchat or whatsapp_web."
    ),
    browser_profile: Path | None = typer.Option(
        None, "--browser-profile", help="Dedicated authenticated WhatsApp browser profile."
    ),
    recipient: str | None = typer.Option(
        None,
        "--recipient",
        help="WhatsApp business recipient in international digits; overrides the environment.",
    ),
) -> None:
    """Discover a website chat entry and verify a reusable channel guide."""

    def progress(message: str) -> None:
        typer.echo(f"[discover] {message}", err=True)

    try:
        if verbose:
            progress("validating draft and command options")
        if channel not in {"webchat", "whatsapp_web"}:
            raise ConfigurationError(
                "discover channel", "supported channels: webchat, whatsapp_web"
            )
        if channel == "whatsapp_web" and browser_profile is None:
            raise ConfigurationError("WhatsApp profile", "--browser-profile is required")
        if channel == "webchat" and browser_profile is not None:
            raise ConfigurationError(
                "browser profile", "browser profiles are only used by WhatsApp"
            )
        if channel == "webchat" and recipient is not None:
            raise ConfigurationError("recipient", "--recipient is only used by WhatsApp")
        draft = load_discovery_draft(scenario)
        target_url = TypeAdapter(AnyHttpUrl).validate_python(url)
        if target_url.username is not None or target_url.password is not None:
            raise ConfigurationError("discover URL", "URL credentials are not permitted")
        selected_language = (
            require_language(language)
            if language is not None
            else draft.persona.communication.language
        )
        if probe_message is not None and not probe_message.strip():
            raise ConfigurationError("discover", "probe_message must not be empty")
        if chat_hint is not None and (not chat_hint.strip() or len(chat_hint) > 256):
            raise ConfigurationError("discover", "chat_hint must contain 1 to 256 characters")
        launcher_id(chat_hint)
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
            progress("beginning channel discovery and fresh browser verification")
        channel_options: _DiscoveryChannelOptions = {}
        if channel == "whatsapp_web":
            channel_options = {"channel_type": channel, "browser_profile": browser_profile}
            if recipient is not None:
                channel_options["recipient"] = recipient
        guide = asyncio.run(
            discover_guide(
                scenario,
                url,
                discovery_model_config=discovery_model_config,
                headed=headed,
                chat_hint=chat_hint,
                max_discovery_attempts=max_discovery_attempts,
                probe_message=probe_message,
                language=selected_language,
                on_progress=progress if verbose else None,
                **channel_options,
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
        completed = publish_guide(
            scenario,
            output,
            guide,
            in_place=in_place,
            language=selected_language if language is not None else None,
        )
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
    scenario: Path = typer.Argument(..., help="Validated version-1 journey YAML."),
    headed: bool = typer.Option(False, "--headed", help="Show the native browser during the run."),
    language: str | None = typer.Option(
        None, "--language", help="Override the persona language: en-US or pt-BR."
    ),
    browser_profile: Path | None = typer.Option(
        None, "--browser-profile", help="Dedicated authenticated WhatsApp browser profile."
    ),
    recipient: str | None = typer.Option(
        None,
        "--recipient",
        help="WhatsApp business recipient in international digits; overrides the environment.",
    ),
) -> None:
    """Run one synthetic customer journey and evaluate its conversation goal."""

    def created(run_id: str, directory: Path) -> None:
        typer.echo(f"Run ID: {run_id} | Evidence: {directory}")
        typer.echo("Running conversation and evaluation...")

    try:
        selected_language = require_language(language) if language is not None else None
        result = asyncio.run(
            execute_run(
                scenario,
                on_created=created,
                headed=headed,
                language=selected_language,
                browser_profile=browser_profile,
                recipient=recipient,
            )
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
    for shutdown_error in result.shutdown_errors:
        typer.echo(
            f"Infrastructure shutdown failure ({shutdown_error.operation}); "
            "any committed report describes the journey before this failure.",
            err=True,
        )
    raise typer.Exit(result.exit_code)


@app.command("replay-behavior")
def replay_behavior_command(
    run_directory: Path = typer.Argument(..., help="Read-only directory containing run evidence."),
    output: Path = typer.Option(..., "--output", help="New replay JSON file; never overwritten."),
    rules: Path | None = typer.Option(
        None, "--rules", help="Optional version-1 rules-only YAML for counterfactual behavior."
    ),
) -> None:
    """Replay committed behavior offline; this is not browser replay or evaluation."""
    try:
        if output.resolve().is_relative_to(run_directory.resolve()):
            raise BehaviorReplayError(
                "output_error", "output must be outside the source run directory"
            )
        if rules is not None and output.resolve() == rules.resolve():
            raise BehaviorReplayError("output_error", "output must differ from the rules input")
        result = replay_behavior(run_directory, rules=rules)
        write_behavior_replay(result, output)
    except BehaviorReplayError as error:
        typer.echo(
            json.dumps({"kind": error.kind, "message": error.message}, sort_keys=True),
            err=True,
        )
        raise typer.Exit(3 if error.kind in {"output_error", "output_exists"} else 2) from None
    except Exception:
        typer.echo(
            json.dumps(
                {
                    "kind": "invalid_source",
                    "message": "behavior replay could not read valid evidence",
                }
            ),
            err=True,
        )
        raise typer.Exit(2) from None
    typer.echo(f"Behavior replay: {output}")
    typer.echo(f"Mode: {result.mode} | Completeness: {result.completeness}")
    if result.comparison_findings:
        typer.echo(f"Behavioral mismatches: {len(result.comparison_findings)}", err=True)
        raise typer.Exit(4)
    if result.completeness == "prefix":
        typer.echo(result.prefix_reason, err=True)
        raise typer.Exit(5)
    typer.echo(
        "Recorded behavior verified."
        if result.mode == "verification"
        else (
            "Counterfactual behavior calculated; recorded conversation is not an alternate outcome."
        )
    )

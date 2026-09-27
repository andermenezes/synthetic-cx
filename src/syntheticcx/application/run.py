"""Build and own one configured run without moving policies into the CLI."""

from __future__ import annotations

import asyncio
import os
import platform
import socket
import sys
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from syntheticcx import __version__
from syntheticcx.adapters.channel.playwright import PlaywrightChannel
from syntheticcx.adapters.llm import OpenAICompatibleProvider, OpenAIProvider
from syntheticcx.adapters.llm.preflight import CapabilityPreflight, preflight_models
from syntheticcx.adapters.persona.tinytroupe import TinyTroupePersonaSimulator
from syntheticcx.adapters.storage.filesystem import FilesystemStorage
from syntheticcx.adapters.storage.recovery import Recovery
from syntheticcx.config.loader import load_config_with_source_hash
from syntheticcx.config.models import OpenAICompatibleModelConfig, ProjectConfig, TelemetryConfig
from syntheticcx.config.sanitize import sanitized_effective_config
from syntheticcx.config.settings import prepare_settings, requires_semantic_evaluation
from syntheticcx.domain.artifacts import ConfigArtifact, MetadataArtifact, Run
from syntheticcx.domain.models import (
    ModelIdentity,
    PersonaDefinition,
    Report,
    RunMetadata,
    RunOwner,
    Scenario,
    TokenUsage,
)
from syntheticcx.errors import ConfigurationError, PersistenceError
from syntheticcx.evaluation.service import EvaluationService
from syntheticcx.privacy import Redactor
from syntheticcx.privacy.prompt_capture import PromptCapture
from syntheticcx.reporting.materialize import build_projections
from syntheticcx.runtime.clock import RealClock, RealWaitController
from syntheticcx.runtime.journal import RunJournal
from syntheticcx.runtime.model_metadata import with_model_evidence
from syntheticcx.runtime.owner import run_evaluated
from syntheticcx.runtime.result import EvaluatedRunResult
from syntheticcx.telemetry.run import RunTelemetry, span


@dataclass(frozen=True)
class RunExecution:
    """The durable report, or only confirmed partial-evidence location."""

    exit_code: int
    report: Report | None
    run_id: str | None
    artifact_location: str | None
    report_omitted: bool = False
    partial_reason: str | None = None


class _LocalProcessProbe:
    """Linux /proc identity check; other platforms cannot prove a dead owner."""

    def supported(self) -> bool:
        return Path("/proc/self/stat").is_file()

    def process_start_evidence(self, pid: int) -> str | None:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
            return stat.rsplit(")", 1)[1].split()[19]
        except (OSError, IndexError):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return None
            except PermissionError:
                pass
            # Live or ambiguous ownership must never be marked dead.
            return "unreadable-live-process"


def _lock_hash() -> str | None:
    for parent in Path(__file__).resolve().parents:
        lock = parent / "uv.lock"
        try:
            return sha256(lock.read_bytes()).hexdigest()
        except FileNotFoundError:
            continue
        except OSError:
            return None
    return None


def _initial_run(
    config: ProjectConfig, run_id: str, preflight: CapabilityPreflight, clock: RealClock
) -> Run:
    probe = _LocalProcessProbe()
    metadata = RunMetadata(
        configuration_version=config.version,
        framework_version=__version__,
        dependency_lock_hash=_lock_hash(),
        persona_model=ModelIdentity(provider=config.model.provider, model=config.model.model),
        evaluation_model=ModelIdentity(
            provider=config.effective_evaluation_model.provider,
            model=config.effective_evaluation_model.model,
        ),
        usage=TokenUsage(unavailable_reason="provider usage not yet available"),
        runtime_platform=platform.platform(),
        owner=RunOwner(
            owner_uuid=str(uuid4()),
            hostname=socket.gethostname(),
            pid=os.getpid(),
            process_start_evidence=probe.process_start_evidence(os.getpid()) or str(uuid4()),
        ),
    )
    metadata = with_model_evidence(
        metadata,
        config,
        verified_capabilities=preflight.verified_capabilities,
        unavailable_capabilities=preflight.unavailable_capabilities,
    )
    return Run(
        id=run_id,
        scenario_id=config.scenario.id,
        persona_id=config.persona.id,
        started_at=clock.now(),
        lifecycle_status="in_progress",
        metadata=metadata,
    )


async def execute_run(
    path: Path, *, on_created: Callable[[str, Path], None] | None = None
) -> RunExecution:
    """Validate, probe, compose, execute, and read the committed report."""
    loaded = load_config_with_source_hash(path)
    config = loaded.config
    settings = prepare_settings(config)
    secrets = (
        settings.persona.credential.get_secret_value(),
        settings.evaluation.credential.get_secret_value(),
    )
    redactor = Redactor(config.privacy, resolved_secrets=secrets)
    prompt_capture = (
        PromptCapture(redactor, config.scenario) if config.privacy.capture_prompts else None
    )
    config_artifact = ConfigArtifact(
        source_content_hash=loaded.source_content_hash,
        effective_config=sanitized_effective_config(config, resolved_secrets=secrets),
    )
    preflight = await preflight_models(config, settings)
    clock = RealClock()
    wait = RealWaitController()
    run_id = f"run-{uuid4().hex}"
    run_root = Path(config.storage.run_root)
    directory = run_root / run_id
    # Recovery cannot rewrite an active or ambiguously owned run.
    Recovery(run_root, _LocalProcessProbe(), socket.gethostname()).scan()
    run = _initial_run(config, run_id, preflight, clock)
    storage = FilesystemStorage(
        run_root,
        redactor,
        timeout_seconds=config.execution.persistence_timeout_seconds,
        materialize_conversation=config.storage.materialize_conversation,
        materialize_evaluation=config.storage.materialize_evaluation,
        materialize_report=config.storage.materialize_report,
        project_config=config,
    )
    try:
        telemetry = RunTelemetry(
            config.telemetry,
            run_id,
            directory,
            redactor,
            remote_timeout=config.execution.close_timeout_seconds,
        )
    except Exception:
        if config.telemetry.remote_export:
            raise ConfigurationError(
                "configure telemetry", "remote telemetry exporter could not be configured"
            ) from None
        telemetry = RunTelemetry(
            TelemetryConfig(enabled=False),
            run_id,
            directory,
            redactor,
            remote_timeout=config.execution.close_timeout_seconds,
        )
    provider: OpenAIProvider | OpenAICompatibleProvider | None = None
    channel: PlaywrightChannel | None = None
    simulator: TinyTroupePersonaSimulator | None = None
    acknowledged = False
    result: EvaluatedRunResult | None = None
    unexpected_failure = False

    def created(identifier: str) -> None:
        nonlocal acknowledged
        acknowledged = True
        if on_created is not None:
            on_created(identifier, directory)

    try:
        if requires_semantic_evaluation(config):
            model = config.effective_evaluation_model
            if isinstance(model, OpenAICompatibleModelConfig):
                provider = OpenAICompatibleProvider(
                    model,
                    settings.evaluation,
                    timeout_seconds=config.execution.model_timeout_seconds,
                    prompt_capture=prompt_capture,
                )
            else:
                provider = OpenAIProvider(
                    model,
                    settings.evaluation,
                    timeout_seconds=config.execution.model_timeout_seconds,
                    prompt_capture=prompt_capture,
                )
        journal = RunJournal(run.id, storage, redactor, clock)
        evaluation = EvaluationService(
            journal=journal,
            wait=wait,
            evaluator_timeout_seconds=config.execution.evaluator_timeout_seconds,
            redactor=redactor,
            provider=provider,
        )
        channel = PlaywrightChannel(config.channel, config.execution)
        simulator = TinyTroupePersonaSimulator(
            config.model,
            settings.persona,
            timeout_seconds=config.execution.model_timeout_seconds,
            prompt_capture=prompt_capture,
        )
        with telemetry.activate(), span("run.initialize"):
            result = await run_evaluated(
                run=run,
                config_artifact=config_artifact,
                scenario=Scenario.model_validate(config.scenario.model_dump()),
                persona=PersonaDefinition.model_validate(config.persona.model_dump()),
                channel=channel,
                simulator=simulator,
                storage=storage,
                journal=journal,
                clock=clock,
                wait=wait,
                limits=config.execution,
                evaluation=evaluation,
                redactor=redactor,
                evaluation_provider=provider,
                project_config=config,
                on_created=created,
            )
    except asyncio.CancelledError:
        raise
    except Exception:
        unexpected_failure = True
    finally:
        if result is None:
            # A constructor or an unexpected owner exception has no completed
            # result. All owned close operations are idempotent and bounded.
            for resource in (channel, simulator, provider, storage):
                if resource is not None:
                    try:
                        await wait.run(resource.close(), config.execution.close_timeout_seconds)
                    except (Exception, asyncio.CancelledError):
                        pass
        try:
            await telemetry.close(
                created=acknowledged, timeout=config.execution.close_timeout_seconds
            )
        except (Exception, asyncio.CancelledError):
            # Supplemental telemetry cannot change the committed journey outcome.
            sys.stderr.write(f"Telemetry shutdown failed for run {run_id}.\n")

    if unexpected_failure or result is None:
        return RunExecution(
            3,
            None,
            run_id if acknowledged else None,
            str(directory) if acknowledged else None,
        )
    if not result.finalized:
        return RunExecution(
            result.exit_code,
            None,
            run_id if acknowledged else None,
            str(directory) if acknowledged else None,
            partial_reason=result.conversation.termination_reason,
        )
    try:
        metadata = MetadataArtifact.model_validate_json((directory / "metadata.json").read_bytes())
        if metadata.lifecycle_status != "finalized" or metadata.outcome != result.outcome:
            raise ValueError("final metadata and outcome disagree")
        if config.storage.materialize_report:
            report = Report.model_validate_json((directory / "report.json").read_bytes())
        else:
            criteria = [
                redactor.redact_criterion(item, config.scenario)
                for item in config.scenario.success_criteria
            ]
            report = build_projections(
                run_id,
                list(result.committed_events),
                criteria,
                artifact_location=str(directory),
                artifacts=metadata.artifacts,
                materialize_conversation=config.storage.materialize_conversation,
                materialize_evaluation=config.storage.materialize_evaluation,
                materialize_report=True,
            ).report
        if report is None or report.run_id != run_id or report.outcome != metadata.outcome:
            raise ValueError("final report and metadata disagree")
    except (OSError, ValueError, PersistenceError):
        return RunExecution(3, None, run_id, str(directory))
    return RunExecution(
        result.exit_code, report, run_id, str(directory), not config.storage.materialize_report
    )

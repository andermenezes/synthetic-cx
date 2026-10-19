"""Build and own one configured run without moving policies into the CLI."""

from __future__ import annotations

import asyncio
import os
import platform
import socket
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from syntheticcx import __version__
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
    BehavioralDetectorMetadata,
    ModelIdentity,
    PersonaDefinition,
    Report,
    RunMetadata,
    RunOwner,
    Scenario,
    TokenUsage,
)
from syntheticcx.domain.protocols import ChannelAdapter
from syntheticcx.errors import ConfigurationError, PersistenceError
from syntheticcx.evaluation.service import EvaluationService
from syntheticcx.language import Language, require_language
from syntheticcx.privacy import Redactor
from syntheticcx.privacy.prompt_capture import PromptCapture
from syntheticcx.reporting.materialize import build_projections
from syntheticcx.runtime.behavior_signals import DetectorRegistration, prepare_behavior
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
    config: ProjectConfig,
    run_id: str,
    preflight: CapabilityPreflight,
    clock: RealClock,
    behavior_detectors: tuple[BehavioralDetectorMetadata, ...] = (),
) -> Run:
    probe = _LocalProcessProbe()
    metadata = RunMetadata(
        configuration_version=config.version,
        behavior_detectors=behavior_detectors,
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
    path: Path,
    *,
    on_created: Callable[[str, Path], None] | None = None,
    headed: bool = False,
    language: Language | None = None,
    behavior_registry: Mapping[str, DetectorRegistration] | None = None,
    browser_profile: Path | None = None,
    recipient: str | None = None,
) -> RunExecution:
    """Validate, probe, compose, execute, and read the committed report."""
    loaded = load_config_with_source_hash(path)
    config = loaded.config
    whatsapp_recipient: str | None = None
    if config.channel.type == "whatsapp_web_playwright":
        from syntheticcx.adapters.channel.whatsapp import (
            resolve_recipient,
            validate_browser_profile,
        )

        if browser_profile is None:
            raise ConfigurationError("WhatsApp profile", "--browser-profile is required")
        browser_profile = validate_browser_profile(browser_profile)
        whatsapp_recipient = resolve_recipient(config.channel, recipient)
        if not config.channel.navigation_steps:
            raise ConfigurationError(
                "WhatsApp channel", "discover the website entry before running"
            )
        if config.storage.store_screenshots or config.storage.store_video:
            raise ConfigurationError(
                "WhatsApp privacy", "account-wide screenshots and video are not supported"
            )
    elif browser_profile is not None:
        raise ConfigurationError("browser profile", "browser profiles are only used by WhatsApp")
    elif recipient is not None:
        raise ConfigurationError("recipient", "--recipient is only used by WhatsApp")
    if language is not None:
        selected = require_language(language)
        config = config.model_copy(
            update={
                "persona": config.persona.model_copy(
                    update={
                        "communication": config.persona.communication.model_copy(
                            update={"language": selected}
                        )
                    }
                )
            }
        )
    prepared_behavior = prepare_behavior(
        config.behavior, config.persona.behavioral_rules, behavior_registry
    )
    settings = prepare_settings(config, behavior_registry=behavior_registry)
    secrets: tuple[str, ...] = (
        settings.persona.credential.get_secret_value(),
        settings.evaluation.credential.get_secret_value(),
    )
    if config.channel.type == "whatsapp_web_playwright":
        assert whatsapp_recipient is not None
        recipient_value = (
            recipient
            if recipient is not None
            else os.environ.get(config.channel.recipient_environment_variable, "")
        )
        secrets += (whatsapp_recipient, recipient_value)
    redactor = Redactor(
        config.privacy,
        resolved_secrets=secrets,
        sensitive_context={
            "persona": config.persona.model_dump(mode="json"),
            "starting_context": config.scenario.starting_context,
            "knowledge": config.scenario.knowledge,
        },
    )
    prompt_capture = (
        PromptCapture(redactor, config.scenario, persona=config.persona)
        if config.privacy.capture_prompts
        else None
    )
    effective_config = sanitized_effective_config(config, resolved_secrets=secrets)
    try:
        original_persona = config.persona.model_dump(mode="json")
        safe_persona = effective_config["persona"]
        original_detectors = config.behavior.model_dump(mode="json")["detectors"]
        safe_detectors = effective_config["behavior"]["detectors"]
        if (
            safe_persona["initial_state"] != original_persona["initial_state"]
            or [
                {key: value for key, value in rule.items() if key != "description"}
                for rule in safe_persona["behavioral_rules"]
            ]
            != [
                {key: value for key, value in rule.items() if key != "description"}
                for rule in original_persona["behavioral_rules"]
            ]
            or [{key: detector[key] for key in ("id", "type")} for detector in safe_detectors]
            != [{key: detector[key] for key in ("id", "type")} for detector in original_detectors]
            or any(
                redactor.redact_model(metadata) != metadata
                for metadata in prepared_behavior.metadata
            )
        ):
            raise ValueError("behavioral configuration identity changed")
    except Exception:
        raise ConfigurationError(
            "prepare behavior", "privacy filtering changes behavioral replay identities or rules"
        ) from None
    config_artifact = ConfigArtifact(
        source_content_hash=loaded.source_content_hash,
        channel_guide_source_hash=loaded.guide_source_hash,
        effective_config=effective_config,
        behavior_detectors=prepared_behavior.metadata,
    )
    behavior_detectors = prepared_behavior.instantiate()
    preflight = await preflight_models(config, settings)
    clock = RealClock()
    wait = RealWaitController()
    run_id = f"run-{uuid4().hex}"
    run_root = Path(config.storage.run_root)
    directory = run_root / run_id
    # Recovery cannot rewrite an active or ambiguously owned run.
    Recovery(run_root, _LocalProcessProbe(), socket.gethostname()).scan()
    run = _initial_run(config, run_id, preflight, clock, prepared_behavior.metadata)
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
    channel: ChannelAdapter | None = None
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
        if config.channel.type == "whatsapp_web_playwright":
            from syntheticcx.adapters.channel.whatsapp import WhatsAppWebChannel

            assert browser_profile is not None
            channel = WhatsAppWebChannel(
                config.channel,
                config.execution,
                browser_profile=browser_profile,
                recipient=whatsapp_recipient,
                headed=headed,
                action_authorizations=[
                    authorization.model_dump(mode="json")
                    for authorization in config.scenario.action_authorizations
                ],
            )
        else:
            from syntheticcx.adapters.channel.dynamic import DynamicPlaywrightChannel

            channel = DynamicPlaywrightChannel(
                config.channel,
                config.execution,
                headed=headed,
                action_authorizations=[
                    authorization.model_dump(mode="json")
                    for authorization in config.scenario.action_authorizations
                ],
            )
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
                behavior_detectors=behavior_detectors,
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

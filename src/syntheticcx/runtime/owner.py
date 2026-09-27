"""One run owner keeps the journal writable through evaluation and finalization."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from syntheticcx.config.models import ExecutionLimits, ProjectConfig
from syntheticcx.domain.artifacts import ConfigArtifact, Run
from syntheticcx.domain.events import (
    EvaluationCompleted,
    EvaluationCompletedEvent,
    OperationFailure,
    RunLifecycleChange,
    committed_errors,
)
from syntheticcx.domain.models import ErrorDetail, EvaluationResult, PersonaDefinition, Scenario
from syntheticcx.domain.outcome import aggregate_outcome
from syntheticcx.domain.protocols import (
    ChannelAdapter,
    Clock,
    PersonaSimulator,
    StorageBackend,
    WaitController,
)
from syntheticcx.errors import ConfigurationError, EvaluationError, PersistenceError
from syntheticcx.evaluation.service import EvaluationService
from syntheticcx.privacy import Redactor
from syntheticcx.runtime.journal import RunJournal
from syntheticcx.runtime.model_metadata import with_committed_interactions
from syntheticcx.runtime.result import EvaluatedRunResult
from syntheticcx.runtime.runner import AsyncCloseable, run_conversation
from syntheticcx.telemetry.run import span

T = TypeVar("T")


async def _finish(operation: Awaitable[T], *, committed_result_wins: bool = False) -> T:
    """Finish one bounded commit or close despite repeated task cancellation."""
    task = asyncio.ensure_future(operation)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    result = await task
    if cancelled and not committed_result_wins:
        raise asyncio.CancelledError
    return result


async def run_evaluated(
    *,
    run: Run,
    config_artifact: ConfigArtifact,
    scenario: Scenario,
    persona: PersonaDefinition,
    channel: ChannelAdapter,
    simulator: PersonaSimulator,
    storage: StorageBackend,
    journal: RunJournal,
    clock: Clock,
    wait: WaitController,
    limits: ExecutionLimits,
    evaluation: EvaluationService,
    redactor: Redactor,
    evaluation_provider: AsyncCloseable | None = None,
    project_config: ProjectConfig | None = None,
    on_created: Callable[[str], None] | None = None,
) -> EvaluatedRunResult:
    """Own conversation, evaluation, terminal trace, metadata commit and cleanup."""
    if project_config is not None and scenario.model_dump(
        mode="json"
    ) != project_config.scenario.model_dump(mode="json"):
        raise ConfigurationError("prepare run", "scenario differs from the effective configuration")
    conversation = await run_conversation(
        run=run,
        config_artifact=config_artifact,
        scenario=scenario,
        persona=persona,
        channel=channel,
        simulator=simulator,
        storage=storage,
        journal=journal,
        clock=clock,
        wait=wait,
        limits=limits,
        project_config=project_config,
        retain_storage=True,
        on_created=on_created,
    )
    results: tuple[EvaluationResult, ...] = ()
    infra = not conversation.cleanup_completed
    final = False
    cleanup_completed = conversation.cleanup_completed
    outcome = None
    exit_code = 3
    reason = conversation.termination_reason
    try:
        if reason == "persistence_error":
            detail = conversation.error or ErrorDetail(
                category="persistence",
                operation="run.evidence",
                message="persistence operation failed",
            )
            try:
                await _finish(journal.append(OperationFailure(error=detail)))
            except (Exception, asyncio.CancelledError):
                pass
        if reason != "persistence_error" and reason != "cancelled":
            try:
                with span("evaluation.run"):
                    results = await evaluation.evaluate(scenario)
            except asyncio.CancelledError:
                reason = "cancelled"
            except PersistenceError:
                reason = "persistence_error"
            except Exception:
                infra = True
                detail = ErrorDetail(
                    category="evaluation", operation="evaluate run", message="evaluation failed"
                )
                await _finish(journal.append(OperationFailure(error=detail)))

        if evaluation_provider is not None:
            try:
                with span("resource.cleanup", operation="evaluation.close"):
                    await _finish(
                        wait.run(evaluation_provider.close(), limits.close_timeout_seconds)
                    )
            except asyncio.CancelledError:
                reason = "cancelled"
            except Exception:
                infra = True
                if reason != "persistence_error":
                    detail = ErrorDetail(
                        category="model",
                        operation="evaluation.close",
                        message="model operation failed",
                    )
                    await _finish(journal.append(OperationFailure(error=detail)))

        if reason != "persistence_error":
            committed_ids = {
                event.data.result.criterion_id
                for event in journal.committed_events
                if isinstance(event, EvaluationCompletedEvent)
            }
            for criterion in scenario.success_criteria:
                if criterion.id not in committed_ids:
                    unavailable = EvaluationResult(
                        evaluator_id=criterion.evaluator_type,
                        criterion_id=criterion.id,
                        required=criterion.required,
                        status="unavailable",
                        explanation="Evaluation did not produce a finding for this criterion.",
                    )
                    await _finish(
                        journal.append(
                            EvaluationCompleted(
                                result=unavailable,
                                criterion=redactor.redact_criterion(criterion, scenario),
                            )
                        )
                    )
            results = tuple(
                event.data.result
                for event in journal.committed_events
                if isinstance(event, EvaluationCompletedEvent)
            )
            decision = aggregate_outcome(
                scenario.success_criteria,
                list(results),
                termination_reason=reason,
                infrastructure_error=infra,
            )
            await _finish(
                journal.append(
                    RunLifecycleChange(
                        previous="in_progress",
                        current="finalized",
                        termination_reason=reason,
                        outcome=decision.outcome,
                    )
                )
            )
            finalized_run = Run.model_validate(
                run.model_copy(
                    update={
                        "ended_at": clock.now(),
                        "lifecycle_status": "finalized",
                        "outcome": decision.outcome,
                        "termination_reason": reason,
                        "errors": committed_errors(journal.committed_events),
                        "metadata": with_committed_interactions(
                            run.metadata, journal.committed_events
                        ),
                    }
                ).model_dump()
            )
            with span("persistence.finalize"):
                await _finish(
                    wait.run(
                        storage.finalize_run(finalized_run), limits.persistence_timeout_seconds
                    ),
                    committed_result_wins=True,
                )
            final = True
            outcome = decision.outcome
            exit_code = decision.exit_code
    except asyncio.CancelledError:
        # A terminal trace event can precede the final metadata commit marker.
        # Never claim finalization when its acknowledgement was interrupted.
        outcome = None
        exit_code = 130
    except EvaluationError:
        outcome = None
        exit_code = 3
        detail = ErrorDetail(
            category="evaluation", operation="aggregate outcome", message="evaluation failed"
        )
        try:
            await _finish(journal.append(OperationFailure(error=detail)))
        except (Exception, asyncio.CancelledError):
            pass
    except Exception:
        # A failed writer/uncertain finalization cannot claim a finalized result.
        outcome = None
        exit_code = 3
        detail = ErrorDetail(
            category="persistence", operation="run.finalize", message="persistence operation failed"
        )
        try:
            await _finish(journal.append(OperationFailure(error=detail)))
        except (Exception, asyncio.CancelledError):
            pass
    finally:
        for resource in (storage,):
            if isinstance(resource, AsyncCloseable):
                try:
                    with span("resource.cleanup", operation="storage.close"):
                        await _finish(wait.run(resource.close(), limits.close_timeout_seconds))
                except (Exception, asyncio.CancelledError):
                    cleanup_completed = False

    return EvaluatedRunResult(
        conversation=conversation,
        evaluations=results,
        outcome=outcome,
        exit_code=exit_code,
        finalized=final,
        cleanup_completed=cleanup_completed,
        committed_events=journal.committed_events,
        errors=tuple(committed_errors(journal.committed_events)),
    )

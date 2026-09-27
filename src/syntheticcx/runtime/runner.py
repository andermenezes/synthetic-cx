"""Own run creation, graph deadlines, terminal evidence, and bounded resource cleanup."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from datetime import timedelta
from typing import Protocol, TypeVar, cast, runtime_checkable

from syntheticcx.config.models import ExecutionLimits, ProjectConfig
from syntheticcx.domain.artifacts import ConfigArtifact, Run
from syntheticcx.domain.events import OperationFailure, RunLifecycleChange
from syntheticcx.domain.models import ArtifactRecord, ErrorDetail, PersonaDefinition, Scenario
from syntheticcx.domain.protocols import (
    ChannelAdapter,
    Clock,
    PersonaSimulator,
    StorageBackend,
    WaitController,
)
from syntheticcx.errors import ErrorCategory, SyntheticCXError
from syntheticcx.runtime.capture import (
    FailureScreenshotSource,
    ScreenshotArtifactWriter,
    ScreenshotPolicy,
)
from syntheticcx.runtime.graph import build_graph
from syntheticcx.runtime.journal import RunJournal
from syntheticcx.runtime.result import RuntimeResult, TerminationReason
from syntheticcx.runtime.state import RuntimeProgress, initial_state

T = TypeVar("T")


@runtime_checkable
class AsyncCloseable(Protocol):
    async def close(self) -> None: ...


def _error(category: ErrorCategory, operation: str) -> ErrorDetail:
    """Do not copy untrusted exception messages into results or evidence."""
    return ErrorDetail(
        category=category, operation=operation, message=f"{category} operation failed"
    )


def _failure(exc: Exception, stage: str) -> tuple[TerminationReason, ErrorDetail]:
    if isinstance(exc, SyntheticCXError):
        category = exc.category
        operation = (
            exc.operation
            if exc.operation
            in {
                "persona.initialize",
                "persona.respond",
                "persona.interactions",
                "channel.greeting",
                "channel.send",
                "channel.receive",
                "append journal",
            }
            else stage
        )
    elif stage in {"storage.create_run", "journal.append"}:
        category, operation = "persistence", stage
    elif stage == "channel.start":
        category, operation = "channel", stage
    else:
        category, operation = "model", stage
    reason: TerminationReason = (
        "persistence_error"
        if category == "persistence"
        else "channel_error"
        if category == "channel"
        else "model_error"
    )
    return reason, _error(category, operation)


async def run_conversation(
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
    project_config: ProjectConfig | None = None,
    retain_storage: bool = False,
) -> RuntimeResult:
    """Execute once and return committed evidence without finalizing the run.

    The caller provides an uncreated run and a journal bound to the same run ID,
    storage, and clock. In owner mode the conversation stop is recorded while
    the lifecycle stays in progress; the owner closes the writer after evaluation.
    Only a successfully started resource is closed; a failed start is also closed
    because it may have acquired partial resources.
    """
    progress = RuntimeProgress(persona.fresh_state())
    reason: TerminationReason | None = None
    error: ErrorDetail | None = None
    cleanup_completed = True
    channel_attempted = False
    stage = "storage.create_run"
    deadline = clock.now() + timedelta(seconds=limits.journey_timeout_seconds)
    secondary_errors: list[ErrorDetail] = []
    screenshot_policy = ScreenshotPolicy.from_config(project_config) if project_config else None

    journal.configure_persistence(wait, limits.persistence_timeout_seconds)

    async def finish_cleanup(operation: Awaitable[T], *, terminal_record: bool = False) -> T:
        """Finish one bounded operation despite cancellation of the runner task."""
        nonlocal reason, error
        task = asyncio.ensure_future(operation)
        while True:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                # A terminal lifecycle event already being written (or a storage
                # close after it) cannot be retracted; keep its termination reason.
                if not terminal_record:
                    reason = "cancelled"
                    error = _error("cancelled", "runtime.cancelled")
                elif error is None:
                    error = _error("cancelled", "runtime.cancelled")
                if task.done():
                    raise

    def remaining_time() -> float:
        return (deadline - clock.now()).total_seconds()

    async def record_failure(detail: ErrorDetail) -> None:
        nonlocal reason, error
        try:
            await journal.append(OperationFailure(error=detail))
        except Exception:
            # A poisoned journal cannot safely record another failure.
            if reason is None or reason not in {"persistence_error", "cancelled"}:
                reason = "persistence_error"
                error = _error("persistence", "journal.append")

    async def record_failure_screenshot() -> None:
        """Record a truthful omission/failure or acknowledged artifact before close."""
        nonlocal reason, error
        policy = screenshot_policy or ScreenshotPolicy()
        record = ArtifactRecord(
            media_type="image/png", status="omitted", reason="screenshots_disabled"
        )
        if policy.enabled:
            record = ArtifactRecord(
                media_type="image/png", status="failed", reason="capture_unavailable"
            )
            if isinstance(channel, FailureScreenshotSource) and isinstance(
                storage, ScreenshotArtifactWriter
            ):
                try:
                    image = await finish_cleanup(
                        wait.run(
                            channel.capture_failure_screenshot(),
                            limits.browser_action_timeout_seconds,
                        )
                    )
                except (Exception, asyncio.CancelledError):
                    record = ArtifactRecord(
                        media_type="image/png", status="failed", reason="capture_failed"
                    )
                else:
                    try:
                        record = await finish_cleanup(
                            wait.run(
                                storage.write_screenshot(
                                    run.id, "artifacts/screenshots/channel-failure.png", image
                                ),
                                limits.persistence_timeout_seconds,
                            )
                        )
                    except (Exception, asyncio.CancelledError):
                        record = ArtifactRecord(
                            media_type="image/png", status="failed", reason="storage_failed"
                        )
                    finally:
                        image = b""
        try:
            await finish_cleanup(journal.append(record))
        except (Exception, asyncio.CancelledError):
            if reason != "persistence_error":
                reason = "persistence_error"
                error = _error("persistence", "journal.append")

    try:
        if remaining_time() <= 0:
            raise TimeoutError("journey deadline expired")
        await wait.run(
            storage.create_run(run, config_artifact),
            min(limits.persistence_timeout_seconds, remaining_time()),
        )
        stage = "channel.start"
        if remaining_time() <= 0:
            raise TimeoutError("journey deadline expired")
        channel_attempted = True
        await wait.run(
            channel.start(), min(limits.browser_action_timeout_seconds, remaining_time())
        )
        stage = "graph"
        remaining = remaining_time()
        if remaining <= 0:
            raise TimeoutError("journey deadline expired")
        graph = build_graph(
            scenario=scenario,
            persona=persona,
            channel=channel,
            simulator=simulator,
            journal=journal,
            wait=wait,
            limits=limits,
            progress=progress,
        )
        completed = await wait.run(
            graph.ainvoke(
                initial_state(persona),
                config={"recursion_limit": 4 * limits.max_turns + 10},
            ),
            remaining,
        )
        reason = cast(TerminationReason, completed["termination_reason"])
        progress.persona_state = completed["persona_state"]
        progress.turn_count = completed["turn_count"]
        if reason is None:
            raise ValueError("graph ended without a termination reason")
    except asyncio.CancelledError:
        reason = "cancelled"
        error = _error("cancelled", "runtime.cancelled")
    except TimeoutError:
        reason = (
            "journey_timeout"
            if stage == "graph" or remaining_time() <= 0
            else ("persistence_error" if stage == "storage.create_run" else "channel_error")
        )
        if reason != "journey_timeout":
            error = _error("persistence" if stage == "storage.create_run" else "channel", stage)
    except Exception as exc:
        reason, error = _failure(exc, stage)
    finally:
        try:
            if channel_attempted and reason in {"channel_error", "journey_timeout"}:
                await record_failure_screenshot()
            if channel_attempted:
                try:
                    await finish_cleanup(wait.run(channel.close(), limits.close_timeout_seconds))
                except (Exception, asyncio.CancelledError):
                    cleanup_completed = False
                    secondary = _error("channel", "channel.close")
                    if error is None:
                        error = secondary
                    secondary_errors.append(secondary)
            if isinstance(simulator, AsyncCloseable):
                try:
                    await finish_cleanup(wait.run(simulator.close(), limits.close_timeout_seconds))
                except (Exception, asyncio.CancelledError):
                    cleanup_completed = False
                    secondary = _error("model", "persona.close")
                    if error is None:
                        error = secondary
                    secondary_errors.append(secondary)
            if error is not None and reason is not None and reason != "persistence_error":
                await finish_cleanup(record_failure(error))
            for secondary in secondary_errors:
                if secondary is not error and reason != "persistence_error":
                    await finish_cleanup(record_failure(secondary))
            if reason is not None and (not retain_storage or reason != "persistence_error"):
                try:
                    await finish_cleanup(
                        journal.append(
                            RunLifecycleChange(
                                previous="in_progress" if journal.committed_events else None,
                                current="in_progress" if retain_storage else "incomplete",
                                termination_reason=reason,
                            )
                        ),
                        terminal_record=True,
                    )
                except (Exception, asyncio.CancelledError):
                    if reason != "persistence_error":
                        reason = "persistence_error"
                        error = _error("persistence", "journal.append")
        finally:
            if not retain_storage and isinstance(storage, AsyncCloseable):
                try:
                    await finish_cleanup(
                        wait.run(storage.close(), limits.close_timeout_seconds),
                        terminal_record=True,
                    )
                except (Exception, asyncio.CancelledError):
                    cleanup_completed = False
                    secondary = _error("persistence", "storage.close")
                    if error is None:
                        error = secondary

    assert reason is not None
    return RuntimeResult(
        run_id=run.id,
        termination_reason=reason,
        persona_state=progress.persona_state.model_copy(deep=True),
        committed_events=journal.committed_events,
        turn_count=progress.turn_count,
        error=error,
        cleanup_completed=cleanup_completed,
    )

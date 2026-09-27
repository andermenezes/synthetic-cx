"""Single-run local storage with serialized, acknowledged durability boundaries.

A failed, cancelled, or timed-out operation makes this instance unusable. A
daemon worker can finish an in-flight syscall later, but cannot acknowledge it
to the caller or accept subsequent work. Recovery belongs to a later checkpoint.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import tempfile
from collections.abc import Callable
from concurrent.futures import Future
from hashlib import sha256
from pathlib import Path, PurePosixPath
from queue import Queue
from threading import Thread

import yaml
from pydantic import BaseModel

from syntheticcx.config.models import Criterion, ProjectConfig
from syntheticcx.domain.artifacts import ConfigArtifact, MetadataArtifact, Run
from syntheticcx.domain.events import RunLifecycleEvent, TypedTraceEvent, committed_errors
from syntheticcx.domain.models import ArtifactRecord
from syntheticcx.errors import PersistenceError
from syntheticcx.privacy import Redactor
from syntheticcx.reporting.materialize import (
    build_projections,
    committed_events_from_lines,
    reconcile_artifacts,
)
from syntheticcx.runtime.capture import ScreenshotPolicy


def canonical_bytes(model: BaseModel) -> bytes:
    """Canonical v1 JSON record: sorted keys, compact UTF-8, one newline."""
    return (
        json.dumps(
            model.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


class _Worker:
    """One daemon thread, with no default-executor shutdown dependency."""

    def __init__(self) -> None:
        self.queue: Queue[tuple[Callable[[], None], Future[None]] | None] = Queue()
        self.thread = Thread(target=self._run, name="syntheticcx-storage", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while (item := self.queue.get()) is not None:
            operation, result = item
            try:
                operation()
            except Exception:
                # Translate here so raw OS exceptions never cross the worker boundary.
                result.set_exception(
                    PersistenceError("write evidence", "filesystem operation failed")
                )
            else:
                result.set_result(None)

    def submit(self, operation: Callable[[], None]) -> Future[None]:
        result: Future[None] = Future()
        self.queue.put((operation, result))
        return result

    def close(self) -> None:
        self.queue.put(None)


class FilesystemStorage:
    """Own one run and one writer; instantiate separately for each new run.

    Call close() from the eventual runner's finally block. No source YAML or
    raw events are retained. Existing run directories are never reopened here.
    """

    def __init__(
        self,
        root: Path,
        redactor: Redactor,
        *,
        timeout_seconds: float,
        criteria: list[dict[str, object]] | None = None,
        materialize_conversation: bool = True,
        materialize_evaluation: bool = True,
        materialize_report: bool = True,
        project_config: ProjectConfig | None = None,
    ) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("persistence timeout must be finite and positive")
        self._root = root
        self._redactor = redactor
        self._timeout = timeout_seconds
        supplied_criteria = [
            Criterion.model_validate(
                {"result_kind": "boolean", "evidence_policy": "required", **criterion}
            )
            for criterion in criteria or []
        ]
        if project_config is not None:
            self._criteria = [
                redactor.redact_criterion(item, project_config.scenario)
                for item in project_config.scenario.success_criteria
            ]
        elif redactor.has_field_redaction and any(
            item.evaluator_type == "semantic_goal_completion" for item in supplied_criteria
        ):
            raise ValueError(
                "semantic criteria require project_config when field redaction is enabled"
            )
        else:
            self._criteria = supplied_criteria
        self._materialize_conversation = materialize_conversation
        self._materialize_evaluation = materialize_evaluation
        self._materialize_report = materialize_report
        self._screenshot_policy = (
            ScreenshotPolicy.from_config(project_config) if project_config else ScreenshotPolicy()
        )
        self._prompt_capture_disabled = (
            project_config is not None and not project_config.privacy.capture_prompts
        )
        self._lock = asyncio.Lock()
        self._worker: _Worker | None = None
        self._failed = False
        self._closed = False
        self._run_id: str | None = None
        self._records: dict[str, bytes] = {}
        self._finalized_metadata: bytes | None = None
        self._finalizing = False

    async def _perform(
        self, operation: str, action: Callable[[], None], *, allow_finalizing: bool = False
    ) -> None:
        try:
            async with asyncio.timeout(self._timeout):
                async with self._lock:
                    if self._failed or self._closed or (self._finalizing and not allow_finalizing):
                        raise PersistenceError(operation, "writer is unavailable")
                    if self._worker is None:
                        self._worker = _Worker()
                    future = asyncio.wrap_future(self._worker.submit(action))
                    # Consume late failures without cancelling the worker's Future.
                    future.add_done_callback(
                        lambda done: done.exception() if not done.cancelled() else None
                    )
                    await asyncio.shield(future)
                    if self._failed or self._closed:
                        raise PersistenceError(operation, "writer is unavailable")
        except asyncio.CancelledError:
            self._failed = True
            await self.close()
            raise
        except PersistenceError:
            if self._finalizing:
                raise PersistenceError(operation, "run is finalized") from None
            self._failed = True
            await self.close()
            raise PersistenceError(operation, "operation failed; writer is unavailable") from None
        except Exception:
            self._failed = True
            await self.close()
            raise PersistenceError(operation, "operation failed; writer is unavailable") from None

    async def create_run(self, run: Run, config_artifact: ConfigArtifact) -> None:
        """Publish initial in-progress evidence before any runtime activity."""
        try:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", run.id):
                raise ValueError("invalid run identifier")
            if (
                run.lifecycle_status != "in_progress"
                or run.ended_at is not None
                or run.events
                or run.messages
                or run.evaluations
                or run.outcome is not None
            ):
                raise ValueError("creation requires an empty in-progress run")
            metadata = MetadataArtifact.model_validate(
                run.model_dump(exclude={"events", "messages", "evaluations"})
            )
            if self._prompt_capture_disabled:
                metadata = metadata.model_copy(
                    update={
                        "metadata": metadata.metadata.model_copy(
                            update={
                                "capture_omissions": list(
                                    dict.fromkeys(
                                        metadata.metadata.capture_omissions
                                        + ["prompts_not_captured"]
                                    )
                                )
                            }
                        )
                    }
                )
            safe_metadata = self._redactor.redact_model(metadata)
            if safe_metadata.id != run.id:
                raise ValueError("redaction changed run identity")
            metadata_bytes = canonical_bytes(safe_metadata)
            config_bytes = yaml.safe_dump(
                self._redactor.redact_model(config_artifact).model_dump(mode="json"),
                sort_keys=True,
                allow_unicode=True,
            ).encode("utf-8")
        except Exception:
            self._failed = True
            await self.close()
            raise PersistenceError("create run", "invalid initial run evidence") from None
        run_id = run.id

        def create() -> None:
            if self._run_id is not None:
                raise ValueError("writer already owns a run")
            self._ensure_directory(self._root)
            directory = self._root / run_id
            directory.mkdir()  # Exclusive: never touch an existing run.
            self._sync_directory(self._root)
            self._atomic_write(directory / "metadata.json", metadata_bytes)
            self._atomic_write(directory / "config.yaml", config_bytes)
            with (directory / "trace.jsonl").open("xb") as stream:
                stream.flush()
                os.fsync(stream.fileno())
            self._sync_directory(directory)
            self._run_id = run_id

        await self._perform("create run", create)

    async def append_event(self, run_id: str, event: TypedTraceEvent) -> None:
        """Append an ordered event, or acknowledge an identical durable retry."""
        async with self._lock:
            if self._finalizing:
                raise PersistenceError("append event", "run is finalized")
        try:
            safe_event = self._redactor.redact_model(event)
            if (safe_event.id, safe_event.sequence, safe_event.timestamp) != (
                event.id,
                event.sequence,
                event.timestamp,
            ):
                raise ValueError("redaction changed event identity")
            record = canonical_bytes(safe_event)
        except Exception:
            self._failed = True
            await self.close()
            raise PersistenceError("append event", "invalid event evidence") from None
        event_id, sequence = safe_event.id, safe_event.sequence

        def append() -> None:
            if self._run_id is None or run_id != self._run_id:
                raise ValueError("unknown run")
            previous = self._records.get(event_id)
            if previous is not None:
                if previous != record:
                    raise ValueError("conflicting retry")
                return
            if sequence != len(self._records):
                raise ValueError("noncontiguous sequence")
            with (self._root / run_id / "trace.jsonl").open("ab") as stream:
                if stream.write(record) != len(record):
                    raise OSError("partial append")
                stream.flush()
                os.fsync(stream.fileno())
            self._records[event_id] = record

        await self._perform("append event", append)

    async def write_screenshot(
        self, run_id: str, relative_path: str, content: bytes
    ) -> ArtifactRecord:
        """Acknowledge an owned, size-bounded PNG before exposing its artifact record."""
        if not self._screenshot_policy.enabled:
            raise PersistenceError("write screenshot", "screenshot storage is disabled")
        path = PurePosixPath(relative_path)
        if (
            str(path) != relative_path
            or path.is_absolute()
            or ".." in path.parts
            or "." in path.parts
            or path.parts[:2] != ("artifacts", "screenshots")
            or len(path.parts) != 3
            or not re.fullmatch(r"[a-z0-9_-]{1,64}\.png", path.name)
            or len(content) > 5 * 1024 * 1024
            or not content.startswith(b"\x89PNG\r\n\x1a\n")
        ):
            raise PersistenceError("write screenshot", "invalid screenshot artifact")
        if self._run_id is None or self._run_id != run_id:
            raise PersistenceError("write screenshot", "run is not owned by this writer")

        def write() -> None:
            run_directory = self._root / run_id
            if run_directory.is_symlink() or not run_directory.is_dir():
                raise ValueError("invalid run directory")
            artifacts = run_directory / "artifacts"
            screenshots = artifacts / "screenshots"
            if artifacts.is_symlink() or screenshots.is_symlink():
                raise ValueError("screenshot directory is not owned")
            self._ensure_directory(screenshots)
            if not screenshots.resolve().is_relative_to(run_directory.resolve()):
                raise ValueError("screenshot path escapes run")
            destination = screenshots / path.name
            if destination.exists() or destination.is_symlink():
                raise ValueError("screenshot already exists")
            self._atomic_write(destination, content)

        await self._perform("write screenshot", write)
        return ArtifactRecord(
            path=relative_path,
            media_type="image/png",
            content_hash=sha256(content).hexdigest(),
            status="captured",
        )

    async def finalize_run(self, run: Run) -> None:
        """Atomically publish enabled projections, then final metadata last.

        The provided run supplies identity, lifecycle transition, and storage
        flags only; artifact views come from the committed trace. Retry with
        identical committed evidence reproduces identical artifacts.
        """
        async with self._lock:
            if self._failed or self._closed:
                raise PersistenceError("finalize run", "writer is unavailable")
            if self._finalizing and self._finalized_metadata is None:
                raise PersistenceError("finalize run", "writer is unavailable")
            # Wait for an append already holding the writer lock, then make the
            # trace immutable before its projection snapshot is read.
            self._finalizing = True
        try:
            if self._run_id is None or run.id != self._run_id:
                raise ValueError("unknown run")
            if run.lifecycle_status != "finalized":
                raise ValueError("finalization requires a finalized run")
            lines = (self._root / run.id / "trace.jsonl").read_text(encoding="utf-8").splitlines()
            events = committed_events_from_lines(lines)
            final_events = [
                event
                for event in events
                if isinstance(event, RunLifecycleEvent) and event.data.current == "finalized"
            ]
            if (
                len(final_events) != 1
                or events[-1] is not final_events[0]
                or final_events[0].data.outcome != run.outcome
                or final_events[0].data.termination_reason != run.termination_reason
                or not final_events[0].data.termination_reason
                or final_events[0].data.outcome is None
            ):
                raise ValueError("final lifecycle does not agree with metadata")
            criteria = [criterion.model_copy(deep=True) for criterion in self._criteria]
            records: list[ArtifactRecord] = [
                ArtifactRecord(
                    path="conversation.json", media_type="application/json", status="captured"
                )
                if self._materialize_conversation
                else ArtifactRecord(
                    media_type="application/json", status="omitted", reason="disabled"
                ),
                ArtifactRecord(
                    path="evaluation.json", media_type="application/json", status="captured"
                )
                if self._materialize_evaluation
                else ArtifactRecord(
                    media_type="application/json", status="omitted", reason="disabled"
                ),
                ArtifactRecord(path="report.json", media_type="application/json", status="captured")
                if self._materialize_report
                else ArtifactRecord(
                    media_type="application/json", status="omitted", reason="disabled"
                ),
            ]
            records = reconcile_artifacts(events, records + run.artifacts)
            safe_records = [self._redactor.redact_model(record) for record in records]
            projections = build_projections(
                run.id,
                events,
                criteria,
                artifact_location=str(self._root / run.id),
                artifacts=safe_records,
                materialize_conversation=self._materialize_conversation,
                materialize_evaluation=self._materialize_evaluation,
                materialize_report=self._materialize_report,
            )
            metadata = MetadataArtifact.model_validate(
                run.model_dump(exclude={"events", "messages", "evaluations"})
            )
            if metadata.lifecycle_status != "finalized":
                raise ValueError("finalization requires a finalized metadata artifact")
            if metadata.errors != committed_errors(events):
                raise ValueError("metadata errors do not agree with the trace")
            omissions = [
                f"screenshot:{record.reason or record.status}"
                for record in safe_records
                if record.media_type == "image/png" and record.status != "captured"
            ]
            if self._prompt_capture_disabled:
                omissions.append("prompts_not_captured")
            safe_metadata = self._redactor.redact_model(
                metadata.model_copy(
                    update={
                        "artifacts": safe_records,
                        "metadata": metadata.metadata.model_copy(
                            update={
                                "capture_omissions": list(
                                    dict.fromkeys(metadata.metadata.capture_omissions + omissions)
                                )
                            }
                        ),
                    }
                )
            )
            if safe_metadata.id != run.id:
                raise ValueError("redaction changed run identity")
            metadata_bytes = canonical_bytes(safe_metadata)
            conversation_bytes = (
                canonical_bytes(projections.conversation)
                if projections.conversation is not None
                else None
            )
            evaluation_bytes = (
                canonical_bytes(projections.evaluation)
                if projections.evaluation is not None
                else None
            )
            report_bytes = (
                canonical_bytes(projections.report) if projections.report is not None else None
            )
            if self._finalized_metadata is not None and self._finalized_metadata != metadata_bytes:
                raise ValueError("conflicting finalization retry")
        except PersistenceError:
            self._failed = True
            await self.close()
            raise
        except Exception:
            self._failed = True
            await self.close()
            raise PersistenceError("finalize run", "invalid finalization evidence") from None
        run_id = run.id

        def finalize() -> None:
            directory = self._root / run_id
            # Projections first; final metadata is the commit marker and is
            # replaced last so a crash never publishes finalized state without
            # its views.
            if conversation_bytes is not None:
                self._atomic_write(directory / "conversation.json", conversation_bytes)
            if evaluation_bytes is not None:
                self._atomic_write(directory / "evaluation.json", evaluation_bytes)
            if report_bytes is not None:
                self._atomic_write(directory / "report.json", report_bytes)
            self._atomic_write(directory / "metadata.json", metadata_bytes)
            self._finalized_metadata = metadata_bytes

        await self._perform("finalize run", finalize, allow_finalizing=True)

    async def close(self) -> None:
        """Stop accepting work; an unresolved syscall may finish on the daemon."""
        if not self._closed:
            self._closed = True
            if self._worker is not None:
                self._worker.close()

    @staticmethod
    def _sync_directory(directory: Path) -> None:
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @classmethod
    def _ensure_directory(cls, directory: Path) -> None:
        if directory.is_dir():
            return
        cls._ensure_directory(directory.parent)
        directory.mkdir(exist_ok=True)
        cls._sync_directory(directory.parent)

    @classmethod
    def _atomic_write(cls, destination: Path, content: bytes) -> None:
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
                temporary = stream.name
                if stream.write(content) != len(content):
                    raise OSError("partial artifact write")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            temporary = None
            cls._sync_directory(destination.parent)
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

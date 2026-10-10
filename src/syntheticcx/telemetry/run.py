"""Owned OpenTelemetry providers and privacy-filtered supplemental local traces."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar, Token
from pathlib import Path
from threading import Lock
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from syntheticcx.config.models import TelemetryConfig
from syntheticcx.privacy import Redactor

_ACTIVE: ContextVar[RunTelemetry | None] = ContextVar("syntheticcx_telemetry", default=None)
_SAFE_NAMES = frozenset(
    {
        "run.initialize",
        "browser.start",
        "conversation.run",
        "conversation.turn",
        "persona.update",
        "persona.initialize",
        "persona.respond",
        "model.call",
        "evaluation.run",
        "evaluation.criterion",
        "persistence.create",
        "persistence.append",
        "persistence.finalize",
        "resource.cleanup",
    }
)
_SAFE_ATTRIBUTES = frozenset({"run.id", "operation", "turn.number", "error.category"})
_OPERATIONS = frozenset(
    {
        "persona.initialize",
        "persona.respond",
        "evaluation.generate",
        "evaluation.close",
        "channel.close",
        "persona.close",
        "storage.close",
    }
)
_CATEGORIES = frozenset(
    {"configuration", "channel", "model", "behavior", "evaluation", "persistence", "cancelled"}
)


class _LocalExporter(SpanExporter):  # type: ignore[misc]
    """Keep only approved fields in memory until the run directory exists."""

    def __init__(self, redactor: Redactor) -> None:
        self.records: list[dict[str, Any]] = []
        self._redactor = redactor
        self._lock = Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        records: list[dict[str, Any]] = []
        for item in spans:
            context = item.get_span_context()
            if context is None or item.name not in _SAFE_NAMES:
                continue
            parent = item.parent
            attributes = {
                key: value
                for key, value in (item.attributes or {}).items()
                if key in _SAFE_ATTRIBUTES and isinstance(value, (str, int, float, bool))
            }
            records.append(
                {
                    "name": item.name,
                    "trace_id": f"{context.trace_id:032x}",
                    "span_id": f"{context.span_id:016x}",
                    "parent_span_id": f"{parent.span_id:016x}" if parent is not None else None,
                    "start_ns": item.start_time,
                    "end_ns": item.end_time,
                    "attributes": self._redactor.redact_value(attributes),
                }
            )
        with self._lock:
            self.records.extend(records)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


class RunTelemetry:
    """A single run owns its provider, exporter and correlation-aware logger."""

    def __init__(
        self,
        config: TelemetryConfig,
        run_id: str,
        directory: Path,
        redactor: Redactor,
        *,
        remote_timeout: float,
    ) -> None:
        self._enabled = config.enabled
        self._directory = directory
        self._run_id = run_id
        self._redactor = redactor
        self._provider: TracerProvider | None = None
        self._local: _LocalExporter | None = None
        self._logger: logging.Logger | None = None
        self._handler: logging.Handler | None = None
        if not config.enabled:
            return
        provider = TracerProvider(
            resource=Resource({"service.name": "syntheticcx"}), shutdown_on_exit=False
        )
        local = _LocalExporter(redactor)
        provider.add_span_processor(SimpleSpanProcessor(local))
        if config.remote_export:
            # The optional package is validated before model and browser activity.
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[import-not-found]
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(
                endpoint=str(config.otlp_endpoint),
                headers={"x-syntheticcx-client": "syntheticcx"},
                timeout=remote_timeout,
            )
            provider.add_span_processor(BatchSpanProcessor(exporter))
        self._provider = provider
        self._local = local
        self._tracer = provider.get_tracer("syntheticcx")
        logger = logging.getLogger(f"syntheticcx.run.{run_id}")
        logger.propagate = False
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        self._logger, self._handler = logger, handler

    @contextmanager
    def activate(self) -> Iterator[None]:
        token: Token[RunTelemetry | None] = _ACTIVE.set(self)
        try:
            yield
        finally:
            _ACTIVE.reset(token)

    @contextmanager
    def span(self, name: str, **attributes: str | int | float | bool) -> Iterator[None]:
        if not self._enabled or name not in _SAFE_NAMES:
            yield
            return
        safe = {key: value for key, value in attributes.items() if key in _SAFE_ATTRIBUTES}
        if safe.get("operation") not in _OPERATIONS:
            safe.pop("operation", None)
        if safe.get("error.category") not in _CATEGORIES:
            safe.pop("error.category", None)
        safe["run.id"] = self._run_id
        safe = self._redactor.redact_value(safe)
        with self._tracer.start_as_current_span(
            name,
            attributes=safe,
            record_exception=False,
            set_status_on_exception=False,
        ):
            yield

    def log(self, event: str) -> None:
        if self._logger is None:
            return
        current = trace.get_current_span().get_span_context()
        self._logger.info(
            "event=%s run_id=%s trace_id=%032x span_id=%016x",
            event if event in _SAFE_NAMES else "operation",
            self._run_id,
            current.trace_id,
            current.span_id,
        )

    async def close(self, *, created: bool, timeout: float) -> None:
        if self._provider is None or self._local is None:
            return
        provider = self._provider
        local = self._local
        if self._logger is not None and self._handler is not None:
            self._logger.removeHandler(self._handler)
            self._handler.close()

        def finish() -> None:
            try:
                if created:
                    target = self._directory / "artifacts" / "telemetry" / "spans.jsonl"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
                    try:
                        with temporary.open("w", encoding="utf-8") as stream:
                            for record in local.records:
                                stream.write(
                                    json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
                                )
                            stream.flush()
                            os.fsync(stream.fileno())
                        os.replace(temporary, target)
                    finally:
                        temporary.unlink(missing_ok=True)
            finally:
                # The optional remote processor can block; local spans are already safe.
                provider.shutdown()

        await asyncio.wait_for(asyncio.to_thread(finish), timeout)


@contextmanager
def span(name: str, **attributes: str | int | float | bool) -> Iterator[None]:
    """No-op without an active run; fixed names and attribute keys only."""
    active = _ACTIVE.get()
    with active.span(name, **attributes) if active is not None else nullcontext():
        if active is not None and name in _SAFE_NAMES:
            active.log(name)
        yield


def log(event: str) -> None:
    active = _ACTIVE.get()
    if active is not None:
        active.log(event)

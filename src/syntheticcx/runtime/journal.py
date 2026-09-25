"""Per-run event identity, sanitization, and durable committed cursor."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from typing import cast
from uuid import uuid4

from pydantic import TypeAdapter

from syntheticcx.domain.events import (
    ConversationStarted,
    EvaluationCompleted,
    MessageObserved,
    OperationFailure,
    RepeatedInformationObserved,
    RunLifecycleChange,
    TypedTraceEvent,
)
from syntheticcx.domain.models import ArtifactRecord, ModelInteraction, PersonaStateTransition
from syntheticcx.domain.protocols import Clock, StorageBackend, WaitController
from syntheticcx.errors import PersistenceError
from syntheticcx.privacy import Redactor

EventData = (
    RunLifecycleChange
    | ConversationStarted
    | MessageObserved
    | PersonaStateTransition
    | RepeatedInformationObserved
    | ModelInteraction
    | EvaluationCompleted
    | OperationFailure
    | ArtifactRecord
)
_EVENT_TYPES = {
    RunLifecycleChange: "run_lifecycle_changed",
    ConversationStarted: "conversation_started",
    MessageObserved: "message_observed",
    PersonaStateTransition: "persona_state_changed",
    RepeatedInformationObserved: "repeated_information",
    ModelInteraction: "model_interaction_completed",
    EvaluationCompleted: "evaluation_completed",
    OperationFailure: "operation_failed",
    ArtifactRecord: "artifact_recorded",
}


class RunJournal:
    """Allocate sequences from zero and expose detached committed evidence only."""

    def __init__(
        self,
        run_id: str,
        storage: StorageBackend,
        redactor: Redactor,
        clock: Clock,
        *,
        id_factory: Callable[[], str] = lambda: str(uuid4()),
    ) -> None:
        self._run_id = run_id
        self._storage = storage
        self._redactor = redactor
        self._clock = clock
        self._id_factory = id_factory
        self._events: list[TypedTraceEvent] = []
        self._lock = asyncio.Lock()
        self._failed = False
        self._wait: WaitController | None = None
        self._persistence_timeout: float | None = None

    def configure_persistence(self, wait: WaitController, timeout: float) -> None:
        """Bound each subsequent append before any run evidence is written."""
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("persistence timeout must be finite and positive")
        if self._events or self._wait is not None:
            raise ValueError("journal persistence boundary must be configured once, before appends")
        self._wait = wait
        self._persistence_timeout = timeout

    @property
    def committed_events(self) -> tuple[TypedTraceEvent, ...]:
        return tuple(event.model_copy(deep=True) for event in self._events)

    async def append(self, data: EventData) -> TypedTraceEvent:
        """Allocate and append once; never advance the cursor on uncertain failure."""
        try:
            async with self._lock:
                if self._failed:
                    raise PersistenceError("append journal", "journal is unavailable")
                event = TypeAdapter(TypedTraceEvent).validate_python(
                    {
                        "id": self._id_factory(),
                        "sequence": len(self._events),
                        "timestamp": self._clock.now(),
                        "type": _EVENT_TYPES[type(data)],
                        "data": data.model_dump(mode="json"),
                    }
                )
                safe_event = self._redactor.redact_model(event)
                if (safe_event.id, safe_event.sequence, safe_event.timestamp) != (
                    event.id,
                    event.sequence,
                    event.timestamp,
                ):
                    raise ValueError("redaction changed event identity")
                event = safe_event
                if any(previous.id == event.id for previous in self._events):
                    raise ValueError("duplicate generated event ID")
                operation = self._storage.append_event(self._run_id, event.model_copy(deep=True))
                if self._wait is not None and self._persistence_timeout is not None:
                    await self._wait.run(operation, self._persistence_timeout)
                else:
                    await operation
                if self._failed:
                    raise PersistenceError("append journal", "journal is unavailable")
                self._events.append(event)
                return cast(TypedTraceEvent, event.model_copy(deep=True))
        except asyncio.CancelledError:
            self._failed = True
            raise
        except Exception:
            self._failed = True
            raise PersistenceError("append journal", "event could not be committed") from None

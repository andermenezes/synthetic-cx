"""Asynchronous owned persona contract over a run-local TinyTroupe session."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Callable
from threading import Event, Thread
from typing import TypeVar

from syntheticcx._compat.blocking import run_blocking_bounded
from syntheticcx._compat.tinytroupe import TinySession
from syntheticcx.adapters.persona._client import TinyModelClient
from syntheticcx.config.models import (
    OpenAICompatibleModelConfig,
    OpenAIModelConfig,
    ResolvedModelSettings,
)
from syntheticcx.domain.models import (
    ConversationObservation,
    CustomerScenarioContext,
    ModelInteraction,
    PersonaAction,
    PersonaDefinition,
    PersonaState,
    PersonaStateTransition,
)
from syntheticcx.errors import ModelError

T = TypeVar("T")


def _customer_action(
    records: list[dict[str, object]], observation: ConversationObservation, state: PersonaState
) -> PersonaAction:
    """Accept one unambiguous external decision from the current action batch."""
    decisions: list[tuple[str, str | None]] = []
    for record in records:
        action = record.get("action") if isinstance(record, dict) else None
        if not isinstance(action, dict):
            raise ModelError("persona.respond", "persona action is malformed")
        kind, content = action.get("type"), action.get("content")
        if kind in {"DONE", "THINK"}:
            continue
        if kind not in {"TALK", "GOAL_COMPLETE", "ABANDON"}:
            raise ModelError("persona.respond", "persona action is unsupported")
        if kind == "TALK":
            if not isinstance(content, str) or not content.strip():
                raise ModelError("persona.respond", "persona customer message is missing")
            decisions.append((kind, content))
        elif content not in {"", None}:
            raise ModelError("persona.respond", "persona terminal action is malformed")
        else:
            decisions.append((kind, None))
    if len(decisions) != 1:
        raise ModelError("persona.respond", "persona decision is ambiguous")
    kind, content = decisions[0]
    if kind == "GOAL_COMPLETE":
        return PersonaAction(state=state, signal="goal_believed_complete")
    if kind == "ABANDON":
        after = state.model_copy(update={"abandoned": True, "willingness_to_continue": 0})
        return PersonaAction(
            state=after,
            signal="abandon",
            transitions=[
                PersonaStateTransition(
                    trigger_ids=[message.id for message in observation.messages],
                    before=state,
                    after=after,
                    applied_rule_ids=[],
                )
            ],
        )
    return PersonaAction(message=content, state=state, signal="continue")


class TinyTroupePersonaSimulator:
    """Own a fresh TinyPerson, bound its blocking work, and discard late results."""

    def __init__(
        self,
        config: OpenAIModelConfig | OpenAICompatibleModelConfig,
        settings: ResolvedModelSettings,
        *,
        timeout_seconds: float,
    ) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("persona timeout must be finite and positive")
        self._client = TinyModelClient(config, settings, timeout_seconds)
        self._session = TinySession(
            self._client, model=config.model, timeout_seconds=timeout_seconds
        )
        self._timeout = timeout_seconds
        self._lock = asyncio.Lock()
        self._abandoned = Event()
        self._late_cleanup_started = Event()
        self._closed = False
        self._initialized = False
        self._initializing = False
        self._completed_interactions: list[ModelInteraction] = []

    def _publish_interactions(self) -> None:
        """Release metadata only after the enclosing persona operation succeeded."""
        self._completed_interactions.extend(self._client.interactions)
        self._client.interactions.clear()

    async def _work(self, operation: Callable[[], T], label: str) -> T:
        if self._closed or self._abandoned.is_set():
            raise ModelError(label, "persona adapter is unavailable")
        async with self._lock:
            if self._closed or self._abandoned.is_set():
                raise ModelError(label, "persona adapter is unavailable")

            def run() -> T:
                if self._abandoned.is_set():
                    raise RuntimeError("persona operation expired before starting")
                return operation()

            try:
                result = await run_blocking_bounded(run, timeout_seconds=self._timeout)
            except Exception:
                self._abandoned.set()
                await self._close_completed_session()
                raise ModelError(label, "persona operation failed") from None
            if result.timed_out or result.cancelled:
                self._abandoned.set()
                self._start_late_cleanup()
                if result.cancelled:
                    raise asyncio.CancelledError
                raise ModelError(label, "persona operation timed out", retryable=True)
            return result.value  # type: ignore[return-value]

    async def _close_completed_session(self) -> None:
        """Close after a completed failed operation without exposing close errors."""
        try:
            result = await run_blocking_bounded(self._session.close, timeout_seconds=self._timeout)
        except Exception:
            return
        if result.timed_out or result.cancelled:
            self._start_late_cleanup()

    def _start_late_cleanup(self) -> None:
        """Close after an unjoinable worker releases TinyTroupe's global lock."""
        if self._late_cleanup_started.is_set():
            return
        self._late_cleanup_started.set()

        def cleanup() -> None:
            try:
                self._session.close()
            except Exception:
                pass

        Thread(target=cleanup, name="syntheticcx-tinytroupe-cleanup", daemon=True).start()

    async def initialize(
        self, definition: PersonaDefinition, scenario_context: CustomerScenarioContext
    ) -> PersonaState:
        if self._initialized or self._initializing:
            raise ModelError("persona.initialize", "persona is already initialized")
        self._initializing = True
        description = json.dumps(
            {
                "identity": definition.name or definition.id,
                "traits": definition.traits,
                "communication": definition.communication.model_dump(),
            },
            ensure_ascii=False,
        )
        goal = json.dumps(
            {
                "goal": scenario_context.goal,
                "starting_context": scenario_context.starting_context,
                "knowledge": scenario_context.knowledge,
                "instructions": (
                    "Act as the customer. TALK once to customer support, or issue "
                    "GOAL_COMPLETE / ABANDON explicitly. DONE only ends this action batch."
                ),
            },
            ensure_ascii=False,
        )
        try:

            def construct() -> None:
                self._session.initialize(description, goal)

            await self._work(construct, "persona.initialize")
            self._initialized = True
            self._publish_interactions()
        finally:
            self._initializing = False
        return definition.fresh_state()

    async def respond(
        self, observation: ConversationObservation, state: PersonaState
    ) -> PersonaAction:
        if not self._initialized:
            raise ModelError("persona.respond", "persona is not initialized")
        if state.abandoned:
            return PersonaAction(state=state, signal="abandon")
        stimulus = json.dumps(
            {
                "kind": observation.kind,
                "assistant_messages": [message.content for message in observation.messages],
                "state": state.model_dump(),
            },
            ensure_ascii=False,
        )
        records = await self._work(lambda: self._session.respond(stimulus), "persona.respond")
        try:
            action = _customer_action(records, observation, state)
        except Exception:
            self._abandoned.set()
            self._start_late_cleanup()
            raise
        self._publish_interactions()
        return action

    def take_interactions(self) -> tuple[ModelInteraction, ...]:
        """Return only completed metadata, for the later journal handoff."""
        if self._abandoned.is_set() or self._closed:
            return ()
        interactions = tuple(self._completed_interactions)
        self._completed_interactions.clear()
        return interactions

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._abandoned.is_set():
            return  # The worker still owns and eventually closes this session.
        async with self._lock:
            try:
                result = await run_blocking_bounded(
                    self._session.close, timeout_seconds=self._timeout
                )
            except Exception:
                self._abandoned.set()
                raise ModelError("persona.close", "persona close failed") from None
            if result.timed_out or result.cancelled:
                self._abandoned.set()
                if result.cancelled:
                    raise asyncio.CancelledError
                raise ModelError("persona.close", "persona close timed out", retryable=True)

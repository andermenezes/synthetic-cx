"""Ephemeral, owned graph state; raw active text never becomes a persisted snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from syntheticcx.domain.models import (
    CompletedResponse,
    ConversationObservation,
    PersonaAction,
    PersonaDefinition,
    PersonaState,
    SendReceipt,
)


class RuntimeState(TypedDict):
    persona_state: PersonaState
    observation: ConversationObservation | None
    response: CompletedResponse | None
    action: PersonaAction | None
    pending_message: str | None  # Legacy text slot; selections are carried by action directly.
    receipt: SendReceipt | None
    turn_count: int
    termination_reason: str | None


@dataclass
class RuntimeProgress:
    """Most recently acknowledged state and successful sends, even if a node later fails."""

    persona_state: PersonaState
    turn_count: int = 0


def initial_state(persona: PersonaDefinition) -> RuntimeState:
    """Create a fresh state for a single graph invocation."""
    return RuntimeState(
        persona_state=persona.fresh_state(),
        observation=None,
        response=None,
        action=None,
        pending_message=None,
        receipt=None,
        turn_count=0,
        termination_reason=None,
    )

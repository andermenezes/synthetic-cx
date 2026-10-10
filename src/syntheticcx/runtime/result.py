"""Owned termination result, independent of later evaluation and final outcome."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from syntheticcx.domain.events import TypedTraceEvent
from syntheticcx.domain.models import (
    DomainModel,
    ErrorDetail,
    EvaluationResult,
    Identifier,
    Outcome,
    PersonaState,
)

TerminationReason = Literal[
    "goal_believed_complete",
    "abandonment",
    "max_turns",
    "journey_timeout",
    "cancelled",
    "channel_error",
    "model_error",
    "behavior_error",
    "persistence_error",
]


class RuntimeResult(DomainModel):
    """Frozen result with detached committed evidence; no passed/failed outcome."""

    run_id: Identifier
    termination_reason: TerminationReason
    persona_state: PersonaState
    committed_events: tuple[TypedTraceEvent, ...]
    turn_count: int = Field(ge=0)
    error: ErrorDetail | None = None
    cleanup_completed: bool


class EvaluatedRunResult(DomainModel):
    """One-owner result; outcome is only present after durable finalization."""

    conversation: RuntimeResult
    evaluations: tuple[EvaluationResult, ...] = ()
    outcome: Outcome | None = None
    exit_code: int
    finalized: bool = False
    cleanup_completed: bool = True
    committed_events: tuple[TypedTraceEvent, ...]
    errors: tuple[ErrorDetail, ...] = ()

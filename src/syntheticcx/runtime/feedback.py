"""Supplemental same-customer reflection; no channel or evaluation dependency."""

from __future__ import annotations

import asyncio
import json
from typing import Protocol, runtime_checkable

from syntheticcx.config.models import ObservationTopic
from syntheticcx.domain.events import MessageObservedEvent
from syntheticcx.domain.models import (
    ModelInteraction,
    PersonaFeedback,
    PersonaFeedbackResponse,
    PersonaReflectionRequest,
    PersonaState,
)
from syntheticcx.domain.protocols import PersonaReflector, PersonaSimulator, WaitController
from syntheticcx.errors import ModelError
from syntheticcx.language import Language
from syntheticcx.runtime.journal import RunJournal
from syntheticcx.telemetry.run import span


@runtime_checkable
class _InteractionSource(Protocol):
    def take_interactions(self) -> tuple[ModelInteraction, ...]: ...


def validate_feedback_response(
    response: PersonaFeedbackResponse, request: PersonaReflectionRequest
) -> None:
    requested = {topic.id for topic in request.topics}
    ids = [topic.topic_id for topic in response.topics]
    if len(ids) != len(set(ids)) or set(ids) != requested:
        raise ValueError("feedback topic coverage is invalid")
    known = {message.id for message in request.messages}
    if any(ref not in known for topic in response.topics for ref in topic.evidence_message_ids):
        raise ValueError("feedback references unknown conversation evidence")


async def collect_feedback(
    *,
    simulator: PersonaSimulator,
    journal: RunJournal,
    wait: WaitController,
    topics: list[ObservationTopic],
    language: Language,
    final_state: PersonaState,
    termination_reason: str,
    timeout: float,
) -> None:
    """One bounded attempt; generation failure is data, append failure is persistence failure."""
    events = [
        event for event in journal.committed_events if isinstance(event, MessageObservedEvent)
    ]
    # Bound replayed experience, retaining actual conversation IDs. Memory stays in the same person.
    selected = []
    budget = 30_000
    clipped = False
    for event in reversed(events[-100:]):
        message = event.data.message
        text = message.content[:2000]
        options = [
            option.model_copy(update={"label": option.label[:256]}, deep=True)
            for option in message.options[:12]
        ]
        metadata = {
            key: value[:160]
            for key, value in message.metadata.items()
            if key in {"action_kind", "option_id", "source_assistant_message_id"}
            and isinstance(value, str)
        }
        excerpt = message.model_copy(
            update={"content": text, "options": options, "metadata": metadata}, deep=True
        )
        size = len(json.dumps(excerpt.model_dump(mode="json"), ensure_ascii=False))
        if size > budget:
            break
        selected.append(excerpt)
        budget -= size
        clipped |= (
            text != message.content
            or len(options) != len(message.options)
            or any(len(option.label) > 256 for option in message.options)
        )
    selected.reverse()
    truncated = len(selected) != len(events) or clipped
    partial = termination_reason != "goal_believed_complete"
    request = PersonaReflectionRequest(
        topics=topics,
        language=language,
        final_state=final_state.model_copy(deep=True),
        messages=selected,
        partial_experience=partial,
        experience_truncated=truncated,
    )
    reason = None
    response = None
    cancelled = False
    attempted = False
    if termination_reason == "cancelled":
        reason = "cancelled"
    elif not events:
        reason = "no_experience"
    elif not isinstance(simulator, PersonaReflector):
        reason = "unsupported"
    elif not simulator.reflection_available:
        reason = "persona_unavailable"
    else:
        attempted = True
        try:
            with span("persona.reflect"), span("model.call", operation="persona.reflect"):
                candidate = await wait.run(simulator.reflect(request), timeout)
            response = PersonaFeedbackResponse.model_validate(candidate.model_dump())
            validate_feedback_response(response, request)
        except asyncio.CancelledError:
            reason = "cancelled"
            cancelled = True
        except TimeoutError:
            reason = "reflection_timeout"
        except ModelError as error:
            reason = (
                "reflection_timeout"
                if error.message == "persona operation timed out"
                else "invalid_feedback"
                if error.message == "persona feedback could not be validated"
                else "reflection_failed"
            )
        except Exception:
            reason = "invalid_feedback"
    feedback = PersonaFeedback(
        status="available" if reason is None else "unavailable",
        language=language,
        topics=topics,
        response=response if reason is None else None,
        reason=reason,
        partial_experience=partial,
        experience_truncated=truncated,
    )
    committed = await journal.append(feedback)
    if attempted and isinstance(simulator, _InteractionSource):
        try:
            interactions = simulator.take_interactions()
        except Exception:
            interactions = ()  # Missing optional metadata cannot invent a test failure.
        selected_ids = {message.id for message in selected}
        inputs = [event.id for event in events if event.data.message.id in selected_ids]
        for interaction in interactions:
            await journal.append(
                interaction.model_copy(
                    update={
                        "input_event_ids": inputs,
                        "output_event_ids": [committed.id],
                    },
                    deep=True,
                )
            )
    if cancelled:
        raise asyncio.CancelledError

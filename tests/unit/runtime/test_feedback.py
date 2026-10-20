"""Independent opt-in, privacy, reference validation, and supplemental failures."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from syntheticcx.config.loader import load_config
from syntheticcx.config.models import ObservationTopic, ProjectConfig
from syntheticcx.domain.events import MessageObserved
from syntheticcx.domain.models import Message, PersonaDefinition, PersonaFeedbackResponse
from syntheticcx.errors import PersistenceError
from syntheticcx.privacy import Redactor
from syntheticcx.runtime.clock import ManualClock, ManualWaitController
from syntheticcx.runtime.feedback import collect_feedback
from syntheticcx.runtime.journal import RunJournal


@pytest.mark.parametrize(
    "topics",
    [
        [{"id": "x", "description": " "}],
        [{"id": "x", "description": "ok"}] * 2,
        [{"id": str(i), "description": "ok"} for i in range(11)],
        [{"id": "bad/id", "description": "ok"}],
    ],
)
def test_invalid_topics_are_rejected(topics):
    data = load_config(Path("examples/telecom/scenario.yaml")).model_dump()
    data["scenario"]["observation_topics"] = topics
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(data)


def test_report_language_follows_effective_persona_unless_overridden():
    data = load_config(Path("examples/telecom/scenario.yaml")).model_dump()
    data["persona"]["communication"]["language"] = "pt-BR"
    assert ProjectConfig.model_validate(data).report_language == "pt-BR"
    data["reporting"]["language"] = "en-US"
    assert ProjectConfig.model_validate(data).report_language == "en-US"


@pytest.mark.parametrize(
    "mode", ["success", "cancelled", "no_experience", "unusable", "invalid", "persistence"]
)
def test_feedback_alone_is_redacted_and_has_explicit_failure_status(mode):
    class Store:
        events = []

        async def append_event(self, run_id, event):
            if mode == "persistence" and event.type == "persona_feedback_recorded":
                raise PersistenceError("append", "failed")
            self.events.append(event)

    class Persona:
        reflection_available = mode != "unusable"
        called = 0

        async def reflect(self, request):
            self.called += 1
            return PersonaFeedbackResponse.model_validate(
                {
                    "summary": "I saw selected-secret in the answer.",
                    "topics": [
                        {
                            "topic_id": "clarity",
                            "status": "observed",
                            "impression": "selected-secret felt clear.",
                            "evidence_message_ids": ["m1" if mode != "invalid" else "invented"],
                        }
                    ],
                }
            )

    async def exercise():
        config = load_config(Path("examples/telecom/scenario.yaml"))
        assert not config.reporting.markdown.enabled
        clock = ManualClock(datetime(2026, 10, 3, tzinfo=UTC))
        store = Store()
        store.events = []
        journal = RunJournal(
            "r1", store, Redactor(config.privacy, resolved_secrets=["selected-secret"]), clock
        )
        persona = Persona()
        if mode != "no_experience":
            await journal.append(
                MessageObserved(
                    message=Message(
                        id="m1",
                        sequence=1,
                        actor="assistant",
                        content="Here is selected-secret",
                        observed_at=clock.now(),
                    )
                )
            )
        operation = collect_feedback(
            simulator=persona,
            journal=journal,
            wait=ManualWaitController(clock),
            topics=[ObservationTopic(id="clarity", description="Was it clear?")],
            language="en-US",
            final_state=PersonaDefinition.model_validate(config.persona.model_dump()).fresh_state(),
            termination_reason="cancelled" if mode == "cancelled" else "abandonment",
            timeout=1,
        )
        if mode == "persistence":
            with pytest.raises(PersistenceError):
                await operation
            assert not journal.available
            return
        await operation
        feedback = journal.committed_events[-1].data
        assert feedback.status == ("available" if mode == "success" else "unavailable")
        assert "selected-secret" not in feedback.model_dump_json()
        assert persona.called == (0 if mode in {"cancelled", "no_experience", "unusable"} else 1)
        assert feedback.partial_experience

    asyncio.run(exercise())


def test_feedback_preserves_redacted_options_and_truthful_click_metadata():
    class Store:
        async def append_event(self, _run_id, _event):
            pass

    class Persona:
        reflection_available = True
        request = None

        async def reflect(self, request):
            self.request = request
            return PersonaFeedbackResponse.model_validate(
                {
                    "summary": "I selected an information topic.",
                    "topics": [
                        {
                            "topic_id": "clarity",
                            "status": "observed",
                            "impression": "Clear menu.",
                            "evidence_message_ids": ["reply", "choice"],
                        }
                    ],
                }
            )

    async def exercise():
        config = load_config(Path("examples/telecom/scenario.yaml"))
        clock = ManualClock(datetime(2026, 10, 3, tzinfo=UTC))
        journal = RunJournal(
            "r", Store(), Redactor(config.privacy, resolved_secrets=["private-label"]), clock
        )
        reply = Message.model_validate(
            {
                "id": "reply",
                "sequence": 0,
                "actor": "assistant",
                "content": "",
                "observed_at": clock.now(),
                "options": [{"id": "opaque-option", "label": "Information private-label"}],
            }
        )
        choice = Message(
            id="choice",
            sequence=1,
            actor="customer",
            content="Information private-label",
            observed_at=clock.now(),
            metadata={
                "action_kind": "select_option",
                "option_id": "opaque-option",
                "source_assistant_message_id": "reply",
            },
        )
        for message in (reply, choice):
            await journal.append(MessageObserved(message=message))
        persona = Persona()
        await collect_feedback(
            simulator=persona,
            journal=journal,
            wait=ManualWaitController(clock),
            topics=[ObservationTopic(id="clarity", description="Was the menu clear?")],
            language="en-US",
            final_state=PersonaDefinition.model_validate(config.persona.model_dump()).fresh_state(),
            termination_reason="max_turns",
            timeout=1,
        )
        assert persona.request is not None
        assert "private-label" not in persona.request.model_dump_json()
        assert persona.request.messages[0].content == ""
        assert persona.request.messages[0].options[0].id == "opaque-option"
        assert persona.request.messages[1].metadata["action_kind"] == "select_option"
        assert not persona.request.experience_truncated

    asyncio.run(exercise())

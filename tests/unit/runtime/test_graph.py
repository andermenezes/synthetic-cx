"""Single-run LangGraph routing against deterministic protocol fakes."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from syntheticcx.config.loader import load_config
from syntheticcx.config.models import BehaviorRule, PrivacyConfig
from syntheticcx.domain.events import TypedTraceEvent
from syntheticcx.domain.models import (
    CompletedResponse,
    ConversationObservation,
    CustomerScenarioContext,
    Message,
    ModelIdentity,
    ModelInteraction,
    PersonaAction,
    PersonaDefinition,
    PersonaState,
    Scenario,
    SendReceipt,
    TokenUsage,
)
from syntheticcx.errors import ChannelError
from syntheticcx.privacy import Redactor
from syntheticcx.runtime.clock import ManualClock, ManualWaitController, RealWaitController
from syntheticcx.runtime.graph import build_graph
from syntheticcx.runtime.journal import RunJournal
from syntheticcx.runtime.state import initial_state

NOW = datetime(2026, 9, 24, tzinfo=UTC)


class Clock:
    def now(self) -> datetime:
        return NOW


class Storage:
    def __init__(self) -> None:
        self.events: list[TypedTraceEvent] = []

    async def append_event(self, run_id: str, event: TypedTraceEvent) -> None:
        assert run_id == "run-1"
        self.events.append(event)


def response(*parts: tuple[str, str]) -> CompletedResponse:
    return CompletedResponse(
        messages=[
            Message(id=identifier, actor="assistant", content=text, observed_at=NOW, sequence=index)
            for index, (identifier, text) in enumerate(parts)
        ],
        started_at=NOW,
        completed_at=NOW,
        completion_strategy="completion_locator",
    )


def model_interaction(
    identifier: str, *, model: str = "fake", requested_seed: int | None = None
) -> ModelInteraction:
    return ModelInteraction(
        id=identifier,
        identity=ModelIdentity(provider="fake", model=model, requested_seed=requested_seed),
        started_at=NOW,
        ended_at=NOW,
        usage=TokenUsage(unavailable_reason="not reported"),
    )


class Channel:
    def __init__(self, responses: list[CompletedResponse], *, fail_send: bool = False) -> None:
        self.responses = iter(responses)
        self.sent: list[str] = []
        self.fail_send = fail_send

    async def send(self, message: str) -> SendReceipt:
        if self.fail_send:
            raise RuntimeError("send failed")
        self.sent.append(message)
        return SendReceipt(message_id=f"customer-{len(self.sent)}", submitted_at=NOW)

    async def receive(self) -> CompletedResponse:
        return next(self.responses)


class Persona:
    def __init__(self, messages: list[str | None]) -> None:
        self.messages = iter(messages)
        self.observations: list[ConversationObservation] = []
        self.states: list[PersonaState] = []

    async def initialize(
        self, definition: PersonaDefinition, scenario_context: CustomerScenarioContext
    ) -> PersonaState:
        assert scenario_context.goal
        return definition.fresh_state()

    async def respond(
        self, observation: ConversationObservation, state: PersonaState
    ) -> PersonaAction:
        self.observations.append(observation)
        self.states.append(state)
        message = next(self.messages)
        return PersonaAction(
            message=message,
            state=state,
            signal="continue" if message is not None else "goal_believed_complete",
        )


def execute(
    channel: Channel,
    persona: Persona,
    *,
    greeting_grace: float = 0,
    max_turns: int = 4,
) -> tuple[dict[str, Any], list[TypedTraceEvent]]:
    config = load_config(Path("examples/telecom/scenario.yaml"))
    definition = PersonaDefinition.model_validate(config.persona.model_dump())
    scenario = Scenario.model_validate(config.scenario.model_dump())
    limits = config.execution.model_copy(
        update={"greeting_grace_seconds": greeting_grace, "max_turns": max_turns}
    )
    storage = Storage()
    journal = RunJournal(
        "run-1",
        storage,
        Redactor(PrivacyConfig()),
        Clock(),
        id_factory=iter(f"event-{index}" for index in range(100)).__next__,
    )
    graph = build_graph(
        scenario=scenario,
        persona=definition,
        channel=channel,
        simulator=persona,
        journal=journal,
        wait=RealWaitController(),
        limits=limits,
    )

    async def invoke() -> dict[str, Any]:
        return await graph.ainvoke(initial_state(definition))

    result = asyncio.run(invoke())
    assert storage.events == list(journal.committed_events)
    assert [event.sequence for event in storage.events] == list(range(len(storage.events)))
    return result, storage.events


def test_no_greeting_then_multipart_response_is_one_observation() -> None:
    channel = Channel([response(("a", "Thanks for your account number"), ("b", "Checking now"))])
    persona = Persona(["My account number is 123", None])
    result, events = execute(channel, persona)
    assert result["termination_reason"] == "goal_believed_complete"
    assert result["turn_count"] == 1
    assert [observation.kind for observation in persona.observations] == [
        "conversation_started",
        "assistant_response",
    ]
    assert [message.id for message in persona.observations[1].messages] == ["a", "b"]
    assert [event.type for event in events] == [
        "run_lifecycle_changed",
        "conversation_started",
        "message_observed",
        "persona_state_changed",
        "message_observed",
        "message_observed",
    ]
    assert events[2].data.message.id == "customer-1"
    assert events[2].data.message.content == "My account number is 123"
    assert result["persona_state"].turn_number == 1


def test_hidden_evaluator_criteria_never_reach_persona_on_subsequent_turns() -> None:
    marker = "EVALUATOR-ONLY-MARKER-9D"
    config = load_config(Path("examples/telecom/scenario.yaml"))
    scenario = Scenario.model_validate(config.scenario.model_dump())
    criteria = [item.model_copy(deep=True) for item in scenario.success_criteria]
    goal_index = next(
        index
        for index, item in enumerate(criteria)
        if item.evaluator_type == "semantic_goal_completion"
    )
    goal = criteria[goal_index]
    goal = goal.model_copy(update={"result_kind": "scored", "threshold": 0.73})
    goal.parameters["expected_facts"] = [marker]
    criteria[goal_index] = goal
    scenario = scenario.model_copy(update={"success_criteria": criteria})
    captured: list[str] = []

    class CapturingPersona(Persona):
        async def initialize(
            self, definition: PersonaDefinition, context: CustomerScenarioContext
        ) -> PersonaState:
            captured.append(context.model_dump_json())
            return await super().initialize(definition, context)

        async def respond(
            self, observation: ConversationObservation, state: PersonaState
        ) -> PersonaAction:
            captured.append(observation.model_dump_json() + state.model_dump_json())
            return await super().respond(observation, state)

    persona = CapturingPersona(["First question", "Second question", None])
    storage = Storage()
    journal = RunJournal("run-1", storage, Redactor(PrivacyConfig()), Clock())
    channel = Channel([response(("a", "First answer")), response(("b", "Second answer"))])
    definition = PersonaDefinition.model_validate(config.persona.model_dump())
    graph = build_graph(
        scenario=scenario,
        persona=definition,
        channel=channel,
        simulator=persona,
        journal=journal,
        wait=RealWaitController(),
        limits=config.execution.model_copy(update={"greeting_grace_seconds": 0}),
    )
    result = asyncio.run(graph.ainvoke(initial_state(definition)))
    assert result["termination_reason"] == "goal_believed_complete"
    assert len(persona.observations) == 3
    assert len(captured) == 4
    assert all(
        marker not in item
        and "expected_facts" not in item
        and "threshold" not in item
        and "0.73" not in item
        for item in captured
    )
    assert marker in goal.model_dump_json()
    assert goal.threshold == 0.73


def test_greeting_is_committed_before_persona_and_terminal_action_never_sends() -> None:
    channel = Channel([response(("hello", "How can I help?"))])
    persona = Persona([None])
    result, events = execute(channel, persona, greeting_grace=1)
    assert result["turn_count"] == 0
    assert channel.sent == []
    assert persona.observations[0].messages[0].id == "hello"
    assert [event.type for event in events] == [
        "run_lifecycle_changed",
        "message_observed",
        "conversation_started",
    ]
    assert events[-1].data.greeting_message_ids == ["hello"]


def test_repetition_event_and_rule_state_are_committed_before_next_response() -> None:
    channel = Channel([response(("ask", "Please give your ACCOUNT-number again"))])
    persona = Persona(["My account number is 123", None])
    result, events = execute(channel, persona)
    assert result["persona_state"].frustration == pytest.approx(0.3)
    assert persona.states[1].frustration == pytest.approx(0.3)
    assert [event.type for event in events][-3:] == [
        "message_observed",
        "repeated_information",
        "persona_state_changed",
    ]
    assert events[-2].data.original_message_id == "customer-1"
    assert events[-2].data.request_message_id == "ask"
    assert events[-1].data.trigger_ids[0] == events[-2].id


def test_max_turns_never_allows_an_extra_send() -> None:
    channel = Channel([response(("answer", "Your bill is higher"))])
    persona = Persona(["Why is it higher?", "I would ask more"])
    result, _ = execute(channel, persona, max_turns=1)
    assert result["turn_count"] == 1
    assert result["termination_reason"] == "max_turns"
    assert channel.sent == ["Why is it higher?"]


def test_failed_send_does_not_increment_turns_or_create_customer_evidence() -> None:
    channel = Channel([], fail_send=True)
    persona = Persona(["My account number is 123"])
    config = load_config(Path("examples/telecom/scenario.yaml"))
    definition = PersonaDefinition.model_validate(config.persona.model_dump())
    scenario = Scenario.model_validate(config.scenario.model_dump())
    storage = Storage()
    journal = RunJournal("run-1", storage, Redactor(PrivacyConfig()), Clock())
    graph = build_graph(
        scenario=scenario,
        persona=definition,
        channel=channel,
        simulator=persona,
        journal=journal,
        wait=RealWaitController(),
        limits=config.execution.model_copy(update={"greeting_grace_seconds": 0}),
    )

    async def invoke() -> None:
        with pytest.raises(ChannelError, match="customer send failed"):
            await graph.ainvoke(initial_state(definition))

    asyncio.run(invoke())
    assert channel.sent == []
    assert [event.type for event in storage.events] == [
        "run_lifecycle_changed",
        "conversation_started",
    ]


def test_abandonment_after_repetition_skips_persona_and_additional_send() -> None:
    config = load_config(Path("examples/telecom/scenario.yaml"))
    definition = PersonaDefinition.model_validate(config.persona.model_dump()).model_copy(
        update={
            "behavioral_rules": [
                BehaviorRule(
                    id="give-up",
                    priority=0,
                    event="repeated_information",
                    willingness_to_continue_delta=-1,
                    abandonment_threshold=0.1,
                )
            ]
        }
    )
    storage = Storage()
    journal = RunJournal("run-1", storage, Redactor(PrivacyConfig()), Clock())
    channel = Channel([response(("ask", "Please repeat the account number?"))])
    persona = Persona(["My account number is 123", "should not be called"])
    graph = build_graph(
        scenario=Scenario.model_validate(config.scenario.model_dump()),
        persona=definition,
        channel=channel,
        simulator=persona,
        journal=journal,
        wait=RealWaitController(),
        limits=config.execution.model_copy(update={"greeting_grace_seconds": 0}),
    )

    async def invoke() -> dict[str, Any]:
        return await graph.ainvoke(initial_state(definition))

    result = asyncio.run(invoke())
    assert result["termination_reason"] == "abandonment"
    assert result["persona_state"].abandoned
    assert len(persona.observations) == 1
    assert channel.sent == ["My account number is 123"]


def test_manual_greeting_deadline_produces_no_greeting_observation() -> None:
    config = load_config(Path("examples/telecom/scenario.yaml"))
    definition = PersonaDefinition.model_validate(config.persona.model_dump())
    clock = ManualClock(NOW)
    storage = Storage()
    journal = RunJournal("run-1", storage, Redactor(PrivacyConfig()), clock)
    receiving = asyncio.Event()

    class SilentChannel(Channel):
        async def receive(self) -> CompletedResponse:
            receiving.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    channel = SilentChannel([])
    persona = Persona([None])
    graph = build_graph(
        scenario=Scenario.model_validate(config.scenario.model_dump()),
        persona=definition,
        channel=channel,
        simulator=persona,
        journal=journal,
        wait=ManualWaitController(clock),
        limits=config.execution.model_copy(update={"greeting_grace_seconds": 2}),
    )

    async def invoke() -> dict[str, Any]:
        task = asyncio.create_task(graph.ainvoke(initial_state(definition)))
        await receiving.wait()
        await clock.advance(2)
        return await task

    result = asyncio.run(invoke())
    assert result["termination_reason"] == "goal_believed_complete"
    assert persona.observations[0].kind == "conversation_started"
    assert storage.events[-1].type == "conversation_started"
    assert not storage.events[-1].data.greeting_observed


def test_persona_interactions_are_acknowledged_before_customer_send_and_routing() -> None:
    class EvidencePersona(Persona):
        def __init__(self) -> None:
            super().__init__(["Why is my bill higher?", None])
            self.pending: list[ModelInteraction] = []

        async def respond(
            self, observation: ConversationObservation, state: PersonaState
        ) -> PersonaAction:
            result = await super().respond(observation, state)
            self.pending.append(
                model_interaction(
                    f"call-{len(self.observations)}", model="example", requested_seed=42
                )
            )
            return result

        def take_interactions(self) -> tuple[ModelInteraction, ...]:
            result = tuple(self.pending)
            self.pending.clear()
            return result

    persona = EvidencePersona()
    result, events = execute(Channel([response(("answer", "Your plan changed"))]), persona)
    assert result["termination_reason"] == "goal_believed_complete"
    interactions = [event for event in events if event.type == "model_interaction_completed"]
    assert len(interactions) == 2
    assert [event.type for event in events] == [
        "run_lifecycle_changed",
        "conversation_started",
        "model_interaction_completed",
        "message_observed",
        "persona_state_changed",
        "message_observed",
        "model_interaction_completed",
    ]
    assert interactions[0].data.input_event_ids == [events[1].id]
    assert interactions[1].data.input_event_ids == [events[5].id]
    for event in interactions:
        assert event.data.output_event_ids == []
        assert "output_event_not_yet_committed" in event.data.capture_omissions
        assert event.data.usage.unavailable_reason is not None
        assert all(
            any(previous.id == ref for previous in events[: event.sequence])
            for ref in event.data.input_event_ids
        )


def test_initialization_interaction_references_committed_lifecycle() -> None:
    class InitializingPersona(Persona):
        def __init__(self) -> None:
            super().__init__([None])
            self.pending: tuple[ModelInteraction, ...] = ()

        async def initialize(
            self, definition: PersonaDefinition, scenario_context: CustomerScenarioContext
        ) -> PersonaState:
            self.pending = (model_interaction("initial-call"),)
            return await super().initialize(definition, scenario_context)

        def take_interactions(self) -> tuple[ModelInteraction, ...]:
            result, self.pending = self.pending, ()
            return result

    _, events = execute(Channel([]), InitializingPersona())
    assert [event.type for event in events] == [
        "run_lifecycle_changed",
        "model_interaction_completed",
        "conversation_started",
    ]
    assert events[1].data.input_event_ids == [events[0].id]
    assert events[1].data.output_event_ids == []


def test_interaction_references_all_ordered_greeting_bubbles() -> None:
    class EvidencePersona(Persona):
        def __init__(self) -> None:
            super().__init__([None])
            self.pending: tuple[ModelInteraction, ...] = ()

        async def respond(
            self, observation: ConversationObservation, state: PersonaState
        ) -> PersonaAction:
            result = await super().respond(observation, state)
            self.pending = (model_interaction("greeting-call"),)
            return result

        def take_interactions(self) -> tuple[ModelInteraction, ...]:
            result, self.pending = self.pending, ()
            return result

    _, events = execute(
        Channel([response(("greeting-1", "Hello"), ("greeting-2", "How can I help?"))]),
        EvidencePersona(),
        greeting_grace=1,
    )
    assert [event.type for event in events] == [
        "run_lifecycle_changed",
        "message_observed",
        "message_observed",
        "conversation_started",
        "model_interaction_completed",
    ]
    assert events[-1].data.input_event_ids == [events[1].id, events[2].id]

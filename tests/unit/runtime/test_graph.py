"""Single-run LangGraph routing against deterministic protocol fakes."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from syntheticcx.config.loader import load_config
from syntheticcx.config.models import BehaviorConfig, BehaviorRule, PrivacyConfig
from syntheticcx.domain.behavior import BehavioralSignal
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
    SelectionReceipt,
    SendReceipt,
    TokenUsage,
)
from syntheticcx.errors import BehaviorError, ChannelError
from syntheticcx.privacy import Redactor
from syntheticcx.runtime.behavior_signals import (
    BehaviorDetectors,
    DetectorRegistration,
    EmptyDetectorParameters,
    prepare_behavior,
)
from syntheticcx.runtime.clock import ManualClock, ManualWaitController, RealWaitController
from syntheticcx.runtime.graph import build_graph
from syntheticcx.runtime.journal import RunJournal
from syntheticcx.runtime.state import initial_state

NOW = datetime(2026, 9, 24, tzinfo=UTC)


def test_initial_menu_reaches_persona_with_zero_grace_and_no_startup_probe():
    class InitialMenuChannel(Channel):
        initial_reads = 0
        selected = []

        async def observe_initial(self):
            self.initial_reads += 1
            return CompletedResponse.model_validate(
                {
                    "messages": [
                        {
                            "id": "initial-menu",
                            "sequence": 0,
                            "actor": "assistant",
                            "content": "Choose a topic",
                            "observed_at": NOW,
                            "options": [{"id": "current-choice", "label": "Plans"}],
                        }
                    ],
                    "started_at": NOW,
                    "completed_at": NOW,
                    "completion_strategy": "quiet_interval",
                }
            )

        async def select_option(self, option_id):
            self.selected.append(option_id)
            return SelectionReceipt(
                message_id="selected",
                submitted_at=NOW,
                selected_option_id=option_id,
                selected_label="Plans",
                source_message_id="initial-menu",
            )

    class Customer(Persona):
        async def respond(self, observation, state):
            self.observations.append(observation)
            if len(self.observations) == 1:
                assert observation.messages[-1].options[0].label == "Plans"
                return PersonaAction(
                    state=state, signal="continue", selected_option_id="current-choice"
                )
            return PersonaAction(state=state, signal="goal_believed_complete")

    async def exercise():
        config = load_config(Path("examples/telecom/scenario.yaml"))
        definition = PersonaDefinition.model_validate(config.persona.model_dump())
        channel = InitialMenuChannel([response(("after-choice", "Plan details"))])
        customer = Customer([])
        storage = Storage()
        graph = build_graph(
            scenario=Scenario.model_validate(config.scenario.model_dump()),
            persona=definition,
            channel=channel,
            simulator=customer,
            journal=RunJournal("run-1", storage, Redactor(PrivacyConfig()), Clock()),
            wait=RealWaitController(),
            limits=config.execution.model_copy(update={"greeting_grace_seconds": 0}),
        )
        result = await graph.ainvoke(initial_state(definition))
        assert result["turn_count"] == 1
        assert channel.initial_reads == 1
        assert channel.selected == ["current-choice"]
        assert channel.sent == []
        assert customer.observations[1].messages[0].id == "after-choice"
        completed = [
            event.data.signal
            for event in storage.events
            if event.type == "behavioral_signal_recorded"
        ]
        assert [item.signal for item in completed] == ["customer_turn_completed"]
        assert completed[0].source_message_ids == ("after-choice",)
        assert not any(event.type == "repeated_information" for event in storage.events)

    asyncio.run(exercise())


def test_failed_initial_observation_never_falls_back_to_text_or_receive():
    class FailedInitialChannel(Channel):
        async def observe_initial(self):
            raise ChannelError("channel.initial", "current state is ambiguous")

        async def receive(self):
            pytest.fail("an initial-state failure must not enter the greeting fallback")

    channel = FailedInitialChannel([])
    customer = Persona(["This must not be sent"])
    with pytest.raises(ChannelError, match="initial chat observation failed"):
        execute(channel, customer, greeting_grace=1)
    assert channel.sent == []
    assert customer.observations == []


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
    def __init__(
        self, messages: list[str | None], discarded_action_counts: list[int] | None = None
    ) -> None:
        self.messages = iter(messages)
        self.discarded_action_counts = iter(discarded_action_counts or [])
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
            discarded_action_count=next(self.discarded_action_counts, 0),
        )


def execute(
    channel: Channel,
    persona: Persona,
    *,
    greeting_grace: float = 0,
    max_turns: int = 4,
    rules: list[BehaviorRule] | None = None,
    behavior_config: BehaviorConfig | None = None,
    behavior_detectors: BehaviorDetectors | None = None,
    privacy: PrivacyConfig | None = None,
) -> tuple[dict[str, Any], list[TypedTraceEvent]]:
    config = load_config(Path("examples/telecom/scenario.yaml"))
    definition = PersonaDefinition.model_validate(config.persona.model_dump())
    if rules is not None:
        definition = definition.model_copy(update={"behavioral_rules": rules})
    scenario = Scenario.model_validate(config.scenario.model_dump())
    limits = config.execution.model_copy(
        update={"greeting_grace_seconds": greeting_grace, "max_turns": max_turns}
    )
    storage = Storage()
    journal = RunJournal(
        "run-1",
        storage,
        Redactor(privacy or PrivacyConfig()),
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
        behavior_config=behavior_config,
        behavior_detectors=behavior_detectors,
    )

    async def invoke() -> dict[str, Any]:
        return await graph.ainvoke(initial_state(definition))

    result = asyncio.run(invoke())
    assert storage.events == list(journal.committed_events)
    assert [event.sequence for event in storage.events] == list(range(len(storage.events)))
    return result, storage.events


def test_initial_observer_without_greeting_does_not_receive_before_submission() -> None:
    class NoGreetingChannel(Channel):
        async def observe_initial(self):
            return None

        async def receive(self):
            assert self.sent, "receive requires a pending customer submission"
            return await super().receive()

    channel = NoGreetingChannel([response(("reply", "Coverage details"))])
    persona = Persona(["Please explain coverage", None])
    result, _ = execute(channel, persona, greeting_grace=1)
    assert result["termination_reason"] == "goal_believed_complete"
    assert result["turn_count"] == 1
    assert persona.observations[0].kind == "conversation_started"


def test_multiple_persona_choices_are_recorded_and_first_action_continues() -> None:
    channel = Channel(
        [response(("assistant-2", "Thanks, please continue"), ("assistant-3", "More details"))]
    )
    persona = Persona(["Please explain the plans", None], discarded_action_counts=[2, 0])
    result, events = execute(channel, persona)
    ambiguity = next(event for event in events if event.type == "persona_output_ambiguity_observed")
    assert ambiguity.data.candidate_action_count == 3
    assert ambiguity.data.selected_action_order == 1
    assert ambiguity.data.discarded_action_count == 2
    assert result["turn_count"] == 1
    assert channel.sent == ["Please explain the plans"]
    assert [message.content for message in persona.observations[1].messages] == [
        "Thanks, please continue",
        "More details",
    ]


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
    assert [message.sequence for message in persona.observations[1].messages] == [1, 2]
    customer = next(
        event.data.message
        for event in events
        if event.type == "message_observed" and event.data.message.actor == "customer"
    )
    assert customer.id == "customer-1"
    assert customer.content == "My account number is 123"
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
    started = next(event for event in events if event.type == "conversation_started")
    boundary = next(event for event in events if event.type == "behavior_observation_recorded")
    assert started.data.greeting_message_ids == ["hello"]
    assert started.sequence < boundary.sequence
    assert boundary.data.kind == "greeting"
    assert boundary.data.signal_count == 0


def test_repetition_event_and_rule_state_are_committed_before_next_response() -> None:
    channel = Channel([response(("ask", "Please give your ACCOUNT-number again"))])
    persona = Persona(["My account number is 123", None])
    result, events = execute(channel, persona)
    assert result["persona_state"].frustration == pytest.approx(0.3)
    assert persona.states[1].frustration == pytest.approx(0.3)
    fact = next(event for event in events if event.type == "repeated_information")
    signal = next(
        event
        for event in events
        if event.type == "behavioral_signal_recorded"
        and event.data.signal.signal == "repeated_information"
    )
    transition = next(
        event
        for event in events
        if event.type == "behavioral_state_changed" and event.data.signal == "repeated_information"
    )
    assert fact.data.original_message_id == "customer-1"
    assert fact.data.request_message_id == "ask"
    assert signal.data.signal.source_event_ids == (fact.id,)
    assert transition.data.signal_event_id == signal.id
    assert fact.sequence < signal.sequence < transition.sequence


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


def test_abandonment_after_repetition_skips_persona_and_additional_send() -> None:
    config = load_config(Path("examples/telecom/scenario.yaml"))
    definition = PersonaDefinition.model_validate(config.persona.model_dump()).model_copy(
        update={
            "behavioral_rules": [
                BehaviorRule(
                    id="give-up",
                    priority=0,
                    when={"signal": "repeated_information"},
                    effects={
                        "willingness_to_continue_delta": -1,
                        "abandonment_threshold": 0.1,
                    },
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
    started = next(event for event in storage.events if event.type == "conversation_started")
    assert not started.data.greeting_observed


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
    started = next(event for event in events if event.type == "conversation_started")
    answer = next(
        event
        for event in events
        if event.type == "message_observed" and event.data.message.id == "answer"
    )
    assert interactions[0].data.input_event_ids == [started.id]
    assert interactions[1].data.input_event_ids == [answer.id]
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
    assert events[-1].data.input_event_ids == [events[1].id, events[2].id]


def test_completed_turn_rule_excludes_greeting_and_stops_before_next_persona_call() -> None:
    channel = Channel(
        [
            response(("greeting", "Hello")),
            response(("first", "First explanation")),
            response(("second", "Second explanation")),
        ]
    )
    persona = Persona(["Explain the price", "Explain the conditions", "must not dispatch"])
    result, events = execute(
        channel,
        persona,
        greeting_grace=1,
        rules=[
            BehaviorRule.model_validate(
                {
                    "id": "leave",
                    "priority": 0,
                    "when": {"signal": "customer_turn_completed", "occurrence_at_least": 2},
                    "effects": {"abandon": True},
                    "fire": "once",
                }
            )
        ],
    )
    assert result["termination_reason"] == "abandonment"
    assert result["turn_count"] == 2
    assert channel.sent == ["Explain the price", "Explain the conditions"]
    assert len(persona.observations) == 2
    transition = next(
        event for event in reversed(events) if event.type == "behavioral_state_changed"
    )
    assert transition.data.applied_rule_ids == ["leave"]
    signals = [event for event in events if event.type == "behavioral_signal_recorded"]
    assert [event.data.signal.signal for event in signals] == [
        "customer_turn_completed",
        "customer_turn_completed",
    ]
    assert [
        event.data.occurrence for event in events if event.type == "behavioral_state_changed"
    ] == [1, 2]
    assert transition.data.signal_event_id == signals[-1].id


def test_disabled_behavior_detectors_do_not_suppress_formal_repetition_facts() -> None:
    channel = Channel([response(("request", "Please repeat your account number?"))])
    persona = Persona(["My account number is 123", None])
    result, events = execute(
        channel, persona, rules=[], behavior_config=BehaviorConfig(detectors=[])
    )
    assert result["termination_reason"] == "goal_believed_complete"
    repetitions = [event for event in events if event.type == "repeated_information"]
    assert len(repetitions) == 1
    customer = next(
        event.data.message
        for event in events
        if event.type == "message_observed" and event.data.message.actor == "customer"
    )
    assert repetitions[0].data.original_message_id == customer.id
    assert repetitions[0].data.request_message_id == "request"
    assert persona.states[-1].frustration == persona.states[0].frustration


def test_completed_batch_commits_all_signals_before_effects_and_freezes_terminal_state() -> None:
    observed = []

    def factory(metadata, parameters):
        class Detector:
            async def detect(self, boundary):
                observed.append(boundary.model_copy(deep=True))
                if boundary.kind == "greeting":
                    return ()
                return tuple(
                    BehavioralSignal(
                        detector_id=metadata.detector_id,
                        detector_type=metadata.detector_type,
                        detector_version=metadata.detector_version,
                        signal=name,
                        deduplication_key=f"{boundary.observation_id}-{name}",
                        observation_id=boundary.observation_id,
                        source_event_ids=boundary.evidence_event_ids,
                        source_message_ids=tuple(message.id for message in boundary.messages),
                    )
                    for name in ("leave_signal", "recover_signal")
                )

        return Detector()

    registered = DetectorRegistration(
        detector_type="custom",
        version="1",
        parameter_model=EmptyDetectorParameters,
        signal_schemas={
            "leave_signal": EmptyDetectorParameters,
            "recover_signal": EmptyDetectorParameters,
        },
        mode="deterministic",
        factory=factory,
    )
    selected = BehaviorConfig.model_validate({"detectors": [{"id": "consumer", "type": "custom"}]})
    rules = [
        BehaviorRule(
            id="leave", priority=0, when={"signal": "leave_signal"}, effects={"abandon": True}
        ),
        BehaviorRule(
            id="recover",
            priority=0,
            when={"signal": "recover_signal"},
            effects={"frustration_delta": 0.9},
        ),
    ]
    detectors = prepare_behavior(selected, rules, {"custom": registered}).instantiate()
    channel = Channel([response(("response", "Done"))])
    persona = Persona(["Ask once", "must not send"])
    result, events = execute(channel, persona, rules=rules, behavior_detectors=detectors)
    signals = [event for event in events if event.type == "behavioral_signal_recorded"]
    transitions = [event for event in events if event.type == "behavioral_state_changed"]
    assert len(signals) == len(transitions) == 2
    assert max(event.sequence for event in signals) < min(event.sequence for event in transitions)
    assert transitions[0].data.applied_rule_ids == ["leave"]
    assert transitions[1].data.applied_rule_ids == []
    assert transitions[1].data.before == transitions[1].data.after == transitions[0].data.after
    assert result["termination_reason"] == "abandonment"
    assert channel.sent == ["Ask once"]
    assert len(persona.observations) == 1
    assert [boundary.kind for boundary in observed] == ["greeting", "customer_turn_completed"]
    assert observed[-1].last_customer_message.content == "Ask once"


def test_once_memory_and_signal_counters_reset_for_each_run() -> None:
    rules = [
        BehaviorRule(
            id="once",
            priority=0,
            when={"signal": "customer_turn_completed"},
            effects={"frustration_delta": 0.2},
            fire="once",
        )
    ]
    for _ in range(2):
        result, events = execute(
            Channel([response(("a", "First")), response(("b", "Second"))]),
            Persona(["First ask", "Second ask", None]),
            rules=rules,
        )
        transitions = [event for event in events if event.type == "behavioral_state_changed"]
        assert [event.data.occurrence for event in transitions] == [1, 2]
        assert [event.data.applied_rule_ids for event in transitions] == [["once"], []]
        assert result["persona_state"].frustration == pytest.approx(0.3)


def test_no_match_boundaries_are_persisted_when_detectors_are_disabled() -> None:
    _, events = execute(
        Channel([response(("answer", "Answer"))]),
        Persona(["Question", None]),
        rules=[],
        behavior_config=BehaviorConfig(detectors=[]),
    )
    boundaries = [event for event in events if event.type == "behavior_observation_recorded"]
    assert [event.data.kind for event in boundaries] == ["greeting", "customer_turn_completed"]
    assert [event.data.signal_count for event in boundaries] == [0, 0]
    assert all(event.data.detector_ids == [] for event in boundaries)
    assert not any(
        event.type in {"behavioral_signal_recorded", "behavioral_state_changed"} for event in events
    )


def test_detectors_receive_original_active_text_but_trace_payload_is_privacy_filtered() -> None:
    observed = []

    class TextPayload(EmptyDetectorParameters):
        text: str

    def factory(metadata, parameters):
        class Detector:
            async def detect(self, boundary):
                observed.append(boundary.model_copy(deep=True))
                if not boundary.messages:
                    return ()
                return (
                    BehavioralSignal(
                        detector_id=metadata.detector_id,
                        detector_type=metadata.detector_type,
                        detector_version=metadata.detector_version,
                        signal="visible_notice",
                        deduplication_key=str(boundary.observation_id),
                        observation_id=boundary.observation_id,
                        source_event_ids=boundary.evidence_event_ids,
                        source_message_ids=tuple(message.id for message in boundary.messages),
                        payload={"text": boundary.messages[0].content},
                    ),
                )

        return Detector()

    registered = DetectorRegistration(
        detector_type="custom",
        version="1",
        parameter_model=EmptyDetectorParameters,
        signal_schemas={"visible_notice": TextPayload},
        mode="deterministic",
        factory=factory,
    )
    selected = BehaviorConfig.model_validate({"detectors": [{"id": "consumer", "type": "custom"}]})
    detectors = prepare_behavior(selected, [], {"custom": registered}).instantiate()
    _, events = execute(
        Channel([response(("reply", "Customer email alice@example.test"))]),
        Persona(["My email alice@example.test", None]),
        rules=[],
        behavior_detectors=detectors,
        privacy=PrivacyConfig(synthetic_id_patterns=[r"alice@example\.test"]),
    )
    assert observed[-1].messages[0].content == "Customer email alice@example.test"
    assert observed[-1].last_customer_message.content == "My email alice@example.test"
    assert "alice@example.test" not in "\n".join(event.model_dump_json() for event in events)
    recorded = next(event for event in events if event.type == "behavioral_signal_recorded")
    assert recorded.data.signal.source_message_ids == ("reply",)
    assert "alice@example.test" not in recorded.data.signal.payload["text"]


def test_detector_failure_stops_before_customer_dispatch() -> None:
    class Detector:
        async def detect(self, boundary):
            raise RuntimeError("private detector failure")

    selected = BehaviorConfig.model_validate({"detectors": [{"id": "consumer", "type": "custom"}]})
    registered = DetectorRegistration(
        detector_type="custom",
        version="1",
        parameter_model=EmptyDetectorParameters,
        signal_schemas={"notice": EmptyDetectorParameters},
        mode="deterministic",
        factory=lambda metadata, parameters: Detector(),
    )
    detectors = prepare_behavior(selected, [], {"custom": registered}).instantiate()
    channel = Channel([])
    persona = Persona(["must not send"])
    with pytest.raises(BehaviorError) as caught:
        execute(channel, persona, rules=[], behavior_detectors=detectors)
    assert "private detector failure" not in str(caught.value)
    assert channel.sent == []
    assert persona.observations == []


def test_six_completed_option_turns_abandon_without_typed_supply_or_extra_action() -> None:
    def menu(identifier):
        reply = response((identifier, "Please repeat your account number?"))
        message = Message.model_validate(
            {
                **reply.messages[0].model_dump(),
                "options": [{"id": "again", "label": "My account number is 123"}],
            }
        )
        return reply.model_copy(update={"messages": [message]})

    class OptionChannel(Channel):
        def __init__(self):
            super().__init__([menu(f"reply-{index}") for index in range(1, 7)])
            self.selected = []

        async def observe_initial(self):
            return menu("greeting")

        async def select_option(self, option_id):
            self.selected.append(option_id)
            count = len(self.selected)
            return SelectionReceipt(
                message_id=f"selection-{count}",
                submitted_at=NOW,
                selected_option_id=option_id,
                selected_label="My account number is 123",
                source_message_id="greeting" if count == 1 else f"reply-{count - 1}",
            )

    class OptionPersona(Persona):
        async def respond(self, observation, state):
            self.observations.append(observation)
            return PersonaAction(state=state, signal="continue", selected_option_id="again")

    channel, simulator = OptionChannel(), OptionPersona([])
    result, events = execute(
        channel,
        simulator,
        max_turns=10,
        rules=[
            BehaviorRule(
                id="leave-after-six",
                priority=0,
                when={"signal": "customer_turn_completed", "occurrence_at_least": 6},
                effects={"abandon": True},
                fire="once",
            )
        ],
    )
    assert result["termination_reason"] == "abandonment"
    assert result["turn_count"] == 6
    assert channel.selected == ["again"] * 6
    assert channel.sent == []
    assert len(simulator.observations) == 6
    signals = [event for event in events if event.type == "behavioral_signal_recorded"]
    assert [event.data.signal.signal for event in signals] == ["customer_turn_completed"] * 6
    assert not any(event.type == "repeated_information" for event in events)


def test_response_timeout_preserves_dispatch_count_without_completed_turn_signal() -> None:
    config = load_config(Path("examples/telecom/scenario.yaml"))
    definition = PersonaDefinition.model_validate(config.persona.model_dump()).model_copy(
        update={"behavioral_rules": []}
    )

    class SilentChannel(Channel):
        async def receive(self):
            await asyncio.Event().wait()

    channel, simulator, storage = (
        SilentChannel([]),
        Persona(["Question", "must not send"]),
        Storage(),
    )
    journal = RunJournal("run-1", storage, Redactor(PrivacyConfig()), Clock())
    graph = build_graph(
        scenario=Scenario.model_validate(config.scenario.model_dump()),
        persona=definition,
        channel=channel,
        simulator=simulator,
        journal=journal,
        wait=RealWaitController(),
        limits=config.execution.model_copy(
            update={
                "greeting_grace_seconds": 0,
                "response_timeout_seconds": 0.01,
            }
        ),
    )

    async def invoke():
        with pytest.raises(ChannelError):
            await graph.ainvoke(initial_state(definition))

    asyncio.run(invoke())
    assert channel.sent == ["Question"]
    assert len(simulator.observations) == 1
    boundaries = [
        event for event in storage.events if event.type == "behavior_observation_recorded"
    ]
    assert [event.data.kind for event in boundaries] == ["greeting"]
    assert not any(event.type == "behavioral_signal_recorded" for event in storage.events)
    dispatch = next(event for event in storage.events if event.type == "persona_state_changed")
    assert dispatch.data.after.turn_number == 1

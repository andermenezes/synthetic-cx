"""Deterministic runner boundaries: persistence, terminal policy, and cleanup."""

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from syntheticcx.config.loader import load_config
from syntheticcx.config.models import PrivacyConfig
from syntheticcx.domain.artifacts import ConfigArtifact, Run
from syntheticcx.domain.events import TypedTraceEvent
from syntheticcx.domain.models import (
    CompletedResponse,
    ConversationObservation,
    CustomerScenarioContext,
    Message,
    PersonaAction,
    PersonaDefinition,
    PersonaState,
    Scenario,
    SendReceipt,
)
from syntheticcx.privacy import Redactor
from syntheticcx.runtime.clock import ManualClock, ManualWaitController
from syntheticcx.runtime.journal import RunJournal
from syntheticcx.runtime.result import RuntimeResult
from syntheticcx.runtime.runner import run_conversation

NOW = datetime(2026, 9, 24, tzinfo=UTC)


def run() -> Run:
    return Run.model_validate(
        {
            "id": "run-1",
            "scenario_id": "billing",
            "persona_id": "standard",
            "started_at": NOW,
            "lifecycle_status": "in_progress",
            "metadata": {
                "configuration_version": "1",
                "framework_version": "0.1.0",
                "persona_model": {"provider": "fake", "model": "fake"},
                "evaluation_model": {"provider": "fake", "model": "fake"},
                "usage": {"unavailable_reason": "none"},
                "runtime_platform": "test",
                "owner": {
                    "owner_uuid": "owner",
                    "hostname": "test",
                    "pid": 1,
                    "process_start_evidence": "start",
                },
            },
        }
    )


class Storage:
    def __init__(
        self,
        *,
        fail_append: bool = False,
        fail_at_sequence: int | None = None,
        fail_create: bool = False,
    ) -> None:
        self.created = 0
        self.finalized = 0
        self.closed = 0
        self.events: list[TypedTraceEvent] = []
        self.fail_append = fail_append
        self.fail_at_sequence = fail_at_sequence
        self.fail_create = fail_create

    async def create_run(self, run: Run, config_artifact: ConfigArtifact) -> None:
        assert run.id == "run-1"
        self.created += 1
        if self.fail_create:
            raise OSError("sensitive-storage-path")

    async def append_event(self, run_id: str, event: TypedTraceEvent) -> None:
        if self.fail_append or event.sequence == self.fail_at_sequence:
            raise OSError("sensitive-storage-path")
        self.events.append(event)

    async def finalize_run(self, run: Run) -> None:
        self.finalized += 1

    async def close(self) -> None:
        self.closed += 1


def reply(content: str) -> CompletedResponse:
    return CompletedResponse(
        messages=[
            Message(
                id="assistant-1", actor="assistant", sequence=1, content=content, observed_at=NOW
            )
        ],
        started_at=NOW,
        completed_at=NOW,
        completion_strategy="completion_locator",
    )


class Channel:
    def __init__(
        self,
        *,
        failure: str | None = None,
        close_failure: bool = False,
        close_hangs: bool = False,
        echo: str | None = None,
    ) -> None:
        self.failure = failure
        self.close_failure = close_failure
        self.close_hangs = close_hangs
        self.echo = echo
        self.started = 0
        self.closed = 0
        self.sent: list[str] = []
        self.receiving = asyncio.Event()
        self.closing = asyncio.Event()

    async def start(self) -> None:
        self.started += 1
        if self.failure == "start":
            raise OSError("sensitive-channel-token")

    async def send(self, message: str) -> SendReceipt:
        if self.failure == "send":
            raise OSError("sensitive-channel-token")
        self.sent.append(message)
        return SendReceipt(
            message_id=f"sent-{len(self.sent)}",
            submitted_at=NOW,
            evidence={"echo": self.echo} if self.echo is not None else {},
        )

    async def receive(self) -> CompletedResponse:
        self.receiving.set()
        if self.failure == "receive":
            raise OSError("sensitive-channel-token")
        if self.failure == "hang":
            await asyncio.Event().wait()
        return reply("Your bill increased due to a plan change")

    async def close(self) -> None:
        self.closed += 1
        self.closing.set()
        if self.close_hangs:
            await asyncio.Event().wait()
        if self.close_failure:
            raise OSError("sensitive-channel-token")


class Persona:
    def __init__(
        self,
        *,
        failure: bool = False,
        complete: bool = True,
        goal_initial: bool = False,
    ) -> None:
        self.failure = failure
        self.complete = complete
        self.goal_initial = goal_initial
        self.calls = 0

    async def initialize(
        self, definition: PersonaDefinition, scenario_context: CustomerScenarioContext
    ) -> PersonaState:
        return definition.fresh_state()

    async def respond(
        self, observation: ConversationObservation, state: PersonaState
    ) -> PersonaAction:
        self.calls += 1
        if self.failure:
            raise OSError("sensitive-model-prompt")
        if self.goal_initial or (self.complete and state.turn_number):
            return PersonaAction(state=state, signal="goal_believed_complete")
        return PersonaAction(message="My account number is 123", state=state, signal="continue")


async def exercise(
    channel: Channel,
    persona: Persona,
    storage: Storage,
    *,
    clock: ManualClock | None = None,
    max_turns: int = 4,
    timeout: float = 20,
    greeting_grace: float = 0,
    definition: PersonaDefinition | None = None,
) -> RuntimeResult:
    config = load_config(Path("examples/telecom/scenario.yaml"))
    logical_clock = clock or ManualClock(NOW)
    journal = RunJournal("run-1", storage, Redactor(PrivacyConfig()), logical_clock)
    result = await run_conversation(
        run=run(),
        config_artifact=ConfigArtifact(source_content_hash="hash", effective_config={}),
        scenario=Scenario.model_validate(config.scenario.model_dump()),
        persona=definition or PersonaDefinition.model_validate(config.persona.model_dump()),
        channel=channel,
        simulator=persona,
        storage=storage,
        journal=journal,
        clock=logical_clock,
        wait=ManualWaitController(logical_clock),
        limits=config.execution.model_copy(
            update={
                "greeting_grace_seconds": greeting_grace,
                "max_turns": max_turns,
                "journey_timeout_seconds": timeout,
            }
        ),
    )
    assert storage.finalized == 0
    assert storage.closed == 1
    assert storage.events == list(result.committed_events)
    return result


def test_success_returns_detached_evidence_without_finalizing() -> None:
    async def scenario() -> None:
        storage = Storage()
        channel = Channel()
        result = await exercise(channel, Persona(), storage)
        assert result.termination_reason == "goal_believed_complete"
        assert result.turn_count == result.persona_state.turn_number == 1
        assert result.error is None and result.cleanup_completed
        assert channel.started == channel.closed == storage.created == 1
        assert [event.type for event in result.committed_events][-1] == "run_lifecycle_changed"
        assert result.committed_events[-1].data.current == "incomplete"
        assert result.committed_events[-1].data.outcome is None
        assert result.committed_events[2].data.message.id == "sent-1"
        assert result.committed_events[2].data.message.content == "My account number is 123"
        assert not hasattr(result, "outcome")

    asyncio.run(scenario())


def test_max_turns_respects_large_graph_recursion_budget() -> None:
    async def scenario() -> None:
        channel = Channel()
        persona = Persona(complete=False)
        result = await exercise(channel, persona, Storage(), max_turns=10)
        assert result.termination_reason == "max_turns"
        assert result.turn_count == 10
        assert len(channel.sent) == 10

    asyncio.run(scenario())


def test_channel_and_model_failures_are_sanitized_and_closed() -> None:
    async def scenario() -> None:
        for channel, persona, expected in [
            (Channel(failure="start"), Persona(), "channel_error"),
            (Channel(failure="send"), Persona(), "channel_error"),
            (Channel(), Persona(failure=True), "model_error"),
        ]:
            storage = Storage()
            result = await exercise(channel, persona, storage)
            assert result.termination_reason == expected
            assert result.turn_count == 0
            assert channel.closed == 1
            assert result.error is not None
            assert "sensitive" not in str(result.model_dump())
            assert [event.type for event in result.committed_events][-2:] == [
                "operation_failed",
                "run_lifecycle_changed",
            ]

    asyncio.run(scenario())


def test_persistence_error_and_secondary_close_error_do_not_hide_primary() -> None:
    async def scenario() -> None:
        storage = Storage(fail_append=True)
        channel = Channel(close_failure=True)
        result = await exercise(channel, Persona(), storage)
        assert result.termination_reason == "persistence_error"
        assert result.error is not None and result.error.category == "persistence"
        assert not result.cleanup_completed
        assert channel.closed == 1
        assert result.committed_events == ()

    asyncio.run(scenario())


def test_successful_receipt_remains_counted_when_customer_evidence_fails() -> None:
    async def scenario() -> None:
        storage = Storage(fail_at_sequence=2)
        channel = Channel()
        result = await exercise(channel, Persona(), storage)
        assert channel.sent == ["My account number is 123"]
        assert result.termination_reason == "persistence_error"
        assert result.turn_count == result.persona_state.turn_number == 1
        assert [event.type for event in result.committed_events] == [
            "run_lifecycle_changed",
            "conversation_started",
        ]
        assert result.error is not None and result.error.category == "persistence"

    asyncio.run(scenario())


def test_close_failure_is_secondary_to_successful_graph_termination() -> None:
    async def scenario() -> None:
        result = await exercise(Channel(close_failure=True), Persona(), Storage())
        assert result.termination_reason == "goal_believed_complete"
        assert not result.cleanup_completed
        assert result.error is not None and result.error.operation == "channel.close"
        assert any(event.type == "operation_failed" for event in result.committed_events)

    asyncio.run(scenario())


def test_close_deadline_is_bounded_and_attempted_once() -> None:
    async def scenario() -> None:
        clock = ManualClock(NOW)
        channel = Channel(close_hangs=True)
        task = asyncio.create_task(exercise(channel, Persona(), Storage(), clock=clock))
        await channel.closing.wait()
        await clock.advance(5)
        result = await task
        assert result.termination_reason == "goal_believed_complete"
        assert result.error is not None and result.error.operation == "channel.close"
        assert result.cleanup_completed is False
        assert channel.closed == 1

    asyncio.run(scenario())


def test_journey_timeout_and_cancellation_close_started_resources_once() -> None:
    async def scenario() -> None:
        clock = ManualClock(NOW)
        channel = Channel(failure="hang")
        task = asyncio.create_task(exercise(channel, Persona(), Storage(), clock=clock, timeout=2))
        await channel.receiving.wait()
        await clock.advance(2)
        timed_out = await task
        assert timed_out.termination_reason == "journey_timeout"
        assert timed_out.turn_count == 1
        assert channel.sent == ["My account number is 123"]
        assert channel.closed == 1

        other = Channel(failure="hang")
        task = asyncio.create_task(exercise(other, Persona(), Storage()))
        await other.receiving.wait()
        task.cancel()
        cancelled = await task
        assert cancelled.termination_reason == "cancelled"
        assert other.closed == 1
        assert cancelled.cleanup_completed

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("mode", "reason", "turns", "sends", "error_category"),
    [
        ("goal_initial", "goal_believed_complete", 0, 0, None),
        ("goal_after_response", "goal_believed_complete", 1, 1, None),
        ("max_turns", "max_turns", 1, 1, None),
        ("abandoned_initial", "abandonment", 0, 0, None),
        ("start_failure", "channel_error", 0, 0, "channel"),
        ("send_failure", "channel_error", 0, 0, "channel"),
        ("receive_failure", "channel_error", 1, 1, "channel"),
        ("model_failure", "model_error", 0, 0, "model"),
        ("journal_failure", "persistence_error", 1, 1, "persistence"),
    ],
)
def test_terminal_state_matrix(
    mode: str, reason: str, turns: int, sends: int, error_category: str | None
) -> None:
    async def scenario() -> None:
        config = load_config(Path("examples/telecom/scenario.yaml"))
        definition = PersonaDefinition.model_validate(config.persona.model_dump())
        if mode == "abandoned_initial":
            definition = definition.model_copy(
                update={
                    "initial_state": definition.initial_state.model_copy(update={"abandoned": True})
                }
            )
        channel = Channel(
            failure={
                "start_failure": "start",
                "send_failure": "send",
                "receive_failure": "receive",
            }.get(mode)
        )
        persona = Persona(
            failure=mode == "model_failure",
            complete=mode != "max_turns",
            goal_initial=mode == "goal_initial",
        )
        storage = Storage(fail_at_sequence=2 if mode == "journal_failure" else None)
        result = await exercise(
            channel,
            persona,
            storage,
            max_turns=2 if mode == "goal_after_response" else 1,
            definition=definition,
        )
        assert result.termination_reason == reason
        assert result.turn_count == result.persona_state.turn_number == turns
        assert len(channel.sent) == sends
        assert channel.closed == storage.closed == 1
        assert result.cleanup_completed
        if error_category is None:
            assert result.error is None
        else:
            assert result.error is not None
            assert result.error.category == error_category
        if mode == "abandoned_initial":
            assert persona.calls == 0
        if mode == "journal_failure":
            assert [event.type for event in result.committed_events] == [
                "run_lifecycle_changed",
                "conversation_started",
            ]
        else:
            assert result.committed_events[-1].type == "run_lifecycle_changed"
            assert result.committed_events[-1].data.termination_reason == reason
            assert result.committed_events[-1].data.outcome is None

    asyncio.run(scenario())


@pytest.mark.parametrize("echo", [None, "My account number is 123", "different echo"])
def test_canonical_customer_content_does_not_depend_on_echo(echo: str | None) -> None:
    async def scenario() -> None:
        result = await exercise(Channel(echo=echo), Persona(), Storage())
        customer = [
            event.data
            for event in result.committed_events
            if event.type == "message_observed" and event.data.message.actor == "customer"
        ]
        assert len(customer) == 1
        assert customer[0].message.id == "sent-1"
        assert customer[0].message.content == "My account number is 123"
        assert customer[0].identity_evidence == ({"echo": echo} if echo is not None else {})

    asyncio.run(scenario())


def test_initial_greeting_and_no_greeting_have_distinct_observations() -> None:
    class GreetingChannel(Channel):
        async def receive(self) -> CompletedResponse:
            return CompletedResponse(
                messages=[
                    Message(
                        id="greeting-1",
                        actor="assistant",
                        sequence=1,
                        content="Hello",
                        observed_at=NOW,
                    ),
                    Message(
                        id="greeting-2",
                        actor="assistant",
                        sequence=2,
                        content="How can I help?",
                        observed_at=NOW,
                    ),
                ],
                started_at=NOW,
                completed_at=NOW,
                completion_strategy="completion_locator",
            )

    async def scenario() -> None:
        for grace, expected in [(0, False), (2, True)]:
            result = await exercise(
                GreetingChannel(),
                Persona(goal_initial=True),
                Storage(),
                greeting_grace=grace,
            )
            events = result.committed_events
            types = [event.type for event in events]
            assert types == (
                [
                    "run_lifecycle_changed",
                    "message_observed",
                    "message_observed",
                    "conversation_started",
                    "run_lifecycle_changed",
                ]
                if expected
                else ["run_lifecycle_changed", "conversation_started", "run_lifecycle_changed"]
            )
            started = next(event.data for event in events if event.type == "conversation_started")
            assert started.greeting_observed is expected
            assert started.greeting_message_ids == (
                ["greeting-1", "greeting-2"] if expected else []
            )
            assert result.turn_count == 0

    asyncio.run(scenario())


def test_result_is_frozen_and_detached_from_storage() -> None:
    async def scenario() -> None:
        storage = Storage()
        result = await exercise(Channel(), Persona(), storage)
        with pytest.raises(ValidationError):
            result.turn_count = 999
        storage.events[2].data.message.metadata["external"] = "changed"
        assert "external" not in result.committed_events[2].data.message.metadata
        assert isinstance(result.committed_events, tuple)

    asyncio.run(scenario())


def test_creation_failure_skips_channel_and_closes_storage_once() -> None:
    async def scenario() -> None:
        channel = Channel()
        storage = Storage(fail_create=True, fail_append=True)
        result = await exercise(channel, Persona(), storage)
        assert result.termination_reason == "persistence_error"
        assert result.error is not None and result.error.category == "persistence"
        assert channel.started == channel.closed == 0
        assert storage.created == storage.closed == 1
        assert result.committed_events == ()

    asyncio.run(scenario())


def test_multipart_repetition_is_committed_once_before_goal_decision() -> None:
    class MultipartChannel(Channel):
        async def receive(self) -> CompletedResponse:
            return CompletedResponse(
                messages=[
                    Message(
                        id="assistant-1",
                        actor="assistant",
                        sequence=1,
                        content="Please provide the account ID",
                        observed_at=NOW,
                    ),
                    Message(
                        id="assistant-2",
                        actor="assistant",
                        sequence=2,
                        content="What is the account number?",
                        observed_at=NOW,
                    ),
                ],
                started_at=NOW,
                completed_at=NOW,
                completion_strategy="completion_locator",
            )

    class ObservingPersona(Persona):
        def __init__(self) -> None:
            super().__init__()
            self.observations: list[ConversationObservation] = []
            self.states: list[PersonaState] = []

        async def respond(
            self, observation: ConversationObservation, state: PersonaState
        ) -> PersonaAction:
            self.observations.append(observation)
            self.states.append(state)
            return await super().respond(observation, state)

    async def scenario() -> None:
        persona = ObservingPersona()
        result = await exercise(MultipartChannel(), persona, Storage())
        assert result.termination_reason == "goal_believed_complete"
        assert len(persona.observations) == 2
        assert [message.id for message in persona.observations[-1].messages] == [
            "assistant-1",
            "assistant-2",
        ]
        assert persona.states[-1].frustration == pytest.approx(0.3)
        events = result.committed_events
        assert [event.type for event in events] == [
            "run_lifecycle_changed",
            "conversation_started",
            "message_observed",
            "persona_state_changed",
            "message_observed",
            "message_observed",
            "repeated_information",
            "persona_state_changed",
            "run_lifecycle_changed",
        ]
        assert events[6].data.original_message_id == events[2].data.message.id
        assert events[6].data.request_message_id == events[4].data.message.id
        assert events[7].data.trigger_ids[0] == events[6].id
        assert [event.sequence for event in events] == list(range(len(events)))
        assert not any(event.type == "evaluation_completed" for event in events)

    asyncio.run(scenario())


def test_max_turns_never_calls_persona_after_final_assistant_response() -> None:
    class FailingAfterFirstCall(Persona):
        async def respond(
            self, observation: ConversationObservation, state: PersonaState
        ) -> PersonaAction:
            if self.calls:
                raise OSError("unexpected extra model invocation")
            return await super().respond(observation, state)

    async def scenario() -> None:
        persona = FailingAfterFirstCall(complete=False)
        result = await exercise(Channel(), persona, Storage(), max_turns=1)
        assert result.termination_reason == "max_turns"
        assert result.error is None
        assert result.turn_count == 1
        assert persona.calls == 1

    asyncio.run(scenario())


def test_journal_append_uses_persistence_deadline_not_journey_deadline() -> None:
    class BlockingStorage(Storage):
        def __init__(self) -> None:
            super().__init__()
            self.appending = asyncio.Event()

        async def append_event(self, run_id: str, event: TypedTraceEvent) -> None:
            if event.sequence == 1:
                self.appending.set()
                await asyncio.Event().wait()
            await super().append_event(run_id, event)

    async def scenario() -> None:
        clock = ManualClock(NOW)
        storage = BlockingStorage()
        channel = Channel()
        task = asyncio.create_task(exercise(channel, Persona(), storage, clock=clock, timeout=20))
        await storage.appending.wait()
        await clock.advance(5)
        result = await task
        assert result.termination_reason == "persistence_error"
        assert result.error is not None and result.error.category == "persistence"
        assert clock.now() == NOW + timedelta(seconds=5)
        assert result.turn_count == 0
        assert [event.type for event in result.committed_events] == ["run_lifecycle_changed"]
        assert channel.closed == storage.closed == 1

    asyncio.run(scenario())


def test_cancellation_during_channel_cleanup_returns_result_and_closes_storage() -> None:
    async def scenario() -> None:
        clock = ManualClock(NOW)
        channel = Channel(close_hangs=True)
        storage = Storage()
        task = asyncio.create_task(exercise(channel, Persona(), storage, clock=clock))
        await channel.closing.wait()
        task.cancel()
        await clock.advance(5)
        result = await task
        assert result.termination_reason == "cancelled"
        assert not result.cleanup_completed
        assert result.error is not None and result.error.category == "cancelled"
        assert result.committed_events[-1].data.termination_reason == "cancelled"
        assert channel.closed == storage.closed == 1

    asyncio.run(scenario())


def test_cancellation_during_storage_cleanup_preserves_committed_termination() -> None:
    class BlockingCloseStorage(Storage):
        def __init__(self) -> None:
            super().__init__()
            self.closing = asyncio.Event()

        async def close(self) -> None:
            self.closed += 1
            self.closing.set()
            await asyncio.Event().wait()

    async def scenario() -> None:
        clock = ManualClock(NOW)
        storage = BlockingCloseStorage()
        channel = Channel()
        # The helper asserts close once and committed events on return.
        task = asyncio.create_task(exercise(channel, Persona(), storage, clock=clock))
        await storage.closing.wait()
        task.cancel()
        await clock.advance(5)
        result = await task
        assert result.termination_reason == "goal_believed_complete"
        assert result.committed_events[-1].data.termination_reason == result.termination_reason
        assert result.error is not None and result.error.category == "cancelled"
        assert not result.cleanup_completed
        assert channel.closed == storage.closed == 1

    asyncio.run(scenario())

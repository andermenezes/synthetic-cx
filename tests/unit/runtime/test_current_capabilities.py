"""Live-control persona decisions and privacy-safe native diagnostics (no model calls)."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from syntheticcx.adapters.persona.tinytroupe import (
    TinyTroupePersonaSimulator,
    _customer_action,
)
from syntheticcx.config.models import ExecutionLimits, PrivacyConfig
from syntheticcx.domain.models import (
    CompletedResponse,
    ConversationCapabilities,
    ConversationObservation,
    Message,
    PersonaAction,
    PersonaDefinition,
    ReplyOption,
    Scenario,
    SelectionReceipt,
)
from syntheticcx.errors import ChannelError, ModelError, safe_channel_error
from syntheticcx.privacy import Redactor
from syntheticcx.runtime.clock import RealWaitController
from syntheticcx.runtime.graph import build_graph
from syntheticcx.runtime.journal import RunJournal
from syntheticcx.runtime.runner import _failure
from syntheticcx.runtime.state import initial_state

NOW = datetime(2026, 10, 5, tzinfo=UTC)
OLD = ReplyOption(id="old-id", label="Old choice")
CURRENT = ReplyOption(id="current-id", label="Billing")


def definition():
    document = yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())
    return PersonaDefinition.model_validate(document["persona"])


def observation(*, capabilities=None):
    return ConversationObservation(
        kind="assistant_response",
        messages=[
            Message(
                id="assistant",
                sequence=0,
                actor="assistant",
                content="Choose",
                observed_at=NOW,
                options=[OLD],
            )
        ],
        capabilities=capabilities,
    )


def decision(kind, content=""):
    return [{"action": {"type": kind, "content": content}}]


def test_legacy_text_and_latest_message_options_are_preserved():
    observed = observation()
    assert observed.capabilities is None
    assert observed.text_available
    assert observed.current_options == [OLD]
    state = definition().fresh_state()
    assert _customer_action(decision("TALK", "My goal"), observed, state).message == "My goal"
    assert _customer_action(decision("SELECT_OPTION", OLD.id), observed, state).selected_option_id


def test_current_menu_rejects_talk_and_historical_ids_but_allows_goal_signal():
    observed = observation(
        capabilities=ConversationCapabilities(
            text_available=False,
            options=[CURRENT],
        )
    )
    state = definition().fresh_state()
    with pytest.raises(ModelError, match="text action is unavailable"):
        _customer_action(decision("TALK", "My goal"), observed, state)
    with pytest.raises(ModelError, match="unavailable option"):
        _customer_action(decision("SELECT_OPTION", OLD.id), observed, state)
    assert (
        _customer_action(
            decision("SELECT_OPTION", CURRENT.id),
            observed,
            state,
        ).selected_option_id
        == CURRENT.id
    )
    assert _customer_action(decision("GOAL_COMPLETE"), observed, state).signal == (
        "goal_believed_complete"
    )


def test_capabilities_reject_dom_targets_and_duplicate_ids():
    with pytest.raises(ValidationError):
        ConversationCapabilities.model_validate(
            {
                "text_available": False,
                "options": [],
                "selector": "#private",
            }
        )
    with pytest.raises(ValidationError):
        ConversationCapabilities(options=[CURRENT, CURRENT])


def test_persona_stimulus_separates_live_controls_from_latest_message_evidence():
    captured = []

    class Session:
        def respond(self, stimulus):
            captured.append(json.loads(stimulus))
            return decision("SELECT_OPTION", CURRENT.id)

    adapter = object.__new__(TinyTroupePersonaSimulator)
    adapter._initialized = True
    adapter._session = Session()

    async def work(operation, label):
        return operation()

    adapter._work = work
    adapter._publish_interactions = lambda: None
    state = definition().fresh_state()
    result = asyncio.run(
        adapter.respond(
            observation(
                capabilities=ConversationCapabilities(
                    text_available=False,
                    options=[CURRENT],
                )
            ),
            state,
        )
    )
    assert result.selected_option_id == CURRENT.id
    assert captured[0]["text_available"] is False
    assert captured[0]["current_options"] == [CURRENT.model_dump()]
    assert captured[0]["reply_options"][0]["options"] == [OLD.model_dump()]
    assert "success_criteria" not in json.dumps(captured)


class Clock:
    def now(self):
        return NOW


class Storage:
    async def append_event(self, run_id, event):
        pass


class Channel:
    reads = 0
    selected = None

    async def observe_initial(self):
        return CompletedResponse(
            messages=observation().messages,
            started_at=NOW,
            completed_at=NOW,
            completion_strategy="quiet_interval",
        )

    async def observe_controls(self):
        self.reads += 1
        return {"text_available": False, "options": [CURRENT] if self.reads == 1 else []}

    async def select_option(self, option_id):
        self.selected = option_id
        return SelectionReceipt(
            message_id="customer",
            submitted_at=NOW,
            selected_option_id=option_id,
            selected_label=CURRENT.label,
            source_message_id="assistant",
        )

    async def send(self, text):
        pytest.fail("menu-only channel must never receive TALK")

    async def receive(self):
        response = await self.observe_initial()
        return response.model_copy(
            update={
                "messages": [
                    message.model_copy(update={"id": "assistant-response"})
                    for message in response.messages
                ]
            }
        )


def invoke(channel, simulator, wait=None):
    document = yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())
    persona = definition()
    graph = build_graph(
        scenario=Scenario.model_validate(document["scenario"]),
        persona=persona,
        channel=channel,
        simulator=simulator,
        journal=RunJournal("run", Storage(), Redactor(PrivacyConfig()), Clock()),
        wait=wait or RealWaitController(),
        limits=ExecutionLimits.model_validate(document["execution"]).model_copy(
            update={"greeting_grace_seconds": 0}
        ),
    )
    return asyncio.run(graph.ainvoke(initial_state(persona)))


class Customer:
    async def initialize(self, persona, context):
        assert context.goal
        assert not hasattr(context, "success_criteria")
        return persona.fresh_state()

    async def respond(self, observed, state):
        assert not observed.text_available
        assert observed.messages[-1].options == [OLD]
        if observed.current_options:
            return PersonaAction(state=state, signal="continue", selected_option_id=CURRENT.id)
        return PersonaAction(state=state, signal="goal_believed_complete")


def test_graph_refreshes_before_every_decision_and_keeps_message_evidence():
    channel = Channel()
    timeouts = []

    class Wait(RealWaitController):
        async def run(self, operation, timeout):
            if operation.cr_code.co_name == "observe_controls":
                timeouts.append(timeout)
            return await super().run(operation, timeout)

    result = invoke(channel, Customer(), Wait())
    assert channel.reads == 2
    assert channel.selected == CURRENT.id
    assert result["turn_count"] == 1
    assert len(timeouts) == 2
    assert all(value == 10 for value in timeouts)


@pytest.mark.parametrize("kind,content", [("TALK", "My goal"), ("SELECT_OPTION", OLD.id)])
def test_graph_independently_rejects_invalid_persona_actions(kind, content):
    class Invalid(Customer):
        async def respond(self, observed, state):
            return PersonaAction(
                state=state,
                signal="continue",
                message=content if kind == "TALK" else None,
                selected_option_id=content if kind == "SELECT_OPTION" else None,
            )

    channel = Channel()
    with pytest.raises(ModelError):
        invoke(channel, Invalid())
    assert channel.selected is None


@pytest.mark.parametrize("operation", ["channel.state", "channel.reader", "channel.ownership"])
def test_native_failure_reasons_survive_graph_without_exception_text(operation):
    class Failed(Channel):
        async def receive(self):
            raise ChannelError(operation, "SECRET DOM #account token=private", retryable=True)

    with pytest.raises(ChannelError) as caught:
        invoke(Failed(), Customer())
    reason, detail = _failure(caught.value, "graph")
    assert reason == "channel_error"
    assert detail.operation == operation
    assert detail.reason_code == operation
    assert detail.retryable
    assert "SECRET" not in detail.model_dump_json()
    assert "private" not in str(caught.value)


def test_unallowlisted_exceptions_never_become_persisted_reasons_or_messages():
    for exc in [RuntimeError("SECRET"), ChannelError("channel.SECRET", "SECRET")]:
        wrapped = safe_channel_error(exc, "channel.receive", "assistant receive failed")
        _, detail = _failure(wrapped, "graph")
        assert detail.operation == "channel.receive"
        assert detail.reason_code is None
        assert "SECRET" not in detail.model_dump_json()


def test_controls_timeout_stops_before_persona_decision():
    class Blocked(Channel):
        cancelled = False

        async def observe_controls(self):
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled = True

    class NeverRespond(Customer):
        async def respond(self, observed, state):
            pytest.fail("failed control observation must not reach persona")

    class FastWait(RealWaitController):
        async def run(self, operation, timeout):
            if operation.cr_code.co_name == "observe_controls":
                timeout = 0.01
            return await super().run(operation, timeout)

    channel = Blocked()
    with pytest.raises(ChannelError) as caught:
        invoke(channel, NeverRespond(), FastWait())
    assert channel.cancelled
    _, detail = _failure(caught.value, "graph")
    assert detail.reason_code == "channel.controls"
    assert channel.selected is None

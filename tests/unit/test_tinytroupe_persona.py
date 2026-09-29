"""Run-local TinyPerson execution with a deterministic private model client."""

import asyncio
import json
import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from syntheticcx._compat import tinytroupe as compatibility
from syntheticcx._compat.blocking import BoundedResult
from syntheticcx.adapters.persona import tinytroupe as adapter_module
from syntheticcx.adapters.persona.tinytroupe import TinyTroupePersonaSimulator, _customer_action
from syntheticcx.config.loader import load_config
from syntheticcx.config.settings import prepare_settings
from syntheticcx.domain.models import (
    ConversationObservation,
    CustomerScenarioContext,
    Message,
    ModelIdentity,
    ModelInteraction,
    PersonaDefinition,
    PersonaState,
    Scenario,
    TokenUsage,
)
from syntheticcx.domain.protocols import PersonaSimulator
from syntheticcx.errors import ModelError


class FakeClient:
    def __init__(self, *args: object) -> None:
        self.actions: list[list[dict[str, str]]] = []
        self.requests: list[str] = []
        self.interactions: list[ModelInteraction] = []
        self.closed = False
        self.memory_settings: list[tuple[bool, bool]] = []

    def send_message(self, current_messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, str]:
        assert "response_format" in kwargs
        configuration = compatibility.import_tinytroupe().config_manager
        self.memory_settings.append(
            (
                configuration.get("enable_memory_consolidation"),
                configuration.get("enable_continuous_contextual_semantic_memory_retrieval"),
            )
        )
        self.requests.append(json.dumps(current_messages))
        actions = self.actions.pop(0)
        return {
            "role": "assistant",
            "content": json.dumps(
                {
                    "actions": actions,
                    "cognitive_state": {
                        "goals": "Understand invoice",
                        "context": [],
                        "attention": "billing",
                        "emotions": "calm",
                    },
                }
            ),
        }

    def close(self) -> None:
        self.closed = True


def action(kind: str, content: str = "") -> dict[str, str]:
    return {"type": kind, "content": content, "target": "Support"}


def definitions() -> tuple[PersonaDefinition, CustomerScenarioContext]:
    config = load_config(Path("examples/telecom/scenario.yaml"))
    scenario = Scenario.model_validate(config.scenario.model_dump())
    scenario.success_criteria[0].parameters["private"] = "HIDDEN_MARKER"
    return (
        PersonaDefinition.model_validate(config.persona.model_dump()),
        CustomerScenarioContext.from_scenario(scenario),
    )


def simulator(monkeypatch: pytest.MonkeyPatch) -> tuple[TinyTroupePersonaSimulator, FakeClient]:
    client = FakeClient()
    monkeypatch.setattr(adapter_module, "TinyModelClient", lambda *args: client)
    config = load_config(Path("examples/telecom/scenario.yaml"))
    settings = prepare_settings(config, {"OPENAI_API_KEY": "selected-secret"})
    return (
        TinyTroupePersonaSimulator(config.model, settings.persona, timeout_seconds=1),
        client,
    )


def observation(text: str = "Here is your billing explanation.") -> ConversationObservation:
    return ConversationObservation(
        kind="assistant_response",
        messages=[
            Message(
                id="assistant-1",
                sequence=1,
                actor="assistant",
                content=text,
                observed_at=datetime.now(UTC),
            )
        ],
    )


def test_real_tinyperson_retains_state_and_restores_globals(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    adapter, client = simulator(monkeypatch)
    contract: PersonaSimulator = adapter
    assert contract is adapter
    person_class = __import__("tinytroupe.agent", fromlist=["TinyPerson"]).TinyPerson
    clients = __import__("tinytroupe.clients", fromlist=["client"])
    original_agents = dict(person_class.all_agents)
    original_registry = dict(clients._api_type_to_client)
    original_override = clients._api_type_override
    original_config = dict(compatibility.import_tinytroupe().config_manager._config)
    original_logger = logging.getLogger("tinytroupe").disabled
    definition, context = definitions()
    client.actions = [
        [action("TALK", "My invoice went up. Why?"), action("DONE")],
        [action("GOAL_COMPLETE"), action("DONE")],
    ]

    async def exercise() -> None:
        state = await adapter.initialize(definition, context)
        first = await adapter.respond(ConversationObservation(kind="conversation_started"), state)
        assert first.message == "My invoice went up. Why?"
        second = await adapter.respond(observation(), state)
        assert second.signal == "goal_believed_complete"
        assert second.message is None
        await adapter.close()
        await adapter.close()

    asyncio.run(exercise())
    assert client.closed
    assert len(client.requests) == 2
    assert client.memory_settings == [(False, False), (False, False)]
    assert all("HIDDEN_MARKER" not in text for text in client.requests)
    assert "Understand why the current invoice is higher" in client.requests[0]
    assert "SYNTHETIC-001" in client.requests[0]
    assert "My invoice went up" in client.requests[1]
    assert "Here is your billing explanation" in client.requests[1]
    assert person_class.all_agents == original_agents
    assert clients._api_type_to_client == original_registry
    assert clients._api_type_override == original_override
    assert compatibility.import_tinytroupe().config_manager._config == original_config
    assert logging.getLogger("tinytroupe").disabled == original_logger
    assert "My invoice went up. Why?" not in caplog.text
    assert "HIDDEN_MARKER" not in caplog.text


def test_persona_is_instructed_to_speak_selected_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    goals: list[str] = []

    class Session:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def initialize(self, _description: str, goal: str) -> None:
            goals.append(goal)

        def close(self) -> None:
            pass

    monkeypatch.setattr(adapter_module, "TinySession", Session)
    adapter, _client = simulator(monkeypatch)
    definition, context = definitions()
    definition = definition.model_copy(
        update={"communication": definition.communication.model_copy(update={"language": "pt-BR"})}
    )

    async def exercise() -> None:
        await adapter.initialize(definition, context)
        await adapter.close()

    asyncio.run(exercise())
    assert "Brazilian Portuguese for every customer TALK message" in goals[0]


@pytest.mark.parametrize(
    "actions",
    [
        [action("DONE")],
        [action("TALK", "one"), action("TALK", "two")],
        [action("TALK", "one"), action("GOAL_COMPLETE")],
        [action("TALK", " ")],
        [action("UNKNOWN")],
    ],
)
def test_ambiguous_or_invalid_action_never_sends(actions: list[dict[str, str]]) -> None:
    with pytest.raises(ModelError):
        _customer_action([{"action": item} for item in actions], observation(), PersonaState())


def test_explicit_abandonment_has_owned_transition() -> None:
    state = PersonaState(frustration=0.5)
    result = _customer_action([{"action": action("ABANDON")}], observation(), state)
    assert result.signal == "abandon"
    assert result.state.abandoned
    assert result.transitions[0].trigger_ids == ["assistant-1"]
    assert result.transitions[0].before == state


def test_timeout_poisons_adapter_and_discards_late_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, client = simulator(monkeypatch)
    adapter._timeout = 0.02
    definition, context = definitions()
    release = threading.Event()
    entered = threading.Event()

    def slow_initialize(description: str, goal: str) -> None:
        entered.set()
        release.wait(1)
        client.interactions.append(
            ModelInteraction(
                id="late-call",
                identity=ModelIdentity(provider="fake", model="fake"),
                started_at=datetime.now(UTC),
                ended_at=datetime.now(UTC),
                usage=TokenUsage(unavailable_reason="not reported"),
            )
        )

    monkeypatch.setattr(adapter._session, "initialize", slow_initialize)

    async def exercise() -> None:
        try:
            with pytest.raises(ModelError, match="timed out"):
                await adapter.initialize(definition, context)
            assert entered.is_set()
            with pytest.raises(ModelError, match="unavailable"):
                await adapter.initialize(definition, context)
        finally:
            release.set()
            await asyncio.sleep(0.05)

    asyncio.run(exercise())
    assert client.closed
    assert adapter.take_interactions() == ()


def test_only_successful_persona_operations_publish_interactions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, client = simulator(monkeypatch)
    definition, context = definitions()
    client.actions = [[action("TALK", "Hello"), action("DONE")]]

    async def exercise() -> None:
        await adapter.initialize(definition, context)
        assert adapter.take_interactions() == ()
        # Simulate the metadata captured by the synchronous SDK client.
        interaction = ModelInteraction(
            id="completed-call",
            identity=ModelIdentity(provider="fake", model="fake"),
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC),
            usage=TokenUsage(unavailable_reason="not reported"),
        )
        client.interactions.append(interaction)
        await adapter.respond(
            ConversationObservation(kind="conversation_started"), definition.fresh_state()
        )
        assert adapter.take_interactions() == (interaction,)
        assert adapter.take_interactions() == ()
        await adapter.close()

    asyncio.run(exercise())


def test_timeout_after_worker_completion_still_closes_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover the deadline-edge race where the worker has just released its scope."""
    adapter, client = simulator(monkeypatch)
    definition, context = definitions()

    async def reported_late_result(
        operation: Callable[[], object], *, timeout_seconds: float
    ) -> BoundedResult[object]:
        operation()
        return BoundedResult(value=None, timed_out=True, cancelled=False)

    async def exercise() -> None:
        await adapter.initialize(definition, context)
        monkeypatch.setattr(adapter._session, "respond", lambda stimulus: [])
        monkeypatch.setattr(adapter_module, "run_blocking_bounded", reported_late_result)
        with pytest.raises(ModelError, match="timed out"):
            await adapter.respond(
                ConversationObservation(kind="conversation_started"), definition.fresh_state()
            )
        for _ in range(20):
            if client.closed:
                break
            await asyncio.sleep(0.01)

    asyncio.run(exercise())
    assert client.closed


def test_failed_model_call_restores_global_state_and_closes_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, client = simulator(monkeypatch)
    definition, context = definitions()
    tinytroupe_module = compatibility.import_tinytroupe()
    original_config = dict(tinytroupe_module.config_manager._config)
    clients = __import__("tinytroupe.clients", fromlist=["client"])
    original_registry = dict(clients._api_type_to_client)
    original_override = clients._api_type_override

    def fail(*args: object, **kwargs: object) -> dict[str, str]:
        raise RuntimeError("secret endpoint response")

    monkeypatch.setattr(client, "send_message", fail)

    async def exercise() -> None:
        await adapter.initialize(definition, context)
        with pytest.raises(ModelError, match="persona operation failed") as exc:
            await adapter.respond(
                ConversationObservation(kind="conversation_started"), definition.fresh_state()
            )
        assert "secret endpoint response" not in str(exc.value)

    asyncio.run(exercise())
    assert client.closed
    assert tinytroupe_module.config_manager._config == original_config
    assert clients._api_type_to_client == original_registry
    assert clients._api_type_override == original_override


def test_cancellation_poisons_adapter_until_worker_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, client = simulator(monkeypatch)
    definition, context = definitions()
    release = threading.Event()
    entered = threading.Event()

    def slow_initialize(description: str, goal: str) -> None:
        entered.set()
        release.wait(1)

    monkeypatch.setattr(adapter._session, "initialize", slow_initialize)

    async def exercise() -> None:
        task = asyncio.create_task(adapter.initialize(definition, context))
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            with pytest.raises(ModelError, match="unavailable"):
                await adapter.initialize(definition, context)
        finally:
            release.set()
            await asyncio.sleep(0.05)

    asyncio.run(exercise())
    assert client.closed


def test_sequential_personas_keep_separate_customer_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[FakeClient] = []

    def make_client(*args: object) -> FakeClient:
        client = FakeClient()
        client.actions = [[action("TALK", f"Turn from persona {len(clients)}"), action("DONE")]]
        clients.append(client)
        return client

    monkeypatch.setattr(adapter_module, "TinyModelClient", make_client)
    config = load_config(Path("examples/telecom/scenario.yaml"))
    settings = prepare_settings(config, {"OPENAI_API_KEY": "selected-secret"})
    definition, context = definitions()

    async def exercise() -> None:
        for index in range(2):
            adapter = TinyTroupePersonaSimulator(config.model, settings.persona, timeout_seconds=1)
            state = await adapter.initialize(
                definition, context.model_copy(update={"goal": f"Private goal {index}"})
            )
            action_result = await adapter.respond(
                ConversationObservation(kind="conversation_started"), state
            )
            assert action_result.message == f"Turn from persona {index}"
            await adapter.close()

    asyncio.run(exercise())
    assert all(client.closed for client in clients)
    assert "Private goal 0" in clients[0].requests[0]
    assert "Private goal 1" not in clients[0].requests[0]
    assert "Private goal 1" in clients[1].requests[0]
    assert "Private goal 0" not in clients[1].requests[0]


def test_late_worker_keeps_global_scope_until_it_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, first_client = simulator(monkeypatch)
    second, _ = simulator(monkeypatch)
    definition, context = definitions()
    first_client.actions = [[action("TALK", "Late result"), action("DONE")]]
    entered = threading.Event()
    release = threading.Event()
    original_send = first_client.send_message

    def blocked_send(messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, str]:
        entered.set()
        release.wait(1)
        return original_send(messages, **kwargs)

    monkeypatch.setattr(first_client, "send_message", blocked_send)
    tinytroupe_module = compatibility.import_tinytroupe()
    original_model = tinytroupe_module.config_manager.get("model")

    async def exercise() -> None:
        await first.initialize(definition, context)
        first._timeout = 0.02
        second._timeout = 0.02
        task = asyncio.create_task(
            first.respond(
                ConversationObservation(kind="conversation_started"), definition.fresh_state()
            )
        )
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            with pytest.raises(ModelError, match="timed out"):
                await task
            with pytest.raises(ModelError, match="timed out"):
                await second.initialize(definition, context)
        finally:
            release.set()
            await asyncio.sleep(0.15)

    asyncio.run(exercise())
    assert first_client.closed
    assert tinytroupe_module.config_manager.get("model") == original_model
    assert first.take_interactions() == ()

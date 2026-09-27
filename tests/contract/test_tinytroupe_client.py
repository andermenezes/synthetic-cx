"""Pinned TinyTroupe client surface against an in-memory OpenAI SDK transport."""

import asyncio
import json
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import httpx
from openai import OpenAI
from pydantic import BaseModel

from syntheticcx.adapters.persona import _client
from syntheticcx.adapters.persona.tinytroupe import TinyTroupePersonaSimulator
from syntheticcx.config.loader import load_config
from syntheticcx.config.models import (
    OpenAICompatibleModelConfig,
    OpenAIModelConfig,
    ResolvedModelSettings,
)
from syntheticcx.domain.models import (
    ConversationObservation,
    CustomerScenarioContext,
    PersonaDefinition,
)
from syntheticcx.privacy import Redactor
from syntheticcx.privacy.prompt_capture import PromptCapture


def install_mock_sdk_transport(
    monkeypatch: Any, respond: Callable[[httpx.Request], httpx.Response]
) -> None:
    """Route the real synchronous SDK through a deterministic local transport."""
    original = _client.OpenAI  # type: ignore[attr-defined]

    def stub_sdk(**kwargs: Any) -> OpenAI:
        kwargs["http_client"] = httpx.Client(
            transport=httpx.MockTransport(respond),
            event_hooks=kwargs["http_client"].event_hooks,
        )
        return original(**kwargs)

    monkeypatch.setattr(_client, "OpenAI", stub_sdk)


def test_tinytroupe_enabled_prompt_capture_redacts_before_evidence(monkeypatch: Any) -> None:
    class Action(BaseModel):  # type: ignore[misc]
        ok: bool

    install_mock_sdk_transport(
        monkeypatch,
        lambda _: httpx.Response(
            200,
            json={
                "id": "response-1",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-5.4-mini",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": '{"ok":true}'}}
                ],
            },
        ),
    )
    project = load_config(Path("examples/telecom/scenario.yaml"))
    capture = PromptCapture(
        Redactor(project.privacy, resolved_secrets=["selected-secret"]), project.scenario
    )
    settings = ResolvedModelSettings(
        provider="openai", model="gpt-5.4-mini", credential="selected-secret"
    )
    client = _client.TinyModelClient(project.model, settings, 5, prompt_capture=capture)
    try:
        client.send_message(
            [{"role": "user", "content": "Account SYNTHETIC-001 selected-secret"}],
            response_format=Action,
        )
        interaction = client.interactions[0]
        assert interaction.captured_prompt is not None
        assert "SYNTHETIC-001" not in interaction.captured_prompt
        assert "selected-secret" not in interaction.captured_prompt
        assert "<redacted:account_number>" in interaction.captured_prompt
        assert "<redacted:secret>" in interaction.captured_prompt
        assert "prompts_not_captured" not in interaction.capture_omissions
    finally:
        client.close()


def test_custom_client_routes_per_run_without_ambient_credentials(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-secret")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "Authorization: Bearer ambient-secret")
    requests: list[httpx.Request] = []
    reported_models = ["local-test-model", ""]

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        content = json.dumps(
            {
                "actions": [
                    {"type": "TALK", "content": "My bill changed", "target": "Support"},
                    {"type": "DONE", "content": "", "target": ""},
                ],
                "cognitive_state": {
                    "goals": "Understand invoice",
                    "context": [],
                    "attention": "billing",
                    "emotions": "calm",
                },
            }
        )
        return httpx.Response(
            200,
            json={
                "id": "response-1",
                "object": "chat.completion",
                "created": 1,
                "model": reported_models[len(requests) - 1],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            },
        )

    install_mock_sdk_transport(monkeypatch, respond)
    action_model = import_module("tinytroupe.agent").CognitiveActionsModel

    for index in range(2):
        secret = f"selected-{index}"
        config = cast(
            OpenAICompatibleModelConfig,
            OpenAICompatibleModelConfig.model_validate(
                {
                    "provider": "openai_compatible",
                    "model": "local-test-model",
                    "base_url": f"http://127.0.0.1:{8000 + index}/v1",
                    "credential_environment_variable": f"LOCAL_{index}",
                    "capabilities": {
                        "chat_completions": True,
                        "structured_output": True,
                        "embeddings": False,
                        "token_usage": True,
                        "model_version": True,
                    },
                }
            ),
        )
        settings = ResolvedModelSettings(
            provider="openai_compatible", model="local-test-model", credential=secret
        )
        client = _client.TinyModelClient(config, settings, 1)
        try:
            response = client.send_message(
                [{"role": "user", "content": "Explain the bill"}],
                response_format=action_model,
            )
            assert response["role"] == "assistant"
            assert json.loads(response["content"])["actions"][0]["type"] == "TALK"
            assert client.interactions[0].usage.input_tokens == 3
            assert client.interactions[0].identity.model_version == (reported_models[index] or None)
            if index:
                assert (
                    "provider_model_version_unavailable" in client.interactions[0].capture_omissions
                )
            assert secret not in client.interactions[0].model_dump_json()
        finally:
            client.close()
    assert [str(request.url) for request in requests] == [
        "http://127.0.0.1:8000/v1/chat/completions",
        "http://127.0.0.1:8001/v1/chat/completions",
    ]
    for index, request in enumerate(requests):
        assert request.headers["authorization"] == f"Bearer selected-{index}"
        assert json.loads(request.content)["model"] == "local-test-model"


def test_pinned_persona_uses_selected_sdk_transport_and_returns_owned_evidence(
    monkeypatch: Any,
) -> None:
    """Exercise real TinyPerson + structured SDK parse with an offline HTTP transport."""
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-secret")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "Authorization: Bearer ambient-secret")
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "response-1",
                "object": "chat.completion",
                "created": 1,
                "model": "reported-model-version",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "actions": [
                                        {
                                            "type": "TALK",
                                            "content": "Please explain my bill",
                                            "target": "Support",
                                        },
                                        {"type": "DONE", "content": "", "target": ""},
                                    ],
                                    "cognitive_state": {
                                        "goals": "Understand the invoice",
                                        "context": [],
                                        "attention": "billing",
                                        "emotions": "calm",
                                    },
                                }
                            ),
                        },
                    }
                ],
                "usage": None,
            },
        )

    install_mock_sdk_transport(monkeypatch, respond)
    project = load_config(Path("examples/telecom/scenario.yaml"))
    definition = PersonaDefinition.model_validate(project.persona.model_dump())
    context = CustomerScenarioContext.from_scenario(project.scenario)
    configurations: list[tuple[OpenAIModelConfig | OpenAICompatibleModelConfig, str, str]] = [
        (
            OpenAIModelConfig(provider="openai", model="gpt-5.4-mini"),
            "selected-openai",
            "api.openai.com",
        ),
        (
            OpenAICompatibleModelConfig.model_validate(
                {
                    "provider": "openai_compatible",
                    "model": "local-test-model",
                    "base_url": "http://127.0.0.1:8088/v1",
                    "credential_environment_variable": "LOCAL_TOKEN",
                    "capabilities": {
                        "chat_completions": True,
                        "structured_output": True,
                        "embeddings": False,
                        "token_usage": False,
                        "model_version": True,
                    },
                }
            ),
            "selected-local",
            "127.0.0.1",
        ),
    ]

    async def exercise() -> None:
        for config, secret, _ in configurations:
            settings = ResolvedModelSettings(
                provider=config.provider, model=config.model, credential=secret
            )
            # TinyTroupe imports and configures several global clients during construction.
            # This contract is offline but can contend with other suite workers, so do not
            # use a timing-sensitive budget intended for timeout behavior tests.
            simulator = TinyTroupePersonaSimulator(config, settings, timeout_seconds=15)
            try:
                state = await simulator.initialize(definition, context)
                action = await simulator.respond(
                    ConversationObservation(kind="conversation_started"), state
                )
                assert action.message == "Please explain my bill"
                interactions = simulator.take_interactions()
                assert len(interactions) == 1
                assert interactions[0].identity.model_version == "reported-model-version"
                assert interactions[0].usage.unavailable_reason is not None
                assert "prompts_not_captured" in interactions[0].capture_omissions
                assert "model_output_not_captured" in interactions[0].capture_omissions
                assert "Please explain my bill" not in interactions[0].model_dump_json()
                assert "ambient-secret" not in interactions[0].model_dump_json()
                assert all(secret not in item.model_dump_json() for item in interactions)
            finally:
                await simulator.close()

    asyncio.run(exercise())
    assert len(requests) == len(configurations)
    for request, (config, secret, host) in zip(requests, configurations):
        assert request.url.host == host
        assert request.headers["authorization"] == f"Bearer {secret}"
        body = json.loads(request.content)
        assert body["model"] == config.model
        assert body["response_format"]["type"] == "json_schema"
        assert ("temperature" in body) == (config.provider == "openai_compatible")

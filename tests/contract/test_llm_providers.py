"""Offline contracts for the two explicitly configured model adapters."""

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from openai import AsyncOpenAI

from syntheticcx.adapters.llm import OpenAICompatibleProvider, OpenAIProvider, _chat
from syntheticcx.config.loader import load_config
from syntheticcx.config.models import (
    OpenAICompatibleModelConfig,
    OpenAIModelConfig,
    ResolvedModelSettings,
)
from syntheticcx.domain.models import LLMMessage, LLMRequest, LLMResponse
from syntheticcx.domain.protocols import LLMProvider
from syntheticcx.errors import ModelError
from syntheticcx.privacy import Redactor
from syntheticcx.privacy.prompt_capture import PromptCapture

_CAPABILITIES = {
    "chat_completions": True,
    "structured_output": True,
    "embeddings": False,
    "token_usage": False,
    "model_version": False,
}


def config_for(kind: str) -> OpenAIModelConfig | OpenAICompatibleModelConfig:
    if kind == "openai":
        return OpenAIModelConfig(provider="openai", model="gpt-5.4-mini")
    return cast(
        OpenAICompatibleModelConfig,
        OpenAICompatibleModelConfig.model_validate(
            {
                "provider": "openai_compatible",
                "model": "custom-model/1",
                "base_url": "http://127.0.0.1:12000/v1",
                "credential_environment_variable": "LOCAL_MODEL_TOKEN",
                "capabilities": _CAPABILITIES,
            }
        ),
    )


def request(*, structured: bool = False) -> LLMRequest:
    return LLMRequest(
        id="request-1",
        messages=[LLMMessage(role="user", content="hello")],
        response_schema={"type": "object", "properties": {"answer": {"type": "string"}}}
        if structured
        else None,
    )


def payload(content: str | None = "Hello", **extra: Any) -> dict[str, Any]:
    return {
        "id": "completion-1",
        "object": "chat.completion",
        "created": 1,
        "model": "reported-version",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        **extra,
    }


def fake_client(handler: httpx.MockTransport) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key="selected-secret",
        base_url="http://127.0.0.1:12000/v1",
        http_client=httpx.AsyncClient(transport=handler),
        max_retries=0,
    )


def provider_for(
    kind: str,
    client: AsyncOpenAI | None = None,
    *,
    timeout_seconds: float = 1,
    prompt_capture: PromptCapture | None = None,
) -> OpenAIProvider | OpenAICompatibleProvider:
    config = config_for(kind)
    settings = ResolvedModelSettings(
        provider=config.provider, model=config.model, credential="selected-secret"
    )
    if isinstance(config, OpenAIModelConfig):
        return OpenAIProvider(
            config,
            settings,
            timeout_seconds=timeout_seconds,
            client=client,
            prompt_capture=prompt_capture,
        )
    return OpenAICompatibleProvider(
        config,
        settings,
        timeout_seconds=timeout_seconds,
        client=client,
        prompt_capture=prompt_capture,
    )


@pytest.mark.parametrize("kind", ["openai", "openai_compatible"])
def test_provider_contract_translates_request_and_usage(kind: str) -> None:
    observed: list[dict[str, Any]] = []

    def respond(http_request: httpx.Request) -> httpx.Response:
        assert http_request.url.path == "/v1/chat/completions"
        assert http_request.headers["authorization"] == "Bearer selected-secret"
        observed.append(json.loads(http_request.content))
        return httpx.Response(200, json=payload())

    async def exercise() -> LLMResponse:
        client = fake_client(httpx.MockTransport(respond))
        adapter = provider_for(kind, client)
        try:
            contract: LLMProvider = adapter
            return await contract.generate(request())
        finally:
            await adapter.close()
            await client.close()

    result = asyncio.run(exercise())
    assert isinstance(result, LLMResponse)
    assert result.content == "Hello"
    assert result.interaction.identity.provider == kind
    assert result.interaction.identity.model == config_for(kind).model
    assert result.interaction.identity.model_version == "reported-version"
    assert result.interaction.usage.input_tokens == 3
    assert result.interaction.usage.output_tokens == 4
    assert result.interaction.input_event_ids == []
    assert "prompts_not_captured" in result.interaction.capture_omissions
    assert "selected-secret" not in result.model_dump_json()
    assert observed[0]["model"] == config_for(kind).model
    assert observed[0]["messages"] == [{"role": "user", "content": "hello"}]
    if kind == "openai":
        assert "temperature" not in observed[0]
        assert "temperature_not_applied" in result.interaction.capture_omissions
        assert "effective_temperature" not in result.interaction.identity.sampling_parameters
    else:
        assert observed[0]["temperature"] == 0


@pytest.mark.parametrize("kind", ["openai", "openai_compatible"])
def test_enabled_prompt_capture_redacts_context_and_selected_secret(kind: str) -> None:
    project = load_config(Path("examples/telecom/scenario.yaml"))
    capture = PromptCapture(
        Redactor(project.privacy, resolved_secrets=["selected-secret"]), project.scenario
    )

    async def exercise() -> LLMResponse:
        client = fake_client(httpx.MockTransport(lambda _: httpx.Response(200, json=payload())))
        adapter = provider_for(kind, client, prompt_capture=capture)
        try:
            return await adapter.generate(
                LLMRequest(
                    id="capture-1",
                    messages=[
                        LLMMessage(
                            role="user", content="Account SYNTHETIC-001, selected-secret: help"
                        )
                    ],
                )
            )
        finally:
            await adapter.close()
            await client.close()

    interaction = asyncio.run(exercise()).interaction
    assert interaction.captured_prompt is not None
    assert "SYNTHETIC-001" not in interaction.captured_prompt
    assert "selected-secret" not in interaction.captured_prompt
    assert "<redacted:account_number>" in interaction.captured_prompt
    assert "<redacted:secret>" in interaction.captured_prompt
    assert "prompts_not_captured" not in interaction.capture_omissions


def test_prompt_capture_redacts_before_json_escaping() -> None:
    project = load_config(Path("examples/telecom/scenario.yaml"))
    scenario = project.scenario.model_copy(
        update={"knowledge": {"account_number": 'quoted"account'}}
    )
    capture = PromptCapture(
        Redactor(project.privacy, resolved_secrets=['quoted"credential']), scenario
    )
    saved = capture.sanitize([{"role": "user", "content": 'quoted"account quoted"credential'}])
    assert json.loads(saved)[0]["content"] == "<redacted:account_number> <redacted:secret>"
    assert "quoted" not in saved


def test_prompt_capture_redacts_values_escaped_inside_json_message() -> None:
    project = load_config(Path("examples/telecom/scenario.yaml"))
    account = 'quoted"account\\id'
    credential = 'quoted"credential\\id'
    scenario = project.scenario.model_copy(update={"knowledge": {"account_number": account}})
    capture = PromptCapture(Redactor(project.privacy, resolved_secrets=[credential]), scenario)
    content = json.dumps({"knowledge": scenario.knowledge, "conversation": [credential]})
    saved = capture.sanitize([{"role": "user", "content": content}])
    assert "quoted" not in saved
    nested = json.loads(json.loads(saved)[0]["content"])
    assert nested["knowledge"]["account_number"] == "<redacted:account_number>"
    assert nested["conversation"] == ["<redacted:secret>"]


@pytest.mark.parametrize("kind", ["openai", "openai_compatible"])
def test_structured_request_and_absent_metadata(kind: str) -> None:
    def respond(http_request: httpx.Request) -> httpx.Response:
        body = json.loads(http_request.content)
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["schema"]["type"] == "object"
        assert body["response_format"]["json_schema"]["strict"] is True
        return httpx.Response(200, json=payload('{"answer":"ok"}', usage=None, model=None))

    async def exercise() -> LLMResponse:
        client = fake_client(httpx.MockTransport(respond))
        adapter = provider_for(kind, client)
        try:
            return await adapter.generate(request(structured=True))
        finally:
            await client.close()

    result = asyncio.run(exercise())
    assert result.interaction.usage.input_tokens is None
    assert result.interaction.usage.unavailable_reason is not None
    assert result.interaction.identity.model_version is None
    assert "provider_model_version_unavailable" in result.interaction.capture_omissions


@pytest.mark.parametrize("kind", ["openai", "openai_compatible"])
@pytest.mark.parametrize(
    "body", [payload(None), payload("  "), payload("not-json"), {"choices": []}]
)
def test_unusable_response_is_safe(kind: str, body: dict[str, Any]) -> None:
    async def exercise() -> None:
        client = fake_client(httpx.MockTransport(lambda _: httpx.Response(200, json=body)))
        adapter = provider_for(kind, client)
        try:
            with pytest.raises(ModelError, match="model response is malformed") as error:
                await adapter.generate(request(structured=True))
            assert "not-json" not in str(error.value)
        finally:
            await client.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("status,retryable", [(400, False), (401, False), (429, True), (503, True)])
def test_provider_error_is_sanitized_and_classified(status: int, retryable: bool) -> None:
    async def exercise() -> None:
        client = fake_client(
            httpx.MockTransport(
                lambda _: httpx.Response(status, json={"error": "private-secret-in-body"})
            )
        )
        adapter = provider_for("openai_compatible", client)
        try:
            with pytest.raises(ModelError) as error:
                await adapter.generate(request())
            assert error.value.retryable is retryable
            assert error.value.message == f"model request failed (HTTP {status})"
            assert "private-secret-in-body" not in str(error.value)
        finally:
            await client.close()

    asyncio.run(exercise())


def test_explicit_credentials_and_base_url_are_not_inherited_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-secret")

    async def exercise() -> None:
        compatible = provider_for("openai_compatible")
        direct = provider_for("openai")
        try:
            compatible_client = cast(AsyncOpenAI, compatible._client)
            direct_client = cast(AsyncOpenAI, direct._client)
            assert compatible_client.api_key == "selected-secret"
            assert str(compatible_client.base_url) == "http://127.0.0.1:12000/v1/"
            assert direct_client.api_key == "selected-secret"
            assert str(direct_client.base_url).startswith("https://api.openai.com/")
        finally:
            await compatible.close()
            await direct.close()
            await compatible.close()

    asyncio.run(exercise())


def test_owned_client_removes_all_ambient_sdk_routing_and_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://ambient.example/v1")
    monkeypatch.setenv("OPENAI_ORG_ID", "ambient-organization")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "ambient-project")
    monkeypatch.setenv(
        "OPENAI_CUSTOM_HEADERS", "X-Ambient-Header: ambient-value\nAuthorization: Bearer ambient"
    )
    observed: list[httpx.Request] = []

    def respond(http_request: httpx.Request) -> httpx.Response:
        observed.append(http_request)
        return httpx.Response(200, json=payload())

    original_client = _chat.AsyncOpenAI  # type: ignore[attr-defined]

    def mock_client(**kwargs: Any) -> AsyncOpenAI:
        kwargs["http_client"] = httpx.AsyncClient(
            transport=httpx.MockTransport(respond),
            event_hooks=kwargs["http_client"].event_hooks,
        )
        return original_client(**kwargs)

    monkeypatch.setattr(_chat, "AsyncOpenAI", mock_client)

    async def exercise() -> None:
        compatible = provider_for("openai_compatible")
        direct = provider_for("openai")
        try:
            await compatible.generate(request())
            await direct.generate(request())
        finally:
            await compatible.close()
            await direct.close()

    asyncio.run(exercise())
    assert [str(http_request.url) for http_request in observed] == [
        "http://127.0.0.1:12000/v1/chat/completions",
        "https://api.openai.com/v1/chat/completions",
    ]
    for http_request in observed:
        assert http_request.headers["authorization"] == "Bearer selected-secret"
        assert "openai-organization" not in http_request.headers
        assert "openai-project" not in http_request.headers
        assert "x-ambient-header" not in http_request.headers


def test_invalid_sampling_and_closed_provider_never_call_transport() -> None:
    called = False

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json=payload())

    async def exercise() -> None:
        client = fake_client(httpx.MockTransport(respond))
        adapter = provider_for("openai_compatible", client)
        try:
            with pytest.raises(ModelError, match="unsupported sampling parameter"):
                await adapter.generate(
                    request().model_copy(update={"sampling_parameters": {"x": 3}})
                )
            await adapter.close()
            with pytest.raises(ModelError, match="provider is closed"):
                await adapter.generate(request())
        finally:
            await client.close()

    asyncio.run(exercise())
    assert not called


def test_seed_override_is_sent_and_recorded_as_effective_request() -> None:
    observed: list[dict[str, Any]] = []

    def respond(http_request: httpx.Request) -> httpx.Response:
        observed.append(json.loads(http_request.content))
        return httpx.Response(200, json=payload(usage=None))

    async def exercise() -> LLMResponse:
        client = fake_client(httpx.MockTransport(respond))
        adapter = provider_for("openai_compatible", client)
        try:
            return await adapter.generate(
                request().model_copy(update={"sampling_parameters": {"seed": 42}})
            )
        finally:
            await client.close()

    result = asyncio.run(exercise())
    assert observed[0]["seed"] == 42
    assert result.interaction.identity.requested_seed == 42
    assert result.interaction.identity.sampling_parameters["requested_seed"] == 42
    assert result.interaction.usage.unavailable_reason is not None


def test_model_timeout_is_bounded_and_safe() -> None:
    async def delayed(_: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.2)
        return httpx.Response(200, json=payload())

    async def exercise() -> None:
        client = fake_client(httpx.MockTransport(delayed))
        adapter = provider_for("openai_compatible", client, timeout_seconds=0.01)
        try:
            with pytest.raises(ModelError, match="model request failed") as error:
                await adapter.generate(request())
            assert error.value.retryable
            assert error.value.message == "model request failed: timed out"
        finally:
            await client.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["refusal", "length"])
def test_response_failure_reports_only_safe_classification(failure: str) -> None:
    def respond(_: httpx.Request) -> httpx.Response:
        result = payload(None)
        if failure == "refusal":
            result["choices"][0]["message"]["refusal"] = "private-refusal-canary"
        else:
            result["choices"][0]["finish_reason"] = "length"
        return httpx.Response(200, json=result)

    async def exercise() -> None:
        client = fake_client(httpx.MockTransport(respond))
        try:
            with pytest.raises(ModelError) as caught:
                await provider_for("openai", client).generate(request(structured=True))
            assert caught.value.message == (
                "model response was refused"
                if failure == "refusal"
                else "model response was truncated"
            )
            assert "private-refusal-canary" not in str(caught.value)
        finally:
            await client.close()

    asyncio.run(exercise())

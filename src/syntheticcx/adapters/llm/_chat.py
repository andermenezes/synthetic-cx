"""Private OpenAI chat transport and translation shared by both provider adapters."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

import httpx
from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, AsyncOpenAI

from syntheticcx.adapters._openai import (
    ModelConfiguration,
    normalize_usage,
    omits_temperature,
    reported_model_version,
    sdk_options,
)
from syntheticcx.adapters._openai import (
    enforce_explicit_headers as apply_explicit_headers,
)
from syntheticcx.config.models import ResolvedModelSettings
from syntheticcx.domain.models import (
    LLMRequest,
    LLMResponse,
    ModelIdentity,
    ModelInteraction,
)
from syntheticcx.errors import ModelError
from syntheticcx.privacy.prompt_capture import PromptCapture


class _ChatCompletions(Protocol):
    def create(self, **kwargs: Any) -> Awaitable[Any]: ...


class _ChatResources(Protocol):
    completions: _ChatCompletions


class _AsyncChatClient(Protocol):
    """Private transport shape used only for deterministic adapter tests."""

    chat: _ChatResources

    async def close(self) -> None: ...


class ChatProvider:
    """A run-owned, bounded chat client; only owned response values leave this adapter."""

    def __init__(
        self,
        config: ModelConfiguration,
        settings: ResolvedModelSettings,
        *,
        timeout_seconds: float,
        client: _AsyncChatClient | None = None,
        prompt_capture: PromptCapture | None = None,
    ) -> None:
        if config.provider != settings.provider or config.model != settings.model:
            raise ValueError("model configuration and resolved settings must match")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("model timeout must be finite and positive")
        self._config = config
        self._timeout_seconds = timeout_seconds
        self._client = client or self._new_client(settings)
        self._owns_client = client is None
        self._prompt_capture = prompt_capture
        self._closed = False

    def _new_client(self, settings: ResolvedModelSettings) -> AsyncOpenAI:
        """Create a client whose destination and credentials cannot inherit ambient settings."""
        credential = settings.credential.get_secret_value()

        async def enforce_explicit_headers(http_request: httpx.Request) -> None:
            apply_explicit_headers(http_request, credential)

        return AsyncOpenAI(
            **sdk_options(self._config, credential, self._timeout_seconds),
            # The SDK also reads these values and OPENAI_CUSTOM_HEADERS from
            # the ambient environment. A request hook removes every ambient
            # header (including an ambient Authorization override) before I/O.
            http_client=httpx.AsyncClient(
                timeout=self._timeout_seconds,
                event_hooks={"request": [enforce_explicit_headers]},
            ),
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Send one generation request and translate its response without raw diagnostics."""
        if self._closed:
            raise ModelError("model.generate", "model provider is closed")
        parameters, identity_parameters, omissions = self._parameters(request)
        started = datetime.now(UTC)
        try:
            # The SDK timeout bounds HTTP phases; this outer deadline also
            # bounds client-side work around the request.
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self._config.model,
                    messages=[
                        {"role": message.role, "content": message.content}
                        for message in request.messages
                    ],
                    **parameters,
                ),
                self._timeout_seconds,
            )
        except asyncio.CancelledError:
            raise
        except (TimeoutError, APITimeoutError):
            raise ModelError(
                "model.generate", "model request failed: timed out", retryable=True
            ) from None
        except APIConnectionError:
            raise ModelError(
                "model.generate", "model request failed: connection error", retryable=True
            ) from None
        except APIStatusError as exc:
            retryable = exc.status_code == 429 or exc.status_code >= 500
            raise ModelError(
                "model.generate",
                f"model request failed (HTTP {exc.status_code})",
                retryable=retryable,
            ) from None
        except APIError:
            raise ModelError("model.generate", "model request failed") from None
        except Exception:
            raise ModelError("model.generate", "model request failed") from None

        ended = datetime.now(UTC)
        try:
            choice = response.choices[0]
            if getattr(choice.message, "refusal", None):
                raise ModelError("model.generate", "model response was refused")
            if getattr(choice, "finish_reason", None) == "length":
                raise ModelError("model.generate", "model response was truncated")
            content = choice.message.content
            if not isinstance(content, str) or not content.strip():
                raise ValueError("missing content")
            if request.response_schema is not None:
                json.loads(content)
            token_usage = normalize_usage(response.usage)
            reported_model = reported_model_version(response, omissions)
            return LLMResponse(
                content=content,
                interaction=ModelInteraction(
                    id=request.id + "-" + str(uuid4()),
                    identity=ModelIdentity(
                        provider=self._config.provider,
                        model=self._config.model,
                        model_version=reported_model,
                        requested_seed=identity_parameters.get("requested_seed"),
                        sampling_parameters=identity_parameters,
                    ),
                    started_at=started,
                    ended_at=ended,
                    usage=token_usage,
                    captured_prompt=self._prompt_capture.sanitize(
                        [message.model_dump(mode="json") for message in request.messages]
                    )
                    if self._prompt_capture is not None
                    else None,
                    capture_omissions=[
                        *(["prompts_not_captured"] if self._prompt_capture is None else []),
                        "model_output_not_captured",
                        *omissions,
                    ],
                ),
            )
        except ModelError:
            raise
        except Exception:
            raise ModelError("model.generate", "model response is malformed") from None

    def _parameters(self, request: LLMRequest) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
        supplied = request.sampling_parameters
        if set(supplied) - {"temperature", "seed"}:
            raise ModelError("model.generate", "unsupported sampling parameter")
        temperature = supplied.get("temperature", self._config.temperature)
        seed = supplied.get("seed", self._config.seed)
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ModelError("model.generate", "invalid temperature")
        if not math.isfinite(temperature) or not 0 <= temperature <= 1:
            raise ModelError("model.generate", "invalid temperature")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ModelError("model.generate", "invalid seed")
        parameters: dict[str, Any] = {}
        omissions: list[str] = []
        identity: dict[str, Any] = {"requested_temperature": temperature}
        if omits_temperature(self._config):
            omissions.append("temperature_not_applied")
        else:
            parameters["temperature"] = temperature
            identity["effective_temperature"] = temperature
        if seed is not None:
            parameters["seed"] = seed
            identity["requested_seed"] = seed
        if request.response_schema is not None:
            parameters["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "syntheticcx_response",
                    "schema": request.response_schema,
                    "strict": True,
                },
            }
        return parameters, identity, omissions

    async def close(self) -> None:
        """Release a transport created by this provider; repeated calls are safe."""
        if not self._closed:
            self._closed = True
            if self._owns_client:
                await self._client.close()

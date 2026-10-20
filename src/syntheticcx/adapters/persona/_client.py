"""Private synchronous model client for the pinned TinyTroupe action path."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx
from openai import OpenAI

from syntheticcx.adapters._openai import (
    ModelConfiguration,
    enforce_explicit_headers,
    normalize_usage,
    omits_temperature,
    reported_model_version,
    sdk_options,
)
from syntheticcx.config.models import (
    ResolvedModelSettings,
)
from syntheticcx.domain.models import ModelIdentity, ModelInteraction
from syntheticcx.privacy.prompt_capture import PromptCapture


class TinyModelClient:
    """Expose only TinyTroupe's exercised synchronous `send_message` surface."""

    def __init__(
        self,
        config: ModelConfiguration,
        settings: ResolvedModelSettings,
        timeout_seconds: float,
        prompt_capture: PromptCapture | None = None,
    ) -> None:
        if (config.provider, config.model) != (settings.provider, settings.model):
            raise ValueError("model settings do not match configuration")
        credential = settings.credential.get_secret_value()

        def enforce_headers(request: httpx.Request) -> None:
            enforce_explicit_headers(request, credential)

        self._sdk = OpenAI(
            **sdk_options(config, credential, timeout_seconds),
            http_client=httpx.Client(
                timeout=timeout_seconds, event_hooks={"request": [enforce_headers]}
            ),
        )
        self._config = config
        self._prompt_capture = prompt_capture
        self.interactions: list[ModelInteraction] = []

    def send_message(
        self,
        current_messages: list[dict[str, Any]],
        *,
        response_format: type[Any],
        model: str | None = None,
        dedent_messages: bool = True,
        **kwargs: Any,
    ) -> dict[str, str]:
        """Return TinyTroupe's role/content mapping, retaining safe call metadata."""
        if kwargs or not dedent_messages or (model is not None and model != self._config.model):
            raise ValueError("unsupported TinyTroupe model request")
        parameters: dict[str, Any] = {}
        temperature_omitted = omits_temperature(self._config)
        if not temperature_omitted:
            parameters["temperature"] = self._config.temperature
        if self._config.seed is not None:
            parameters["seed"] = self._config.seed
        started = datetime.now(UTC)
        response = self._sdk.beta.chat.completions.parse(
            model=self._config.model,
            messages=current_messages,
            response_format=response_format,
            **parameters,
        )
        ended = datetime.now(UTC)
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("model response is empty")
        # This parse method enforces the Pydantic action shape before TinyTroupe
        # extracts its JSON. Do not include the raw content in evidence.
        json.loads(content)
        omissions: list[str] = []
        tokens = normalize_usage(response.usage)
        self.interactions.append(
            ModelInteraction(
                id=str(uuid4()),
                identity=ModelIdentity(
                    provider=self._config.provider,
                    model=self._config.model,
                    model_version=reported_model_version(response, omissions),
                    requested_seed=self._config.seed,
                    sampling_parameters={
                        "requested_temperature": self._config.temperature,
                        **(
                            {"effective_temperature": self._config.temperature}
                            if "temperature" in parameters
                            else {}
                        ),
                    },
                ),
                started_at=started,
                ended_at=ended,
                usage=tokens,
                captured_prompt=(
                    self._prompt_capture.sanitize(current_messages)
                    if self._prompt_capture is not None
                    else None
                ),
                capture_omissions=[
                    *(["prompts_not_captured"] if self._prompt_capture is None else []),
                    "model_output_not_captured",
                    *(["temperature_not_applied"] if temperature_omitted else []),
                    *omissions,
                ],
            )
        )
        return {"role": "assistant", "content": content}

    def close(self) -> None:
        self._sdk.close()

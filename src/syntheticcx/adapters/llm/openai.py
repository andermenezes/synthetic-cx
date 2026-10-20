"""Explicitly credentialed OpenAI chat provider."""

from __future__ import annotations

from syntheticcx.adapters.llm._chat import ChatProvider, _AsyncChatClient
from syntheticcx.config.models import OpenAIModelConfig, ResolvedModelSettings
from syntheticcx.privacy.prompt_capture import PromptCapture


class OpenAIProvider(ChatProvider):
    def __init__(
        self,
        config: OpenAIModelConfig,
        settings: ResolvedModelSettings,
        *,
        timeout_seconds: float,
        client: _AsyncChatClient | None = None,
        prompt_capture: PromptCapture | None = None,
    ) -> None:
        super().__init__(
            config,
            settings,
            timeout_seconds=timeout_seconds,
            client=client,
            prompt_capture=prompt_capture,
        )

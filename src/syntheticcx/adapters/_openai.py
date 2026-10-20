"""Private OpenAI SDK configuration and metadata helpers shared by adapters."""

from __future__ import annotations

import os
from typing import Any

import httpx

from syntheticcx.config.models import OpenAICompatibleModelConfig, OpenAIModelConfig
from syntheticcx.domain.models import TokenUsage

ModelConfiguration = OpenAIModelConfig | OpenAICompatibleModelConfig


def explicit_base_url(config: ModelConfiguration) -> str:
    """Return the configured origin without consulting ambient SDK configuration."""
    if isinstance(config, OpenAICompatibleModelConfig):
        return str(config.base_url)
    return "https://api.openai.com/v1"


def ambient_header_names() -> set[str]:
    """Return headers the pinned SDK could otherwise import from its environment."""
    names = {"Authorization", "OpenAI-Organization", "OpenAI-Project"}
    for line in os.environ.get("OPENAI_CUSTOM_HEADERS", "").splitlines():
        name, separator, _ = line.partition(":")
        if separator and name.strip():
            names.add(name.strip())
    return names


def enforce_explicit_headers(request: httpx.Request, credential: str) -> None:
    """Remove ambient SDK headers and apply the explicitly selected credential."""
    for name in ambient_header_names():
        if name in request.headers:
            del request.headers[name]
    request.headers["Authorization"] = f"Bearer {credential}"


def sdk_options(
    config: ModelConfiguration, credential: str, timeout_seconds: float
) -> dict[str, Any]:
    """Return SDK arguments that prevent ambient destination/identity inheritance."""
    return {
        "api_key": credential,
        "base_url": explicit_base_url(config),
        "organization": "",
        "project": "",
        "admin_api_key": "",
        "webhook_secret": "",
        "timeout": timeout_seconds,
        "max_retries": 0,
    }


def omits_temperature(config: ModelConfiguration) -> bool:
    """Return whether the reference OpenAI GPT-5 path rejects temperature."""
    return config.provider == "openai" and config.model.startswith("gpt-5")


def normalize_usage(usage: Any) -> TokenUsage:
    """Represent incomplete provider usage explicitly rather than as zero values."""
    input_tokens = getattr(usage, "prompt_tokens", None)
    output_tokens = getattr(usage, "completion_tokens", None)
    if (
        type(input_tokens) is int
        and type(output_tokens) is int
        and input_tokens >= 0
        and output_tokens >= 0
    ):
        return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
    return TokenUsage(unavailable_reason="provider did not report complete usage")


def reported_model_version(response: Any, omissions: list[str]) -> str | None:
    """Extract model version and make its absence explicit in interaction evidence."""
    model = getattr(response, "model", None)
    if isinstance(model, str) and model:
        return model
    omissions.append("provider_model_version_unavailable")
    return None

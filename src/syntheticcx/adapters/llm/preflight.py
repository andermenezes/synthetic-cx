"""Bounded, credential-specific provider probes before any browser activity."""

from __future__ import annotations

import asyncio
import json

from pydantic import Field

from syntheticcx.adapters.llm import OpenAICompatibleProvider, OpenAIProvider
from syntheticcx.config.models import (
    OpenAICompatibleModelConfig,
    OpenAIModelConfig,
    ProjectConfig,
    ProviderCapabilities,
    ResolvedModelSettings,
    StrictModel,
)
from syntheticcx.config.settings import PreparedSettings, requires_semantic_evaluation
from syntheticcx.domain.models import LLMMessage, LLMRequest, LLMResponse
from syntheticcx.domain.protocols import LLMProvider
from syntheticcx.errors import ConfigurationError, ModelError

_PROBE_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


class CapabilityPreflight(StrictModel):
    """Owned metadata snapshots; false also includes optional, untested features."""

    verified_capabilities: dict[str, ProviderCapabilities] = Field(default_factory=dict)
    unavailable_capabilities: dict[str, list[str]] = Field(default_factory=dict)


def _provider(
    config: OpenAIModelConfig | OpenAICompatibleModelConfig,
    settings: ResolvedModelSettings,
    timeout: float,
) -> OpenAIProvider | OpenAICompatibleProvider:
    if isinstance(config, OpenAICompatibleModelConfig):
        return OpenAICompatibleProvider(config, settings, timeout_seconds=timeout)
    return OpenAIProvider(config, settings, timeout_seconds=timeout)


async def _probe(
    provider: LLMProvider,
    *,
    role: str,
    structured: bool,
) -> LLMResponse:
    request = LLMRequest(
        id=f"preflight-{role}-{'structured' if structured else 'chat'}",
        messages=[LLMMessage(role="user", content="Reply with a short acknowledgement.")],
        response_schema=_PROBE_SCHEMA if structured else None,
    )
    try:
        response = await provider.generate(request)
        if structured and json.loads(response.content) != {"ok": True}:
            raise ValueError("structured probe returned an unexpected shape")
        return response
    except (ModelError, ValueError):
        capability = "structured_output" if structured else "chat_completions"
        raise ConfigurationError(
            "probe model capabilities", f"{role}.{capability} is unavailable"
        ) from None


async def _probe_role(
    config: OpenAIModelConfig | OpenAICompatibleModelConfig,
    settings: ResolvedModelSettings,
    *,
    role: str,
    timeout: float,
    close_timeout: float,
) -> tuple[ProviderCapabilities, list[str]]:
    # prepare_settings owns configuration declaration checks. This layer only
    # verifies the declared required features against the selected endpoint.
    provider = _provider(config, settings, timeout)
    try:
        chat = await _probe(provider, role=role, structured=False)
        structured = await _probe(provider, role=role, structured=True)
    finally:
        try:
            await asyncio.wait_for(provider.close(), close_timeout)
        except Exception:
            raise ConfigurationError("probe model capabilities", f"{role}.close failed") from None

    declared = config.capabilities if isinstance(config, OpenAICompatibleModelConfig) else None
    available_usage = all(
        response.interaction.usage.unavailable_reason is None for response in (chat, structured)
    )
    available_version = all(
        response.interaction.identity.model_version is not None for response in (chat, structured)
    )
    verified = ProviderCapabilities(
        chat_completions=True,
        structured_output=True,
        embeddings=False,  # The selected TinyPerson action path does not request embeddings.
        token_usage=available_usage and (declared is None or declared.token_usage),
        model_version=available_version and (declared is None or declared.model_version),
    )
    unavailable = [
        name
        for name in ("embeddings", "token_usage", "model_version")
        if not getattr(verified, name)
    ]
    return verified, unavailable


async def preflight_models(
    config: ProjectConfig,
    settings: PreparedSettings,
) -> CapabilityPreflight:
    """Check each active model role before constructing or starting a browser.

    Parsing and credential resolution must already have succeeded. This function
    performs no browser work and never returns probe prompts or generated text.
    """
    roles: list[
        tuple[str, OpenAIModelConfig | OpenAICompatibleModelConfig, ResolvedModelSettings]
    ] = [("persona", config.model, settings.persona)]
    if requires_semantic_evaluation(config):
        roles.append(("evaluation", config.effective_evaluation_model, settings.evaluation))

    verified: dict[str, ProviderCapabilities] = {}
    unavailable: dict[str, list[str]] = {}
    for role, model, resolved in roles:
        if role == "evaluation" and model == config.model and resolved == settings.persona:
            verified[role] = verified["persona"]
            unavailable[role] = list(unavailable["persona"])
            continue
        verified[role], unavailable[role] = await _probe_role(
            model,
            resolved,
            role=role,
            timeout=config.execution.model_timeout_seconds,
            close_timeout=config.execution.close_timeout_seconds,
        )
    return CapabilityPreflight(verified_capabilities=verified, unavailable_capabilities=unavailable)

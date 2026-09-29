"""Compose one bounded discovery attempt without publishing unverified controls."""

from __future__ import annotations

import asyncio
import os
import shutil
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path

import yaml
from pydantic import TypeAdapter, ValidationError

from syntheticcx.adapters.discovery.playwright_mcp import ManagedPlaywrightMCP
from syntheticcx.adapters.llm import OpenAICompatibleProvider, OpenAIProvider
from syntheticcx.adapters.llm.preflight import _probe_role
from syntheticcx.config.loader import load_discovery_draft
from syntheticcx.config.models import (
    ModelConfig,
    OpenAICompatibleModelConfig,
    OpenAIModelConfig,
    ResolvedModelSettings,
)
from syntheticcx.discovery.explore import DiscoveryCandidate, explore
from syntheticcx.errors import ConfigurationError
from syntheticcx.privacy import Redactor


def _selected_model(
    path: Path | None, inherited: OpenAIModelConfig | OpenAICompatibleModelConfig
) -> OpenAIModelConfig | OpenAICompatibleModelConfig:
    if path is None:
        return inherited
    try:
        model = yaml.safe_load(path.read_text(encoding="utf-8"))
        return TypeAdapter(ModelConfig).validate_python(model)  # type: ignore[no-any-return]
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError):
        raise ConfigurationError(
            "discover model", "discovery model configuration is invalid"
        ) from None


async def discover_candidate(
    scenario: Path,
    url: str,
    *,
    discovery_model_config: Path | None = None,
    headed: bool = False,
    chat_hint: str | None = None,
    max_discovery_attempts: int = 3,
    on_progress: Callable[[str], None] | None = None,
) -> DiscoveryCandidate:
    """Validate credentials, probe the selected model, and explore with owned resources."""
    draft = load_discovery_draft(scenario)
    if on_progress is not None:
        on_progress("scenario draft loaded")
    model = _selected_model(discovery_model_config, draft.model)
    if isinstance(model, OpenAICompatibleModelConfig) and (
        not model.capabilities.chat_completions or not model.capabilities.structured_output
    ):
        raise ConfigurationError(
            "discover model", "chat_completions and structured_output are required"
        )
    credential = os.environ.get(model.credential_environment_variable)
    if not credential:
        raise ConfigurationError(
            "discover credentials",
            f"environment variable {model.credential_environment_variable} is required",
        )
    settings = ResolvedModelSettings(
        provider=model.provider, model=model.model, credential=credential
    )
    redactor = Redactor(draft.privacy, resolved_secrets=[credential])
    safe_chat_hint = (
        redactor.redact_text_with_field_values(
            chat_hint,
            {
                "starting_context": draft.scenario.starting_context,
                "knowledge": draft.scenario.knowledge,
            },
        )
        if chat_hint is not None
        else None
    )
    if find_spec("mcp") is None or shutil.which("npx") is None:
        raise ConfigurationError("discover setup", "install the discovery extra and Node.js")
    if on_progress is not None:
        on_progress("checking selected discovery model capabilities")
    await _probe_role(
        model,
        settings,
        role="discovery",
        timeout=draft.execution.model_timeout_seconds,
        close_timeout=draft.execution.close_timeout_seconds,
    )
    if on_progress is not None:
        on_progress("discovery model capability check passed")
    provider = (
        OpenAICompatibleProvider(
            model, settings, timeout_seconds=draft.execution.model_timeout_seconds
        )
        if isinstance(model, OpenAICompatibleModelConfig)
        else OpenAIProvider(model, settings, timeout_seconds=draft.execution.model_timeout_seconds)
    )
    try:
        if on_progress is not None:
            mode = "headed" if headed else "headless"
            on_progress(f"starting managed Playwright MCP browser ({mode})")
        async with (
            asyncio.timeout(draft.execution.journey_timeout_seconds),
            ManagedPlaywrightMCP(
                url,
                timeout_seconds=draft.execution.browser_action_timeout_seconds,
                headed=headed,
                chat_hint=chat_hint,
            ) as browser,
        ):
            if on_progress is not None:
                on_progress("managed Playwright MCP browser started")
            return await explore(
                url,
                browser,
                provider,
                redactor,
                model_timeout_seconds=draft.execution.model_timeout_seconds,
                sensitive_context={
                    "starting_context": draft.scenario.starting_context,
                    "knowledge": draft.scenario.knowledge,
                },
                chat_hint=safe_chat_hint,
                max_discovery_attempts=max_discovery_attempts,
                on_progress=on_progress,
            )
    finally:
        await asyncio.wait_for(provider.close(), draft.execution.close_timeout_seconds)

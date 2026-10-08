"""Compose one bounded discovery attempt without publishing unverified controls."""

from __future__ import annotations

import asyncio
import os
import shutil
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from importlib.util import find_spec
from pathlib import Path
from time import monotonic
from typing import cast

import yaml
from pydantic import AnyHttpUrl, TypeAdapter, ValidationError

from syntheticcx.adapters.discovery.playwright_mcp import ManagedPlaywrightMCP
from syntheticcx.adapters.llm import OpenAICompatibleProvider, OpenAIProvider
from syntheticcx.adapters.llm.preflight import _probe_role
from syntheticcx.config.loader import load_discovery_draft
from syntheticcx.config.models import (
    InterfaceChannelGuide,
    ModelConfig,
    OpenAICompatibleModelConfig,
    OpenAIModelConfig,
    ResolvedModelSettings,
)
from syntheticcx.discovery.calibrate import calibrate_candidate
from syntheticcx.discovery.explore import DiscoveryCandidate, explore
from syntheticcx.discovery.hints import launcher_id
from syntheticcx.errors import ChannelError, ConfigurationError
from syntheticcx.language import Language, first_probe
from syntheticcx.privacy import Redactor


@asynccontextmanager
async def _discovery_deadline(deadline: float) -> AsyncIterator[None]:
    try:
        async with asyncio.timeout_at(deadline):
            yield
    except TimeoutError:
        raise ChannelError(
            "discover budget", "discovery time or operation budget exhausted"
        ) from None


async def _close_discovery(
    operation: Awaitable[None], *, deadline: float, close_timeout_seconds: float, slots: int = 1
) -> bool:
    """Cleanup never receives a new allowance beyond the owned absolute deadline."""
    try:
        await asyncio.wait_for(
            operation, timeout=max(0, min(close_timeout_seconds, (deadline - monotonic()) / slots))
        )
        return True
    except Exception:
        return False


def _cleanup_deadline(deadline: float, slots: int = 1) -> float:
    now = monotonic()
    return now + max(0, deadline - now) / slots


def _selected_model(
    path: Path | None, inherited: OpenAIModelConfig | OpenAICompatibleModelConfig
) -> OpenAIModelConfig | OpenAICompatibleModelConfig:
    if path is None:
        return inherited
    try:
        model = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cast(
            OpenAIModelConfig | OpenAICompatibleModelConfig,
            TypeAdapter(ModelConfig).validate_python(model),
        )
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
    probe_message: str | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> DiscoveryCandidate:
    """Validate credentials, probe the selected model, and explore with owned resources."""
    id_launcher_requested = launcher_id(chat_hint) is not None
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
        async with asyncio.timeout(draft.execution.journey_timeout_seconds):
            async with ManagedPlaywrightMCP(
                url,
                timeout_seconds=draft.execution.browser_action_timeout_seconds,
                headed=headed,
                chat_hint=chat_hint,
            ) as browser:
                if on_progress is not None:
                    on_progress("managed Playwright MCP browser started")
                candidate = await explore(
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
                    id_launcher_requested=id_launcher_requested,
                    max_discovery_attempts=max_discovery_attempts,
                    on_progress=on_progress,
                )
            return await calibrate_candidate(
                candidate,
                draft.execution,
                probe_message
                if probe_message is not None
                else first_probe(draft.persona.communication.language),
                provider,
                redactor,
                headed=headed,
                sensitive_context={
                    "starting_context": draft.scenario.starting_context,
                    "knowledge": draft.scenario.knowledge,
                },
                max_attempts=max_discovery_attempts,
                on_progress=on_progress,
            )
    finally:
        await asyncio.wait_for(provider.close(), draft.execution.close_timeout_seconds)


async def discover_guide(
    scenario: Path,
    url: str,
    *,
    discovery_model_config: Path | None = None,
    headed: bool = False,
    chat_hint: str | None = None,
    max_discovery_attempts: int = 3,
    probe_message: str | None = None,
    language: Language | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> InterfaceChannelGuide:
    """Own model preflight, one learning browser and one fresh replay deadline."""
    from syntheticcx.discovery.budget import DiscoveryBudget
    from syntheticcx.discovery.learn import learn
    from syntheticcx.discovery.replay import replay
    from syntheticcx.language import second_probe, third_probe

    if not 1 <= max_discovery_attempts <= 5:
        raise ConfigurationError(
            "discover budget", "max_discovery_attempts must be between 1 and 5"
        )
    try:
        target_url = TypeAdapter(AnyHttpUrl).validate_python(url)
    except ValidationError:
        raise ConfigurationError("discover URL", "target must be an HTTP or HTTPS URL") from None
    if target_url.username is not None or target_url.password is not None:
        raise ConfigurationError("discover URL", "URL credentials are not permitted")
    draft = load_discovery_draft(scenario)
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
    context = {
        "persona": draft.persona.model_dump(mode="json"),
        "starting_context": draft.scenario.starting_context,
        "knowledge": draft.scenario.knowledge,
    }
    redactor = Redactor(
        draft.privacy,
        resolved_secrets=[credential, url, *([probe_message] if probe_message else [])],
        sensitive_context=context,
    )
    selected_language = language if language is not None else draft.persona.communication.language
    probes = {
        "first": first_probe(selected_language),
        "second": second_probe(selected_language),
        "third": third_probe(selected_language),
    }
    if probe_message is not None:
        if not probe_message.strip():
            raise ConfigurationError("discover probe", "explicit probe must be nonempty")
        probes = {"explicit": probe_message}
    provider = (
        OpenAICompatibleProvider(
            model, settings, timeout_seconds=draft.execution.model_timeout_seconds
        )
        if isinstance(model, OpenAICompatibleModelConfig)
        else OpenAIProvider(model, settings, timeout_seconds=draft.execution.model_timeout_seconds)
    )
    limits = draft.execution
    # The outside deadline includes both sessions, capability checks and all cleanup.
    cleanup_reserve = min(2 * limits.close_timeout_seconds, limits.journey_timeout_seconds / 3)
    budget = DiscoveryBudget.create(limits.journey_timeout_seconds, reserve_seconds=cleanup_reserve)
    learning = fresh = None
    failed = False
    async with _discovery_deadline(budget.deadline):
        try:
            await budget.run(
                _probe_role(
                    model,
                    settings,
                    role="discovery",
                    timeout=limits.model_timeout_seconds,
                    close_timeout=limits.close_timeout_seconds,
                ),
                limits.model_timeout_seconds + limits.close_timeout_seconds,
            )
            from syntheticcx.adapters.channel.broker import NativeChatBroker

            if on_progress:
                on_progress("starting owned autonomous learning browser")
            learning = NativeChatBroker(
                url, limits, headed=headed, chat_hint=chat_hint, contract=None
            )
            await budget.run(
                learning.start(on_progress=on_progress), limits.browser_action_timeout_seconds
            )
            remaining = budget.deadline - cleanup_reserve - monotonic()
            reserve = cleanup_reserve + min(
                remaining * 0.5,
                3 * limits.response_timeout_seconds + limits.close_timeout_seconds,
            )
            learned = await learn(
                learning,
                provider,
                redactor,
                budget=DiscoveryBudget(budget.deadline, reserve),
                probes=probes,
                model_timeout_seconds=limits.model_timeout_seconds,
                browser_action_timeout_seconds=limits.browser_action_timeout_seconds,
                response_timeout_seconds=limits.response_timeout_seconds,
                max_discovery_attempts=max_discovery_attempts,
                on_progress=on_progress,
            )
            if not await _close_discovery(
                learning.close(deadline=budget.deadline - cleanup_reserve),
                deadline=budget.deadline - cleanup_reserve,
                close_timeout_seconds=limits.close_timeout_seconds,
            ):
                raise ChannelError("discover cleanup", "learning browser cleanup did not complete")
            learning = None
            if on_progress:
                on_progress("replaying learned recipe in one fresh browser")
            fresh = NativeChatBroker(
                url, limits, headed=headed, chat_hint=chat_hint, contract=learned.config
            )
            await budget.run(
                fresh.start(on_progress=on_progress), limits.browser_action_timeout_seconds
            )
            guide = await replay(
                fresh,
                learned,
                budget=DiscoveryBudget(budget.deadline, cleanup_reserve),
                browser_action_timeout_seconds=limits.browser_action_timeout_seconds,
                response_timeout_seconds=limits.response_timeout_seconds,
                on_progress=on_progress,
            )
            from syntheticcx.config.sanitize import sanitized_channel_guide

            sanitized_channel_guide(
                guide,
                draft.privacy,
                resolved_secrets=[credential, *([probe_message] if probe_message else [])],
                sensitive_context=context,
            )
            return guide
        except BaseException:
            failed = True
            raise
        finally:
            cleanup_ok = True
            try:
                if learning is not None:
                    cleanup_ok = (
                        await _close_discovery(
                            learning.close(
                                deadline=_cleanup_deadline(
                                    budget.deadline, 2 + int(fresh is not None)
                                )
                            ),
                            deadline=budget.deadline,
                            close_timeout_seconds=limits.close_timeout_seconds,
                            slots=2 + int(fresh is not None),
                        )
                        and cleanup_ok
                    )
            finally:
                try:
                    if fresh is not None:
                        cleanup_ok = (
                            await _close_discovery(
                                fresh.close(deadline=_cleanup_deadline(budget.deadline, 2)),
                                deadline=budget.deadline,
                                close_timeout_seconds=limits.close_timeout_seconds,
                                slots=2,
                            )
                            and cleanup_ok
                        )
                finally:
                    cleanup_ok = (
                        await _close_discovery(
                            provider.close(),
                            deadline=budget.deadline,
                            close_timeout_seconds=limits.close_timeout_seconds,
                        )
                        and cleanup_ok
                    )
            if not failed and not cleanup_ok:
                raise ChannelError("discover cleanup", "resource cleanup exceeded discovery budget")

"""Verify a website-discovered WhatsApp entry with one owned probe and exact replay.

A new browser does not reset the server-side conversation. Only the channel's
current-session, recipient-bound echo and incremental incoming evidence qualify.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any
from urllib.parse import urlsplit

from pydantic import AnyHttpUrl, TypeAdapter, ValidationError

from syntheticcx.adapters.channel.broker import NativeChatBroker
from syntheticcx.adapters.channel.whatsapp import (
    WhatsAppWebChannel,
    resolve_recipient,
    validate_browser_profile,
)
from syntheticcx.adapters.llm import OpenAICompatibleProvider, OpenAIProvider
from syntheticcx.adapters.llm.preflight import _probe_role
from syntheticcx.config.loader import load_discovery_draft
from syntheticcx.config.models import (
    OpenAICompatibleModelConfig,
    ResolvedModelSettings,
    WhatsAppChannelConfig,
    WhatsAppChannelGuide,
    WhatsAppGuideValidation,
)
from syntheticcx.config.sanitize import sanitized_channel_guide
from syntheticcx.config.settings import validate_behavior
from syntheticcx.discovery.budget import DiscoveryBudget
from syntheticcx.discovery.learn import explore_website
from syntheticcx.discovery.policy import (
    portability_flow,
    require_discovery_probe,
    require_discovery_target,
)
from syntheticcx.domain.models import CompletedResponse, SendReceipt
from syntheticcx.errors import ChannelError, ConfigurationError
from syntheticcx.language import Language, first_probe
from syntheticcx.privacy import Redactor


async def _exchange(
    channel: WhatsAppWebChannel,
    probe: str,
    *,
    budget: DiscoveryBudget,
    action_timeout: float,
    response_timeout: float,
    on_progress: Callable[[str], None] | None = None,
) -> None:
    budget.consume("observations")
    await budget.run(channel.observe_initial(), response_timeout)
    budget.consume("actions")
    if on_progress:
        on_progress("submitting owned text probe to the verified conversation")
    receipt = await budget.run(channel.send(probe), action_timeout)
    if not isinstance(receipt, SendReceipt) or receipt.evidence.get("observed_echo") is not True:
        raise ChannelError("discover probe", "owned submission echo was not verified")
    if on_progress:
        on_progress("owned outgoing echo verified; waiting for a fresh reply")
    budget.consume("observations")
    response = await budget.run(channel.receive(), response_timeout)
    if (
        not isinstance(response, CompletedResponse)
        or not channel.response_bubble_attached
        or response.started_at < receipt.submitted_at
        or any(
            not message.content.strip() or message.observed_at < receipt.submitted_at
            for message in response.messages
        )
    ):
        raise ChannelError("discover probe", "fresh completed incoming response was not verified")
    if any(portability_flow(message.content) for message in response.messages):
        raise ChannelError("discover probe", "number-portability flows are forbidden in discovery")
    if on_progress:
        on_progress("fresh completed reply verified")


async def discover_whatsapp_guide(
    scenario: Path,
    url: str,
    *,
    browser_profile: Path,
    recipient: str | None = None,
    discovery_model_config: Path | None = None,
    headed: bool = False,
    chat_hint: str | None = None,
    max_discovery_attempts: int = 3,
    probe_message: str | None = None,
    language: Language | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> WhatsAppChannelGuide:
    """Publishable evidence requires two fresh sessions, never messaging retries."""
    require_discovery_probe(probe_message)
    require_discovery_target(url)
    from syntheticcx.discovery.run import (
        _close_discovery,
        _discovery_deadline,
        _selected_model,
    )

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
    validate_behavior(draft.behavior, draft.persona.behavioral_rules)
    if draft.storage.store_screenshots or draft.storage.store_video:
        raise ConfigurationError(
            "discover privacy", "WhatsApp account-wide screenshots and video are unsupported"
        )
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
    try:
        config = WhatsAppChannelConfig(type="whatsapp_web_playwright", url=target_url)
    except ValidationError:
        raise ConfigurationError(
            "discover URL",
            "WhatsApp discovery requires a public website URL without protected data",
        ) from None
    recipient_value = (
        recipient
        if recipient is not None
        else os.environ.get(config.recipient_environment_variable, "")
    )
    digits = resolve_recipient(config, recipient)
    browser_profile = validate_browser_profile(browser_profile)
    selected_language = language if language is not None else draft.persona.communication.language
    probe = probe_message if probe_message is not None else first_probe(selected_language)
    if not probe.strip():
        raise ConfigurationError("discover probe", "explicit probe must be nonempty")
    context = {
        "persona": draft.persona.model_dump(mode="json"),
        "starting_context": draft.scenario.starting_context,
        "knowledge": draft.scenario.knowledge,
    }
    secrets = [credential, recipient_value, digits, probe]
    redactor = Redactor(draft.privacy, resolved_secrets=secrets, sensitive_context=context)
    settings = ResolvedModelSettings(
        provider=model.provider, model=model.model, credential=credential
    )
    limits = draft.execution
    cleanup_reserve = min(3 * limits.close_timeout_seconds, limits.journey_timeout_seconds / 3)
    budget = DiscoveryBudget.create(limits.journey_timeout_seconds, reserve_seconds=cleanup_reserve)
    provider = (
        OpenAICompatibleProvider(model, settings, timeout_seconds=limits.model_timeout_seconds)
        if isinstance(model, OpenAICompatibleModelConfig)
        else OpenAIProvider(model, settings, timeout_seconds=limits.model_timeout_seconds)
    )
    channel: WhatsAppWebChannel | None = None
    failed = False
    async with _discovery_deadline(budget.deadline):
        try:
            if on_progress:
                on_progress("checking selected discovery model capabilities")
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

            async def explore_entry(browser: NativeChatBroker) -> None:
                browser.chat_hint = chat_hint
                remaining = max(0, budget.deadline - cleanup_reserve - monotonic())
                verification_reserve = min(
                    remaining * 0.5,
                    2 * limits.response_timeout_seconds
                    + limits.browser_action_timeout_seconds
                    + limits.close_timeout_seconds,
                )
                website_budget = DiscoveryBudget(
                    budget.deadline, cleanup_reserve + verification_reserve
                )
                await explore_website(
                    browser,
                    provider,
                    redactor,
                    budget=website_budget,
                    target="whatsapp",
                    exploration_context={
                        "description": draft.scenario.description,
                        "goal": draft.scenario.goal,
                        "starting_context": draft.scenario.starting_context,
                        "language": selected_language,
                    },
                    website_action_limit=limits.website_action_limit,
                    website_observation_limit=limits.website_observation_limit,
                    model_timeout_seconds=limits.model_timeout_seconds,
                    browser_action_timeout_seconds=limits.browser_action_timeout_seconds,
                    max_discovery_attempts=max_discovery_attempts,
                    on_progress=on_progress,
                )

            authorizations: list[dict[str, Any]] = [
                item.model_dump(mode="json") for item in draft.scenario.action_authorizations
            ]
            for session in range(2):
                budget.check()
                if on_progress:
                    on_progress(
                        "verifying owned WhatsApp probe"
                        if session == 0
                        else "replaying exact probe in a fresh browser; server state persists"
                    )
                channel = WhatsAppWebChannel(
                    config,
                    limits,
                    browser_profile=browser_profile,
                    recipient=digits,
                    headed=headed,
                    action_authorizations=authorizations,
                    website_explorer=explore_entry if session == 0 else None,
                    discovery_only=True,
                )
                await budget.run(
                    channel.start(),
                    limits.journey_timeout_seconds
                    if session == 0
                    else limits.browser_action_timeout_seconds * (len(config.navigation_steps) + 1),
                )
                if on_progress:
                    on_progress("recipient verified and text composer ready")
                config = channel.resolved_config
                if not config.navigation_steps:
                    raise ChannelError("discover entry", "native entry contract is unavailable")
                await _exchange(
                    channel,
                    probe,
                    budget=budget,
                    action_timeout=limits.browser_action_timeout_seconds,
                    response_timeout=limits.response_timeout_seconds,
                    on_progress=on_progress,
                )
                if not await _close_discovery(
                    channel.close(),
                    deadline=budget.deadline - cleanup_reserve,
                    close_timeout_seconds=limits.close_timeout_seconds,
                ):
                    raise ChannelError("discover cleanup", "browser cleanup did not complete")
                channel = None
            origin = urlsplit(str(target_url))
            guide = WhatsAppChannelGuide(
                channel=config,
                validation=WhatsAppGuideValidation(
                    status="verified",
                    verified_at=datetime.now(UTC),
                    target_origin=f"{origin.scheme}://{origin.netloc}",
                    observed_behaviors=["open", "send", "receive", "completion"],
                    replay_verified=True,
                    server_state_reset=False,
                ),
            )
            sanitized_channel_guide(
                guide, draft.privacy, resolved_secrets=secrets, sensitive_context=context
            )
            return guide
        except BaseException:
            failed = True
            raise
        finally:
            cleanup_ok = True
            try:
                if channel is not None:
                    cleanup_ok = await _close_discovery(
                        channel.close(),
                        deadline=budget.deadline,
                        close_timeout_seconds=limits.close_timeout_seconds,
                        slots=2,
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

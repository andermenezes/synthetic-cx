"""Explicit environment-backed preparation of secret runtime settings."""

from __future__ import annotations

import os
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version

from syntheticcx.config.models import (
    OpenAICompatibleModelConfig,
    ProjectConfig,
    ResolvedModelSettings,
    StrictModel,
)
from syntheticcx.errors import ConfigurationError


def resolve_model_settings(
    config: ProjectConfig, environment: Mapping[str, str] | None = None, *, evaluation: bool = False
) -> ResolvedModelSettings:
    """Resolve only the configured provider credential before adapter construction."""
    source = os.environ if environment is None else environment
    model = config.effective_evaluation_model if evaluation else config.model
    credential = source.get(model.credential_environment_variable)
    if not credential:
        raise ConfigurationError(
            "resolve credentials",
            f"environment variable {model.credential_environment_variable} is required",
        )
    return ResolvedModelSettings(provider=model.provider, model=model.model, credential=credential)


class PreparedSettings(StrictModel):
    """Secret-aware settings for both model roles, separate from YAML evidence."""

    persona: ResolvedModelSettings
    evaluation: ResolvedModelSettings


def prepare_settings(
    config: ProjectConfig, environment: Mapping[str, str] | None = None
) -> PreparedSettings:
    """Validate local prerequisites without importing adapters or probing endpoints.

    Phase 6 will verify declared capabilities through bounded network preflight.
    """
    for role, model in [
        ("model", config.model),
        ("evaluation_model", config.effective_evaluation_model),
    ]:
        if isinstance(model, OpenAICompatibleModelConfig):
            if not model.capabilities.chat_completions:
                raise ConfigurationError(
                    "prepare configuration", f"{role}.capabilities.chat_completions is required"
                )
    semantic = any(
        criterion.evaluator_type == "semantic_goal_completion"
        for criterion in config.scenario.success_criteria
    )
    evaluator = config.effective_evaluation_model
    if semantic and isinstance(evaluator, OpenAICompatibleModelConfig):
        if not evaluator.capabilities.structured_output:
            raise ConfigurationError(
                "prepare configuration",
                "evaluation_model.capabilities.structured_output is required",
            )
    if config.telemetry.remote_export:
        if not config.telemetry.enabled:
            raise ConfigurationError(
                "prepare configuration", "telemetry.remote_export requires telemetry.enabled"
            )
        try:
            version("opentelemetry-exporter-otlp")
        except PackageNotFoundError:
            raise ConfigurationError(
                "prepare configuration", "telemetry.remote_export requires the telemetry extra"
            ) from None
    return PreparedSettings(
        persona=resolve_model_settings(config, environment),
        evaluation=resolve_model_settings(config, environment, evaluation=True),
    )

"""Explicit environment-backed preparation of secret runtime settings."""

from __future__ import annotations

import os
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from syntheticcx.config.models import (
    BehaviorConfig,
    BehaviorRule,
    OpenAICompatibleModelConfig,
    ProjectConfig,
    ResolvedModelSettings,
    StrictModel,
)
from syntheticcx.errors import ConfigurationError

if TYPE_CHECKING:
    from syntheticcx.evaluation.registry import EvaluatorRegistration
    from syntheticcx.runtime.behavior_signals import DetectorRegistration, PreparedBehavior


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


def requires_semantic_evaluation(config: ProjectConfig) -> bool:
    """Return whether the selected criteria need an evaluation-model call."""
    return any(
        criterion.evaluator_type == "semantic_goal_completion"
        for criterion in config.scenario.success_criteria
    )


def validate_behavior(
    behavior: BehaviorConfig,
    rules: list[BehaviorRule],
    registry: Mapping[str, DetectorRegistration] | None = None,
) -> PreparedBehavior:
    """Resolve explicit registrations without constructing trusted detectors."""
    from syntheticcx.runtime.behavior_signals import prepare_behavior

    return prepare_behavior(behavior, rules, registry)


def prepare_settings(
    config: ProjectConfig,
    environment: Mapping[str, str] | None = None,
    *,
    behavior_registry: Mapping[str, DetectorRegistration] | None = None,
    evaluator_registry: Mapping[str, EvaluatorRegistration] | None = None,
) -> PreparedSettings:
    """Validate local prerequisites without importing adapters or probing endpoints."""
    validate_behavior(config.behavior, config.persona.behavioral_rules, behavior_registry)
    from syntheticcx.evaluation.registry import prepare_evaluators

    prepare_evaluators(config.scenario.success_criteria, evaluator_registry)
    semantic = requires_semantic_evaluation(config)
    for role, model in [
        ("model", config.model),
        ("evaluation_model", config.effective_evaluation_model),
    ]:
        if (
            isinstance(model, OpenAICompatibleModelConfig)
            and (role == "model" or semantic)
            and not model.capabilities.chat_completions
        ):
            raise ConfigurationError(
                "prepare configuration", f"{role}.capabilities.chat_completions is required"
            )
    persona = config.model
    if (
        isinstance(persona, OpenAICompatibleModelConfig)
        and not persona.capabilities.structured_output
    ):
        raise ConfigurationError(
            "prepare configuration", "model.capabilities.structured_output is required"
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

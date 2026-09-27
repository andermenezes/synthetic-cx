"""Pre-run, secret-free model identity and capability metadata composition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from syntheticcx.config.models import (
    OpenAICompatibleModelConfig,
    OpenAIModelConfig,
    ProjectConfig,
    ProviderCapabilities,
)
from syntheticcx.domain.events import ModelInteractionCompletedEvent, TypedTraceEvent
from syntheticcx.domain.models import ModelIdentity, RunMetadata, TokenUsage


def with_model_evidence(
    metadata: RunMetadata,
    config: ProjectConfig,
    *,
    verified_capabilities: Mapping[str, ProviderCapabilities],
    unavailable_capabilities: Mapping[str, Sequence[str]],
) -> RunMetadata:
    """Prepare a validated metadata snapshot before the storage run is created."""

    def identity(model: OpenAIModelConfig | OpenAICompatibleModelConfig) -> ModelIdentity:
        return ModelIdentity(
            provider=model.provider,
            model=model.model,
            requested_seed=model.seed,
            sampling_parameters={"requested_temperature": model.temperature},
        )

    result = RunMetadata.model_validate(
        {
            **metadata.model_dump(mode="python"),
            "persona_model": identity(config.model),
            "evaluation_model": identity(config.effective_evaluation_model),
            "verified_capabilities": dict(verified_capabilities),
            "unavailable_capabilities": {
                role: list(features) for role, features in unavailable_capabilities.items()
            },
        }
    )
    assert isinstance(result, RunMetadata)
    return result


def with_committed_interactions(
    metadata: RunMetadata, events: tuple[TypedTraceEvent, ...]
) -> RunMetadata:
    """Snapshot reported versions and usage from durable model calls only."""
    interactions = [
        event.data for event in events if isinstance(event, ModelInteractionCompletedEvent)
    ]
    if not interactions:
        return metadata

    def identity_with_version(identity: ModelIdentity) -> ModelIdentity:
        versions = {
            item.identity.model_version
            for item in interactions
            if (item.identity.provider, item.identity.model) == (identity.provider, identity.model)
            and item.identity.model_version is not None
        }
        return cast(
            ModelIdentity,
            identity.model_copy(
                update={"model_version": next(iter(versions)) if len(versions) == 1 else None}
            ),
        )

    versions = {item.identity.model_version for item in interactions if item.identity.model_version}
    if all(item.usage.unavailable_reason is None for item in interactions):
        usage = TokenUsage(
            input_tokens=sum(item.usage.input_tokens or 0 for item in interactions),
            output_tokens=sum(item.usage.output_tokens or 0 for item in interactions),
        )
    else:
        usage = TokenUsage(unavailable_reason="one or more model calls did not report usage")
    return cast(
        RunMetadata,
        metadata.model_copy(
            update={
                "usage": usage,
                "provider_model_version": next(iter(versions)) if len(versions) == 1 else None,
                "persona_model": identity_with_version(metadata.persona_model),
                "evaluation_model": identity_with_version(metadata.evaluation_model),
            }
        ),
    )

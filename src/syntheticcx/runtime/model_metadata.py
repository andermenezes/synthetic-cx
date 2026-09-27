"""Pre-run, secret-free model identity and capability metadata composition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from syntheticcx.config.models import (
    OpenAICompatibleModelConfig,
    OpenAIModelConfig,
    ProjectConfig,
    ProviderCapabilities,
)
from syntheticcx.domain.models import ModelIdentity, RunMetadata


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

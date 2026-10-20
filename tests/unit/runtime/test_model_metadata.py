"""Pre-run evidence uses model configuration and probe snapshots, never credentials."""

from pathlib import Path

from syntheticcx.config.loader import load_config
from syntheticcx.config.models import OpenAIModelConfig, ProviderCapabilities
from syntheticcx.domain.models import RunMetadata
from syntheticcx.runtime.model_metadata import with_model_evidence


def test_prepared_model_metadata_is_detached_and_secret_free() -> None:
    config = load_config(Path("examples/telecom/scenario.yaml"))
    config = config.model_copy(
        update={
            "evaluation_model": OpenAIModelConfig(
                provider="openai", model="evaluation-model", seed=123, temperature=0.25
            )
        }
    )
    metadata = RunMetadata.model_validate(
        {
            "configuration_version": "1",
            "framework_version": "0.1.0",
            "persona_model": {"provider": "fake", "model": "old"},
            "evaluation_model": {"provider": "fake", "model": "old"},
            "usage": {"unavailable_reason": "no call yet"},
            "runtime_platform": "test",
            "owner": {
                "owner_uuid": "owner",
                "hostname": "test",
                "pid": 1,
                "process_start_evidence": "start",
            },
        }
    )
    verified = {
        "persona": ProviderCapabilities(
            chat_completions=True,
            structured_output=True,
            embeddings=False,
            token_usage=False,
            model_version=False,
        )
    }
    unavailable = {"persona": ["embeddings", "token_usage"]}
    result = with_model_evidence(
        metadata,
        config,
        verified_capabilities=verified,
        unavailable_capabilities=unavailable,
    )
    unavailable["persona"].append("model_version")
    assert result.unavailable_capabilities["persona"] == ["embeddings", "token_usage"]
    assert metadata.persona_model.model == "old"
    assert result.persona_model.model == config.model.model
    assert result.evaluation_model.model == "evaluation-model"
    assert result.evaluation_model.requested_seed == 123
    assert result.evaluation_model.sampling_parameters == {"requested_temperature": 0.25}
    assert result.usage.unavailable_reason == "no call yet"
    assert "credential" not in result.model_dump_json()

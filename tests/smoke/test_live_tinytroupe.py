"""Opt-in reference interaction against the selected OpenAI model."""

import asyncio
import os
from pathlib import Path

import pytest

from syntheticcx.adapters.llm.preflight import preflight_models
from syntheticcx.adapters.persona import TinyTroupePersonaSimulator
from syntheticcx.config.loader import load_config
from syntheticcx.config.settings import prepare_settings
from syntheticcx.domain.models import (
    ConversationObservation,
    CustomerScenarioContext,
    ModelInteraction,
    PersonaAction,
    PersonaDefinition,
)


def test_live_tinytroupe_openai_reference_interaction() -> None:
    """Confirm pinned TinyTroupe action translation, not nondeterministic wording."""
    if os.getenv("SYNTHETICCX_RUN_LIVE_SMOKE") != "1":
        pytest.skip("set SYNTHETICCX_RUN_LIVE_SMOKE=1 to enable the live smoke test")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required after explicit live-smoke opt-in")
    credential = os.environ["OPENAI_API_KEY"]

    config = load_config(Path("examples/telecom/scenario.yaml"))
    assert config.model.provider == "openai" and config.model.model == "gpt-5.4-mini"
    config = config.model_copy(
        update={
            "execution": config.execution.model_copy(
                update={"model_timeout_seconds": 90, "close_timeout_seconds": 10}
            )
        }
    )
    settings = prepare_settings(config, {"OPENAI_API_KEY": credential})

    async def interact() -> None:
        capabilities = await preflight_models(config, settings)
        assert capabilities.verified_capabilities["persona"].chat_completions
        assert capabilities.verified_capabilities["persona"].structured_output
        assert not capabilities.verified_capabilities["persona"].embeddings

        simulator = TinyTroupePersonaSimulator(
            config.model, settings.persona, timeout_seconds=config.execution.model_timeout_seconds
        )
        try:
            state = await simulator.initialize(
                PersonaDefinition.model_validate(config.persona.model_dump()),
                CustomerScenarioContext.from_scenario(config.scenario),
            )
            action = await simulator.respond(
                ConversationObservation(kind="conversation_started"), state
            )
            assert isinstance(action, PersonaAction)
            if action.signal == "continue":
                assert action.message and action.message.strip()
            else:
                assert action.message is None
            interactions = simulator.take_interactions()
            assert interactions
            assert all(isinstance(interaction, ModelInteraction) for interaction in interactions)
            assert all(
                interaction.identity.provider == "openai"
                and interaction.identity.model == "gpt-5.4-mini"
                for interaction in interactions
            )
            assert all(
                (item.usage.input_tokens is not None and item.usage.output_tokens is not None)
                or item.usage.unavailable_reason is not None
                for item in interactions
            )
            assert all("prompts_not_captured" in item.capture_omissions for item in interactions)
        finally:
            await asyncio.wait_for(simulator.close(), config.execution.close_timeout_seconds)

    outer_timeout = (
        4 * config.execution.model_timeout_seconds + 2 * config.execution.close_timeout_seconds + 10
    )
    asyncio.run(asyncio.wait_for(interact(), timeout=outer_timeout))

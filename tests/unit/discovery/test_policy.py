"""Portability is forbidden in discovery, regardless of the scenario or model."""

import asyncio
from pathlib import Path

import pytest

from syntheticcx.adapters.channel.options import eligibility
from syntheticcx.discovery.policy import (
    portability_flow,
    require_discovery_probe,
    require_discovery_target,
)
from syntheticcx.discovery.run import discover_guide
from syntheticcx.discovery.whatsapp import discover_whatsapp_guide
from syntheticcx.errors import ConfigurationError


@pytest.mark.parametrize(
    "label",
    [
        "Portabilidade 📱",
        "PORTABILIDADE",
        "Portabilid\u200bade",
        "Trazer meu número pra Claro",
        "Mantenha seu número",
        "Number portability",
        "Port your number",
        "Keep my phone number",
    ],
)
def test_portability_choices_are_not_discovery_capabilities(label):
    assert portability_flow(label)
    assert eligibility(
        "Escolha uma opção", label, {"safe": True, "visible": True, "enabled": True}
    ) == ("portability_forbidden_in_discovery")
    with pytest.raises(ConfigurationError, match="forbidden in discovery"):
        require_discovery_probe(label)


def test_runtime_and_unrelated_discovery_options_are_unchanged():
    metadata = {"safe": True, "visible": True, "enabled": True}
    assert eligibility("Menu móvel", "Portabilidade", metadata, dynamic_runtime=True) is None
    assert eligibility("Escolha um dos planos", "Planos", metadata) is None
    assert eligibility("Portabilidade: mantenha seu número", "Continuar", metadata) == (
        "portability_forbidden_in_discovery"
    )
    assert eligibility("Deseja ver os planos de Portabilidade?", "Ver planos", metadata) == (
        "portability_forbidden_in_discovery"
    )
    with pytest.raises(ConfigurationError, match="forbidden in discovery"):
        require_discovery_target("https://service.example/portabilidade")
    with pytest.raises(ConfigurationError, match="forbidden in discovery"):
        require_discovery_target("https://service.example/%70ortabilidade")


@pytest.mark.parametrize("whatsapp", [False, True])
def test_explicit_portability_probe_is_rejected_before_loading_or_browser_effects(whatsapp):
    async def exercise():
        kwargs = {"probe_message": "Quero fazer portabilidade"}
        with pytest.raises(ConfigurationError, match="forbidden in discovery"):
            if whatsapp:
                await discover_whatsapp_guide(
                    Path("DOES-NOT-EXIST.yaml"),
                    "https://example.com/",
                    browser_profile=Path("DOES-NOT-EXIST"),
                    **kwargs,
                )
            else:
                await discover_guide(Path("DOES-NOT-EXIST.yaml"), "https://example.com/", **kwargs)

    asyncio.run(exercise())

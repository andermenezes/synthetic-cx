"""Canonical demo configuration uses the public dynamic interface, offline."""

import runpy
from pathlib import Path

import pytest

from syntheticcx.config.loader import load_config
from syntheticcx.config.models import InterfaceChannelConfig, ProjectConfig

EXAMPLE = Path("examples/telecom/scenario.yaml")


def test_canonical_example_has_attested_interface_without_fixed_action_controls():
    config = load_config(EXAMPLE)
    assert isinstance(config.channel, InterfaceChannelConfig)
    assert config.channel.ownership.proof == "chat_embed"
    assert config.channel.frame_locator.value == "chat-frame"
    assert config.channel.ownership.root_locator.value == "chat-widget"
    channel = config.channel.model_dump()
    assert "input_locator" not in channel
    assert "send_strategy" not in channel
    assert config.scenario.goal
    assert config.scenario.success_criteria


@pytest.mark.parametrize("persona", ["standard", "impatient"])
@pytest.mark.parametrize("variant", ["success", "repeated_question"])
def test_demo_derivation_retains_named_frame_boundary(
    monkeypatch,
    tmp_path,
    persona,
    variant,
):
    monkeypatch.syspath_prepend(str(EXAMPLE.parent.resolve()))
    demo = runpy.run_path(str(EXAMPLE.with_name("run_demo.py")))
    document = demo["configured_example"](
        "http://127.0.0.1:8123",
        variant=variant,
        persona=persona,
        run_root=tmp_path,
    )
    config = ProjectConfig.model_validate(document)
    assert str(config.channel.url) == f"http://127.0.0.1:8123/?variant={variant}&view=frame"
    assert Path(config.storage.run_root) == tmp_path
    assert config.channel.ownership == load_config(EXAMPLE).channel.ownership
    if persona == "impatient":
        assert config.persona.initial_state.willingness_to_continue == 0.3
        assert config.persona.behavioral_rules[0].abandonment_threshold == 0.1
    else:
        assert config.persona.initial_state.willingness_to_continue == 0.9

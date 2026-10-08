"""The public interface guide contains no executable discovery graph."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from syntheticcx.config.loader import load_config, load_config_with_source_hash
from syntheticcx.config.models import (
    AnyChannelGuide,
    InterfaceChannelConfig,
    InterfaceChannelGuide,
    PrivacyConfig,
    StatefulChannelConfig,
)
from syntheticcx.config.sanitize import sanitized_channel_guide
from syntheticcx.discovery import publish
from syntheticcx.discovery.budget import DiscoveryBudget
from syntheticcx.discovery.contracts import LearnResult
from syntheticcx.discovery.publish import publish_guide
from syntheticcx.discovery.replay import replay
from syntheticcx.domain.models import CompletedResponse, Message, SelectionReceipt
from syntheticcx.errors import ChannelError, ConfigurationError


def document():
    return {
        "version": "1",
        "channel": {
            "type": "webchat_playwright_dynamic",
            "url": "https://chat.example/support",
            "ownership": {
                "root_locator": {"strategy": "test_id", "value": "chat"},
                "proof": "chat_dialog",
            },
            "assistant_message_locator": {"strategy": "test_id", "value": "assistant"},
            "message_identity": {"strategy": "dom_node"},
            "response_completion": {"strategy": "quiet_interval", "quiet_interval_seconds": 0.1},
        },
        "validation": {
            "status": "verified",
            "verified_at": "2026-10-05T12:00:00Z",
            "target_origin": "https://chat.example",
            "observed_behaviors": ["open", "select_option", "receive", "completion"],
        },
    }


def draft_file(tmp_path):
    draft = yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())
    draft.pop("channel")
    source = tmp_path / "draft.yaml"
    source.write_text(yaml.safe_dump(draft))
    return source


def test_dynamic_publication_roundtrip_and_no_clobber(tmp_path):
    guide = InterfaceChannelGuide.model_validate(document())
    source = draft_file(tmp_path)
    output = tmp_path / "chat.yaml"
    completed = publish_guide(source, output, guide, in_place=False)
    loaded = load_config_with_source_hash(completed)
    assert isinstance(loaded.config.channel, InterfaceChannelConfig)
    assert loaded.config.channel == guide.channel
    assert loaded.guide_source_hash is not None
    saved = output.read_bytes()
    with pytest.raises(ConfigurationError, match="already exists"):
        publish_guide(source, output, guide, in_place=False)
    assert output.read_bytes() == saved


@pytest.mark.parametrize(
    "version,kind", [("1", "webchat_playwright"), ("2", "webchat_playwright_stateful")]
)
def test_old_guides_rejected_without_conversion(tmp_path, version, kind):
    old = document()
    old["version"] = version
    old["channel"]["type"] = kind
    source = draft_file(tmp_path)
    reference = yaml.safe_load(source.read_text())
    reference["channel_guide"] = "old.yaml"
    source.write_text(yaml.safe_dump(reference))
    (tmp_path / "old.yaml").write_text(yaml.safe_dump(old))
    with pytest.raises(ConfigurationError):
        load_config(source)
    with pytest.raises(ValidationError):
        AnyChannelGuide.model_validate(old)


@pytest.mark.parametrize(
    "mutation",
    [
        "graph",
        "origin",
        "credential",
        "evidence",
        "duplicate",
        "bootstrap",
        "exchange",
        "root",
        "identity",
    ],
)
def test_invalid_boundaries_fail_closed(mutation):
    value = document()
    channel = value["channel"]
    if mutation == "graph":
        channel["states"] = []
    elif mutation == "origin":
        value["validation"]["target_origin"] = "https://other.example"
    elif mutation == "credential":
        channel["url"] = "https://CANARY:SECRET@chat.example/support"
    elif mutation == "evidence":
        value["validation"]["observed_behaviors"].remove("completion")
    elif mutation == "duplicate":
        value["validation"]["observed_behaviors"].append("open")
    elif mutation == "bootstrap":
        channel["bootstrap_input_locator"] = {"strategy": "label", "value": "Message"}
    elif mutation == "exchange":
        channel["reader_evidence"] = "observed_exchange"
    elif mutation == "root":
        channel["ownership"]["root_locator"] = {"strategy": "css", "value": "body"}
    elif mutation == "identity":
        channel["message_identity"] = {"strategy": "actor_sequence_content_hash"}
    with pytest.raises(ValidationError):
        InterfaceChannelGuide.model_validate(value)


@pytest.mark.parametrize("selector", [r"#CANARY\ TOKEN", r"#CANARY\20 TOKEN"])
def test_privacy_rejects_escaped_selectors_before_writes(tmp_path, selector):
    value = document()
    value["channel"]["ownership"]["root_locator"] = {"strategy": "css", "value": selector}
    guide = InterfaceChannelGuide.model_validate(value)
    output = tmp_path / "chat.yaml"
    with pytest.raises(ConfigurationError, match="sensitive data") as caught:
        publish_guide(
            draft_file(tmp_path), output, guide, in_place=False, resolved_secrets=["CANARY TOKEN"]
        )
    assert "CANARY" not in str(caught.value)
    assert not output.exists()
    assert not (tmp_path / "draft.completed.yaml").exists()


def test_publication_revalidates_and_preserves_selectors():
    guide = InterfaceChannelGuide.model_validate(document())
    assert sanitized_channel_guide(guide, PrivacyConfig()) == deepcopy(
        guide.model_dump(mode="json", exclude_none=True)
    )
    unsafe = guide.model_copy(update={"version": "2"})
    with pytest.raises(ConfigurationError, match="invalid channel guide contract"):
        sanitized_channel_guide(unsafe, PrivacyConfig())


def test_publication_rolls_back_when_ordinary_loader_rejects(tmp_path, monkeypatch):
    source = draft_file(tmp_path)
    original = source.read_bytes()
    output = tmp_path / "chat.yaml"

    def reject(_path):
        raise ConfigurationError("validate configuration", "controlled failure")

    monkeypatch.setattr(publish, "load_config", reject)
    with pytest.raises(ConfigurationError, match="controlled failure"):
        publish_guide(
            source, output, InterfaceChannelGuide.model_validate(document()), in_place=True
        )
    assert source.read_bytes() == original
    assert not output.exists()
    assert not source.with_name("draft.yaml.bak").exists()


def test_schema_has_only_dynamic_public_contract():
    schema = json.loads(Path("schemas/channel-guide.schema.json").read_text())
    validator = Draft202012Validator(schema)
    assert validator.is_valid(document())
    old = document()
    old["channel"]["type"] = "webchat_playwright_stateful"
    assert not validator.is_valid(old)
    assert "ChatState" not in schema.get("$defs", {})
    project = json.loads(Path("schemas/project.schema.json").read_text())
    assert "StatefulChannelConfig" not in project["$defs"]


def replay_setup(composer=False):
    boundary = document()["channel"]
    boundary["type"] = "webchat_playwright_stateful"
    state = {
        "id": "initial",
        "kind": "menu",
        "option_locator": {"strategy": "role", "value": "button"},
        "option_labels": ["Help"],
    }
    transition = {
        "id": "step",
        "from_state": "initial",
        "to_state": "initial",
        "action_kind": "select_option",
        "option_label": "Help",
    }
    if composer:
        state = {
            "id": "initial",
            "kind": "composer",
            "input_locator": {"strategy": "label", "value": "Message"},
            "send_strategy": {"type": "press_enter"},
        }
        transition = {
            "id": "step",
            "from_state": "initial",
            "to_state": "initial",
            "action_kind": "probe",
            "probe_id": "explicit",
        }
    config = StatefulChannelConfig.model_validate(
        {
            **boundary,
            "initial_state_id": "initial",
            "states": [state],
            "transitions": [transition],
            "verification_recipe": [{"transition_id": "step"}],
        }
    )
    now = datetime.now(UTC)
    response = CompletedResponse(
        messages=[
            Message(id="reply", sequence=1, actor="assistant", content="Help", observed_at=now)
        ],
        started_at=now,
        completed_at=now,
        completion_strategy="quiet_interval",
    )
    receipt = SelectionReceipt(
        message_id="sent",
        submitted_at=now,
        selected_option_id="fresh",
        selected_label="Help",
        source_message_id="reply",
    )
    broker = SimpleNamespace(
        current_state=config.states[0],
        observe_initial=AsyncMock(return_value=response),
        observe=AsyncMock(
            return_value={
                "revision": "fresh",
                "ownership_bound": True,
                "reader_bound": True,
                "composer_id": "input",
                "available_options": [{"id": "fresh", "label": "Help"}],
            }
        ),
        select_option=AsyncMock(return_value=receipt),
        submit_probe=AsyncMock(return_value=receipt),
        receive=AsyncMock(return_value=response),
    )
    return broker, LearnResult(config=config, probes={"explicit": "PRIVATE PROBE"})


@pytest.mark.parametrize("composer", [False, True])
def test_exact_private_replay_exports_boundary_only(composer):
    broker, learned = replay_setup(composer)
    guide = asyncio.run(replay(broker, learned, budget=DiscoveryBudget.create(30)))
    assert isinstance(guide, InterfaceChannelGuide)
    assert (guide.channel.bootstrap_input_locator is not None) == composer
    assert (guide.channel.bootstrap_send_strategy is not None) == composer
    serialized = guide.model_dump_json()
    for private in [
        "states",
        "transitions",
        "verification_recipe",
        "PRIVATE PROBE",
        "verified_state_ids",
        "verified_transition_ids",
        "option_labels",
    ]:
        assert private not in serialized
    assert learned.config.states  # The private graph remains intact.


def test_replay_still_fails_closed_on_private_state_mismatch():
    broker, learned = replay_setup()
    broker.current_state = None
    with pytest.raises(ChannelError, match="fresh state"):
        asyncio.run(replay(broker, learned, budget=DiscoveryBudget.create(30)))
    broker.select_option.assert_not_awaited()


def test_three_step_private_recipe_is_replayed_but_not_exported():
    broker, learned = replay_setup()
    private = learned.config.model_dump()
    private["transitions"] = [
        {**private["transitions"][0], "id": f"step_{index}"} for index in range(3)
    ]
    private["verification_recipe"] = [{"transition_id": f"step_{index}"} for index in range(3)]
    learned.config = StatefulChannelConfig.model_validate(private)
    budget = DiscoveryBudget.create(30)
    guide = asyncio.run(replay(broker, learned, budget=budget))
    assert budget.actions == broker.select_option.await_count == 3
    assert "step_" not in guide.model_dump_json()


def test_menu_first_replay_does_not_publish_later_composer_as_bootstrap():
    broker, learned = replay_setup()
    private = learned.config.model_dump()
    private["states"].append(
        {
            "id": "composer",
            "kind": "composer",
            "input_locator": {"strategy": "label", "value": "Message"},
            "send_strategy": {"type": "press_enter"},
        }
    )
    private["transitions"][0]["to_state"] = "composer"
    learned.config = StatefulChannelConfig.model_validate(private)
    response = broker.receive.return_value

    async def receive():
        broker.current_state = learned.config.states[1]
        return response

    broker.receive.side_effect = receive
    guide = asyncio.run(replay(broker, learned, budget=DiscoveryBudget.create(30)))
    assert guide.channel.bootstrap_input_locator is None
    assert guide.channel.bootstrap_send_strategy is None

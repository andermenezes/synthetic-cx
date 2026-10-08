"""Stateful contracts survive ordinary publication and reject invented evidence."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator
from pydantic import AnyHttpUrl, TypeAdapter, ValidationError

from syntheticcx.config.loader import load_config, load_config_with_source_hash
from syntheticcx.config.models import (
    AnyChannelGuide,
    ChannelGuide,
    InterfaceChannelConfig,
    InterfaceChannelGuide,
    PrivacyConfig,
    StatefulChannelGuide,
)
from syntheticcx.config.sanitize import sanitized_channel_guide
from syntheticcx.discovery import publish
from syntheticcx.errors import ConfigurationError


def guide_document() -> dict[str, Any]:
    locator = {"strategy": "test_id", "value": "assistant-message"}
    return {
        "version": "2",
        "channel": {
            "type": "webchat_playwright_stateful",
            "url": "https://chat.example/support",
            "ownership": {
                "root_locator": {"strategy": "role", "value": "dialog", "name": "Chat"},
                "proof": "chat_dialog",
            },
            "assistant_message_locator": locator,
            "message_identity": {"strategy": "dom_node"},
            "response_completion": {"strategy": "quiet_interval", "quiet_interval_seconds": 0.1},
            "initial_state_id": "menu",
            "states": [
                {
                    "id": "menu",
                    "kind": "menu",
                    "option_locator": {"strategy": "role", "value": "button"},
                    "option_labels": ["Help"],
                },
                {
                    "id": "composer",
                    "kind": "composer",
                    "input_locator": {"strategy": "label", "value": "Message"},
                    "send_strategy": {"type": "press_enter"},
                },
            ],
            "transitions": [
                {
                    "id": "help",
                    "from_state": "menu",
                    "to_state": "composer",
                    "action_kind": "select_option",
                    "option_label": "Help",
                }
            ],
            "verification_recipe": [{"transition_id": "help"}],
        },
        "validation": {
            "status": "verified",
            "verified_at": "2026-10-04T12:00:00Z",
            "target_origin": "https://chat.example",
            "observed_behaviors": ["open", "select_option", "receive", "completion"],
            "verified_state_ids": ["menu", "composer"],
            "verified_transition_ids": ["help"],
        },
    }


def draft_file(tmp_path: Path) -> Path:
    document = yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())
    document.pop("channel")
    path = tmp_path / "draft.yaml"
    path.write_text(yaml.safe_dump(document))
    return path


def interface_document() -> dict[str, Any]:
    """Public boundary fixture; the learned graph remains a separate private sample."""
    private = guide_document()
    channel = private["channel"]
    for field in ("initial_state_id", "states", "transitions", "verification_recipe"):
        channel.pop(field)
    channel["type"] = "webchat_playwright_dynamic"
    private["version"] = "1"
    private["validation"].pop("verified_state_ids")
    private["validation"].pop("verified_transition_ids")
    return private


def test_stateful_publication_roundtrip_and_no_clobber(tmp_path: Path) -> None:
    guide = InterfaceChannelGuide.model_validate(interface_document())
    source = draft_file(tmp_path)
    output = tmp_path / "guides" / "chat.yaml"
    completed = publish.publish_guide(source, output, guide, in_place=False)
    loaded = load_config_with_source_hash(completed)
    assert isinstance(loaded.config.channel, InterfaceChannelConfig)
    assert loaded.config.channel == guide.channel
    assert loaded.guide_source_hash is not None
    assert "probe_message" not in output.read_text()
    saved = output.read_bytes()
    with pytest.raises(ConfigurationError, match="already exists"):
        publish.publish_guide(source, output, guide, in_place=False)
    assert output.read_bytes() == saved


def test_existing_stateful_guides_default_to_marked_readers():
    guide = StatefulChannelGuide.model_validate(guide_document())
    assert guide.channel.reader_evidence == "marked"


def test_exchange_readers_cannot_be_declared_without_echo_and_first_probe():
    document = guide_document()
    document["channel"]["reader_evidence"] = "observed_exchange"
    with pytest.raises(ValidationError, match="separated customer locator"):
        StatefulChannelGuide.model_validate(document)
    document["channel"]["customer_message_locator"] = {"strategy": "css", "value": ".right > p"}
    with pytest.raises(ValidationError, match="first composer probe"):
        StatefulChannelGuide.model_validate(document)


def test_only_interface_version_is_publicly_supported() -> None:
    adapter = TypeAdapter(AnyChannelGuide)
    v2 = guide_document()
    original = yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())
    legacy = original["channel"]
    legacy["type"] = "webchat_playwright"
    legacy.pop("ownership")
    legacy["input_locator"] = {"strategy": "test_id", "value": "chat-input"}
    legacy["send_strategy"] = {"type": "press_enter"}
    v1 = {
        "version": "1",
        "channel": original["channel"],
        "validation": {
            "status": "verified",
            "verified_at": "2026-10-04T12:00:00Z",
            "target_origin": "http://127.0.0.1:8000",
            "observed_behaviors": ["open", "send", "receive", "completion"],
        },
    }
    assert isinstance(ChannelGuide.model_validate(v1), ChannelGuide)
    assert isinstance(StatefulChannelGuide.model_validate(v2), StatefulChannelGuide)
    for old in (v1, v2):
        with pytest.raises(ValidationError):
            adapter.validate_python(old)
    schema = Draft202012Validator(adapter.json_schema())
    assert not schema.is_valid(v1) and not schema.is_valid(v2)
    assert schema.is_valid(interface_document())
    v1["validation"]["observed_behaviors"].remove("send")
    with pytest.raises(ValidationError):
        adapter.validate_python(v1)


@pytest.mark.parametrize("selector", [r"#CANARY\ TOKEN", r"#CANARY\20 TOKEN"])
def test_publication_rejects_reversibly_escaped_locator_credentials(selector):
    document = interface_document()
    document["channel"]["ownership"]["root_locator"] = {"strategy": "css", "value": selector}
    guide = InterfaceChannelGuide.model_validate(document)
    with pytest.raises(ConfigurationError, match="sensitive data") as error:
        sanitized_channel_guide(guide, PrivacyConfig(), resolved_secrets=["CANARY TOKEN"])
    assert "CANARY" not in str(error.value)


@pytest.mark.parametrize(
    "mutation",
    [
        "initial",
        "endpoint",
        "recipe",
        "chain",
        "uncovered",
        "state_evidence",
        "transition_evidence",
        "send_claim",
        "missing_completion",
        "origin",
        "paired",
        "kind",
        "duplicate_labels",
        "descriptor",
        "unknown_option",
        "raw_probe",
        "handle",
        "root",
        "missing_identity",
        "too_many_steps",
    ],
)
def test_invalid_contracts_fail_closed(mutation: str) -> None:
    document = guide_document()
    channel = document["channel"]
    validation = document["validation"]
    transition = channel["transitions"][0]
    if mutation == "initial":
        channel["initial_state_id"] = "absent"
    elif mutation == "endpoint":
        transition["to_state"] = "absent"
    elif mutation == "recipe":
        channel["verification_recipe"][0]["transition_id"] = "absent"
    elif mutation == "chain":
        channel["initial_state_id"] = "composer"
    elif mutation == "uncovered":
        channel["states"].append({"id": "unused", "kind": "terminal"})
    elif mutation == "state_evidence":
        validation["verified_state_ids"] = ["menu"]
    elif mutation == "transition_evidence":
        validation["verified_transition_ids"] = ["invented"]
    elif mutation == "send_claim":
        validation["observed_behaviors"].append("send")
    elif mutation == "missing_completion":
        validation["observed_behaviors"].remove("completion")
    elif mutation == "origin":
        validation["target_origin"] = "https://other.example"
    elif mutation == "paired":
        channel["states"][1].pop("send_strategy")
    elif mutation == "kind":
        channel["states"][0]["kind"] = "composer"
    elif mutation == "duplicate_labels":
        channel["states"][0]["option_labels"].append("  HELP  ")
    elif mutation == "descriptor":
        transition["probe_id"] = "explicit"
    elif mutation == "unknown_option":
        transition["option_label"] = "Invented"
    elif mutation == "raw_probe":
        transition["probe_message"] = "private raw text"
    elif mutation == "handle":
        channel["states"][0]["target_id"] = "ephemeral"
    elif mutation == "root":
        channel["ownership"]["root_locator"] = {"strategy": "css", "value": "body"}
    elif mutation == "missing_identity":
        channel.pop("message_identity")
    elif mutation == "too_many_steps":
        channel["verification_recipe"] *= 4
    with pytest.raises(ValidationError):
        StatefulChannelGuide.model_validate(document)


def test_explicit_probe_persists_only_identifier() -> None:
    document = guide_document()
    channel = document["channel"]
    channel["transitions"].append(
        {
            "id": "probe",
            "from_state": "composer",
            "to_state": "composer",
            "action_kind": "probe",
            "probe_id": "explicit",
        }
    )
    channel["verification_recipe"].append({"transition_id": "probe"})
    document["validation"]["verified_transition_ids"].append("probe")
    document["validation"]["observed_behaviors"].append("send")
    guide = StatefulChannelGuide.model_validate(document)
    assert guide.channel.transitions[1].probe_id == "explicit"
    assert guide.model_dump(mode="json") == StatefulChannelGuide.model_validate(
        guide.model_dump(mode="json")
    ).model_dump(mode="json")


def test_three_step_menu_chain_covers_four_states() -> None:
    document = guide_document()
    channel = document["channel"]
    second = deepcopy(channel["states"][0])
    second["id"] = "second_menu"
    second["option_labels"] = ["Support"]
    channel["states"].insert(1, second)
    channel["states"].append({"id": "done", "kind": "terminal"})
    channel["transitions"][0]["to_state"] = "second_menu"
    channel["transitions"].extend(
        [
            {
                "id": "support",
                "from_state": "second_menu",
                "to_state": "composer",
                "action_kind": "select_option",
                "option_label": "Support",
            },
            {
                "id": "probe",
                "from_state": "composer",
                "to_state": "done",
                "action_kind": "probe",
                "probe_id": "first",
            },
        ]
    )
    channel["verification_recipe"].extend(
        [
            {"transition_id": "support"},
            {"transition_id": "probe"},
        ]
    )
    document["validation"]["verified_state_ids"] = ["menu", "second_menu", "composer", "done"]
    document["validation"]["verified_transition_ids"] = ["help", "support", "probe"]
    document["validation"]["observed_behaviors"].append("send")
    assert len(StatefulChannelGuide.model_validate(document).channel.states) == 4
    channel["states"].append({"id": "extra", "kind": "terminal"})
    with pytest.raises(ValidationError):
        StatefulChannelGuide.model_validate(document)


def test_unknown_version_diagnostic_never_contains_supplied_value(tmp_path: Path) -> None:
    source = draft_file(tmp_path)
    document = yaml.safe_load(source.read_text())
    document["channel_guide"] = "chat.yaml"
    source.write_text(yaml.safe_dump(document))
    guide = guide_document()
    guide["version"] = "private-secret-version"
    (tmp_path / "chat.yaml").write_text(yaml.safe_dump(guide))
    with pytest.raises(ConfigurationError) as caught:
        load_config(source)
    assert "version:" in str(caught.value)
    assert "private-secret-version" not in str(caught.value)


def test_sensitive_selectors_rejected_not_silently_changed(tmp_path: Path) -> None:
    document = interface_document()
    document["channel"]["bootstrap_input_locator"] = {
        "strategy": "label",
        "value": "secret-selector",
    }
    document["channel"]["bootstrap_send_strategy"] = {"type": "press_enter"}
    guide = InterfaceChannelGuide.model_validate(document)
    with pytest.raises(ConfigurationError, match="sensitive data") as caught:
        publish.publish_guide(
            draft_file(tmp_path),
            tmp_path / "guide.yaml",
            guide,
            in_place=False,
            resolved_secrets=["secret-selector"],
        )
    assert "secret-selector" not in str(caught.value)
    assert not (tmp_path / "guide.yaml").exists()
    assert guide.channel.bootstrap_input_locator is not None
    assert guide.channel.bootstrap_input_locator.value == "secret-selector"


def test_allowlist_revalidates_constructed_contract() -> None:
    guide = InterfaceChannelGuide.model_validate(interface_document())
    unsafe = guide.model_copy(
        update={"channel": guide.channel.model_copy(update={"type": "unsafe"})}
    )
    with pytest.raises(ConfigurationError, match="invalid channel guide contract"):
        sanitized_channel_guide(unsafe, PrivacyConfig())
    assert sanitized_channel_guide(guide, PrivacyConfig()) == deepcopy(
        guide.model_dump(mode="json", exclude_none=True)
    )


def test_loader_rejection_rolls_back_stateful_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = draft_file(tmp_path)
    original = source.read_bytes()
    output = tmp_path / "guide.yaml"

    def fail_loader(_path: Path) -> None:
        raise ConfigurationError("validate configuration", "controlled failure")

    monkeypatch.setattr(publish, "load_config", fail_loader)
    with pytest.raises(ConfigurationError):
        publish.publish_guide(
            source,
            output,
            InterfaceChannelGuide.model_validate(interface_document()),
            in_place=True,
        )
    assert source.read_bytes() == original
    assert not output.exists()
    assert not source.with_name("draft.yaml.bak").exists()


@pytest.mark.parametrize("field", ["url", "target_origin"])
@pytest.mark.parametrize(
    "userinfo", ["CANARY-USER@", ":CANARY-PASSWORD@", "CANARY-USER:CANARY-PASSWORD@"]
)
def test_v2_userinfo_rejected_by_loader_and_publication(
    tmp_path: Path,
    field: str,
    userinfo: str,
) -> None:
    document = interface_document()
    section = "channel" if field == "url" else "validation"
    document[section][field] = f"https://{userinfo}chat.example/"
    source = draft_file(tmp_path)
    reference = yaml.safe_load(source.read_text())
    reference["channel_guide"] = "untrusted.yaml"
    run = tmp_path / "run.yaml"
    run.write_text(yaml.safe_dump(reference))
    (tmp_path / "untrusted.yaml").write_text(yaml.safe_dump(document))
    with pytest.raises(ConfigurationError) as caught:
        load_config(run)
    assert field in str(caught.value)
    assert "credentials" in str(caught.value)
    assert "CANARY-USER" not in str(caught.value)
    assert "CANARY-PASSWORD" not in str(caught.value)

    # Even an unchecked model_copy must not bypass the publication boundary.
    guide = InterfaceChannelGuide.model_validate(interface_document())
    component = getattr(guide, section)
    unsafe = guide.model_copy(
        update={
            section: component.model_copy(update={field: AnyHttpUrl(document[section][field])}),
        }
    )
    output = tmp_path / "published.yaml"
    with pytest.raises(ConfigurationError) as caught:
        publish.publish_guide(source, output, unsafe, in_place=False)
    assert str(caught.value) == "publish discovery: invalid channel guide contract"
    assert not output.exists()
    assert not source.with_name("draft.completed.yaml").exists()


def test_legacy_userinfo_validation_is_unchanged() -> None:
    document = yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())
    channel = document["channel"]
    channel["type"] = "webchat_playwright"
    channel.pop("ownership")
    channel["input_locator"] = {"strategy": "test_id", "value": "chat-input"}
    channel["send_strategy"] = {"type": "press_enter"}
    channel["url"] = "https://legacy-user:legacy-password@chat.example/"
    channel["message_identity"] = {"strategy": "dom_node"}
    guide = ChannelGuide.model_validate(
        {
            "version": "1",
            "channel": channel,
            "validation": {
                "status": "verified",
                "verified_at": "2026-10-04T12:00:00Z",
                "target_origin": "https://legacy-user:legacy-password@chat.example/",
                "observed_behaviors": ["open", "send", "receive", "completion"],
            },
        }
    )
    assert guide.channel.url.username == "legacy-user"

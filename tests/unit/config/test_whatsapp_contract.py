"""WhatsApp guide verification boundaries and ordinary configuration loading."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import TypeAdapter, ValidationError

from syntheticcx.config.loader import load_config_with_source_hash
from syntheticcx.config.models import (
    AnyChannelGuide,
    WebsiteNavigationStep,
    WhatsAppChannelConfig,
    WhatsAppChannelGuide,
)
from syntheticcx.discovery.publish import publish_guide
from syntheticcx.errors import ConfigurationError


def guide_document():
    return {
        "channel": {
            "type": "whatsapp_web_playwright",
            "url": "https://telecom.example/support",
            "navigation_steps": [
                {
                    "action": "click",
                    "locator": {"strategy": "role", "value": "link", "name": "WhatsApp support"},
                }
            ],
        },
        "validation": {
            "status": "verified",
            "verified_at": "2026-10-10T12:00:00Z",
            "target_origin": "https://telecom.example",
            "observed_behaviors": ["open", "send", "receive", "completion"],
            "replay_verified": True,
            "server_state_reset": False,
        },
    }


def test_whatsapp_guide_publication_loads_through_normal_journey(tmp_path):
    draft = yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())
    draft.pop("channel")
    source = tmp_path / "draft.yaml"
    source.write_text(yaml.safe_dump(draft))
    guide = TypeAdapter(AnyChannelGuide).validate_python(guide_document())
    assert isinstance(guide, WhatsAppChannelGuide)
    output = tmp_path / "whatsapp.yaml"
    completed = publish_guide(source, output, guide, in_place=False)
    loaded = load_config_with_source_hash(completed)
    assert isinstance(loaded.config.channel, WhatsAppChannelConfig)
    assert loaded.config.channel.navigation_steps[0].locator.name == "WhatsApp support"
    assert loaded.guide_source_hash is not None
    saved = output.read_bytes()
    with pytest.raises(ConfigurationError):
        publish_guide(source, output, guide, in_place=False)
    assert saved == output.read_bytes()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["channel"].update(navigation_steps=[]),
        lambda d: d["validation"].update(observed_behaviors=["open", "send", "receive"]),
        lambda d: d["validation"].update(replay_verified=False),
        lambda d: d["validation"].update(server_state_reset=True),
        lambda d: d["validation"].update(target_origin="https://other.example"),
        lambda d: d["validation"].update(target_origin="https://telecom.example/private"),
        lambda d: d["validation"].update(target_origin="https://user:secret@telecom.example"),
    ],
)
def test_unverified_or_misbound_guide_cannot_be_loaded(mutation):
    document = deepcopy(guide_document())
    mutation(document)
    with pytest.raises(ValidationError):
        TypeAdapter(AnyChannelGuide).validate_python(document)


@pytest.mark.parametrize("interval", [0, -1, 11, float("nan"), float("inf")])
def test_reply_settling_policy_is_bounded(interval):
    document = guide_document()["channel"]
    document["quiet_interval_seconds"] = interval
    with pytest.raises(ValidationError):
        WhatsAppChannelConfig.model_validate(document)


@pytest.mark.parametrize(
    "url",
    [
        "https://telecom.example/support?session=PRIVATE_TOKEN",
        "https://telecom.example/support#PRIVATE_TOKEN",
        "https://telecom.example/support/%31%35%35%35%31%32%33%34%35%36%37",
        "https://telecom.example/support/%2531%2535%2535%2535%2531%2532%2533%2534%2535%2536%2537",
    ],
)
def test_protected_website_urls_are_not_portable_guide_data(url):
    document = guide_document()
    document["channel"]["url"] = url
    with pytest.raises(ValidationError):
        WhatsAppChannelGuide.model_validate(document)


@pytest.mark.parametrize("action", ["click", "hover", "follow_link"])
def test_targeted_navigation_requires_exact_locator(action):
    with pytest.raises(ValidationError):
        WebsiteNavigationStep.model_validate({"action": action})
    step = WebsiteNavigationStep.model_validate(
        {"action": action, "locator": {"strategy": "css", "value": "#menu"}}
    )
    assert step.locator.value == "#menu"


@pytest.mark.parametrize("action", ["scroll_down", "scroll_up", "back", "wait"])
def test_page_navigation_forbids_locator(action):
    assert WebsiteNavigationStep.model_validate({"action": action}).locator is None
    with pytest.raises(ValidationError):
        WebsiteNavigationStep.model_validate(
            {"action": action, "locator": {"strategy": "css", "value": "#menu"}}
        )


@pytest.mark.parametrize("legacy", ["entry_locator", "entry_steps"])
def test_legacy_whatsapp_route_fields_are_rejected(legacy):
    document = guide_document()["channel"]
    document[legacy] = None if legacy == "entry_locator" else []
    with pytest.raises(ValidationError):
        WhatsAppChannelConfig.model_validate(document)


def test_navigation_route_is_typed_and_bounded():
    document = guide_document()["channel"]
    document["navigation_steps"] = [{"strategy": "css", "value": "#menu"}]
    with pytest.raises(ValidationError):
        WhatsAppChannelConfig.model_validate(document)
    document["navigation_steps"] = [{"action": "wait"}] * 128
    assert len(WhatsAppChannelConfig.model_validate(document).navigation_steps) == 128
    document["navigation_steps"].append({"action": "wait"})
    with pytest.raises(ValidationError):
        WhatsAppChannelConfig.model_validate(document)

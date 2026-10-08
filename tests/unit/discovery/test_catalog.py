"""Owned candidates and strict proposal contracts, including legacy compatibility."""

import asyncio
import json
from pathlib import Path

import pytest
import yaml

from syntheticcx.config.models import ChannelConfig, PrivacyConfig
from syntheticcx.discovery.calibrate import (
    _fresh_echo_observed,
    _reading_catalog,
    _reading_proposal,
    _validate_messages,
)
from syntheticcx.discovery.catalog import (
    LocatorCatalog,
    css_string,
    interaction_catalog,
    node_locators,
)
from syntheticcx.discovery.explore import explore
from syntheticcx.discovery.proposals import INTERACTION_SCHEMA, READING_SCHEMA, parse_reading
from syntheticcx.errors import ChannelError, ModelError
from syntheticcx.privacy import Redactor


def config():
    return ChannelConfig.model_validate(
        yaml.safe_load(Path("tests/fixtures/discovery-sample.yaml").read_text())["channel"]
    )


def test_quoted_css_strings_escape_quotes_backslashes_and_controls():
    assert css_string('a"b\\c\n\x01') == '"a\\"b\\\\c\\a \\1 "'
    locators = node_locators({"tag": "div", "attributes": {"class": "flex:row w-[40px]"}})
    assert 'div[class~="flex:row"][class~="w-[40px]"]' in [x.value for x in locators]
    assert all(".flex:row" not in x.value for x in locators)


def test_catalog_limits_revision_kind_and_private_selector_boundary():
    catalog = LocatorCatalog()
    for kind in ("input", "customer", "assistant", "option", "scope", "identity"):
        for index in range(20):
            catalog.add(kind, f"PRIVATE-{kind}-{index}", f"Observed {kind} structure {index}")
    assert len(catalog.items) == 64
    assert sum(item.kind == "input" for item in catalog.items.values()) == 12
    assert "PRIVATE" not in json.dumps(catalog.public())
    with pytest.raises(ValueError, match="revision"):
        catalog.resolve("stale", "input-1", "input")
    with pytest.raises(ValueError, match="wrong-kind"):
        catalog.resolve(catalog.revision, "input-1", "option")


def test_initial_catalog_preserves_chat_controls_on_large_pages():
    snapshot = "\n".join(f'- button "Unrelated {index}" [ref=e{index}]' for index in range(20))
    snapshot += '\n- button "Support chat" [ref=e21]'
    snapshot += "\n" + "\n".join(f'- textbox "Page field {index}"' for index in range(20))
    inspection = json.dumps(
        {
            "elements": [
                {
                    "tag": "textarea",
                    "editable": True,
                    "visible": True,
                    "in_dialog": True,
                    "attributes": {"aria-label": "Message"},
                }
            ]
        }
    )
    catalog = interaction_catalog(snapshot, inspection)
    assert catalog.resolve(catalog.revision, "opener-1", "opener") == ("e21", "Support chat")
    assert catalog.resolve(catalog.revision, "input-1", "input").value == "Message"


@pytest.mark.parametrize("schema", [INTERACTION_SCHEMA, READING_SCHEMA])
def test_every_schema_object_is_closed_and_all_properties_required(schema):
    def check(node):
        if isinstance(node, dict):
            assert "oneOf" not in node and "discriminator" not in node
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for value in node.values():
                check(value)
        elif isinstance(node, list):
            for value in node:
                check(value)

    check(schema)
    assert "channel_json" not in json.dumps(schema)
    assert "css" not in json.dumps(schema)


def reading_catalog():
    catalog = LocatorCatalog()
    catalog.add("customer", config().customer_message_locator, "Observed echo wrapper")
    catalog.add("assistant", config().assistant_message_locator, "Observed assistant wrapper")
    catalog.add("identity", {"strategy": "dom_node"}, "Browser nodes")
    catalog.add("completion", {"strategy": "quiet_interval", "quiet_interval_seconds": 1}, "Quiet")
    return catalog


def reading_payload(catalog):
    return dict(
        revision=catalog.revision,
        customer_id="customer-1",
        assistant_id="assistant-1",
        identity_id="identity-1",
        completion_id="completion-1",
        scope_id=None,
        option_id=None,
    )


@pytest.mark.parametrize("bad", ["revision", "unknown", "extra", "missing", "nested"])
def test_malformed_new_reading_never_falls_back_to_legacy(bad):
    catalog = reading_catalog()
    payload = reading_payload(catalog)
    if bad == "revision":
        payload["revision"] = "stale"
    if bad == "unknown":
        payload["assistant_id"] = "unknown"
    if bad == "extra":
        payload["assistant_css"] = "*"
    if bad == "missing":
        del payload["identity_id"]
    if bad == "nested":
        legacy = config().model_dump(mode="json")
        legacy["url"] = "__SYNTHETICCX_TARGET_URL__"
        payload["channel_json"] = json.dumps(legacy)
    with pytest.raises(ChannelError, match="catalog schema or IDs"):
        _reading_proposal(json.dumps(payload), config(), catalog)


def test_typed_reading_changes_no_interaction_controls_and_legacy_still_parses():
    catalog = reading_catalog()
    selected = parse_reading(json.dumps(reading_payload(catalog)), config(), catalog)
    assert selected.input_locator == config().input_locator
    assert selected.message_identity.strategy == "dom_node"
    legacy = config().model_dump(mode="json")
    legacy["url"] = "__SYNTHETICCX_TARGET_URL__"
    assert _reading_proposal(json.dumps({"channel_json": json.dumps(legacy)}), config()) == config()


def test_actual_node_evidence_allows_repeated_prose_but_not_historical_wrapper():
    channel = config().model_copy(
        update={
            "message_identity": config().message_identity.model_copy(
                update={"strategy": "dom_node", "attribute_name": None}
            )
        }
    )
    baseline = {"node_ids": ["old"], "entries": [{"text": "Hello"}]}
    evidence = {
        "text": "Hello",
        "identity": "actual",
        "node_id": "new",
        "tag": "p",
        "visible": True,
        "contains_input": False,
    }
    _validate_messages([], [evidence], baseline, "probe", channel)
    with pytest.raises(ChannelError, match="new reply"):
        _validate_messages([], [{**evidence, "node_id": "old"}], baseline, "probe", channel)


@pytest.mark.parametrize("fresh_echo", [False, True])
def test_catalog_ignores_historical_probe_greeting_and_preserves_optional_echo(fresh_echo):
    old = {
        "node_id": "old",
        "tag": "p",
        "text": "Hello",
        "attributes": {"class": "historical", "data-actor": "assistant"},
    }
    baseline = {"node_ids": ["old"], "entries": [old]}
    after = {
        "entries": [
            old,
            {
                "node_id": "reply",
                "tag": "p",
                "text": "Information reply",
                "attributes": {"class": "reply"},
            },
        ]
    }
    if fresh_echo:
        after["entries"].append(
            {"node_id": "echo", "tag": "p", "text": "Hello", "attributes": {"class": "customer"}}
        )

    class Browser:
        async def locator_matches(self, locator):
            return 1

    catalog = asyncio.run(_reading_catalog(Browser(), baseline, after, "Hello"))
    customers = [item.value for item in catalog.items.values() if item.kind == "customer"]
    assert _fresh_echo_observed(baseline, after, "Hello") == fresh_echo
    assert all("historical" not in locator.value for locator in customers)
    assert any(locator.value == ":not(*)" for locator in customers) != fresh_echo


def test_historical_selected_greeting_cannot_satisfy_observed_fresh_customer_echo():
    baseline = {"node_ids": ["old"], "entries": [{"node_id": "old", "text": "Hello"}]}
    after = {
        "entries": [{"node_id": "old", "text": "Hello"}, {"node_id": "new-echo", "text": "Hello"}]
    }
    old = {
        "node_id": "old",
        "text": "Hello",
        "identity": "old-customer",
        "visible": True,
        "tag": "p",
        "contains_input": False,
    }
    reply = {
        "node_id": "reply",
        "text": "Information reply",
        "identity": "reply",
        "visible": True,
        "tag": "p",
        "contains_input": False,
    }
    for customers in ([old], []):
        with pytest.raises(ChannelError, match="observed probe echo"):
            _validate_messages(customers, [reply], baseline, "Hello", config(), after)
    _validate_messages(
        [{**old, "node_id": "new-echo"}], [reply], baseline, "Hello", config(), after
    )


@pytest.mark.parametrize("stale", [False, True])
def test_typed_exploration_bounds_stale_decision_corrections_without_submission(stale):
    requests = []

    class Browser:
        async def navigate(self, url):
            return '- textbox "Message" [ref=e1]\n- button "Send" [ref=e2]'

        async def inspect_chat(self):
            return "{}"

        async def click(self, *args):
            pytest.fail("no exploration clicks are needed")

    class Provider:
        async def generate(self, request):
            requests.append(request)
            catalog, _ = json.JSONDecoder().raw_decode(
                request.messages[-1].content.split("Current candidate catalog:\n")[1]
            )
            payload = dict(
                action="propose",
                revision="stale" if stale else catalog["revision"],
                target_id=None,
                input_id="input-1",
                frame_id=None,
                send_id="send-1",
                send_mode="click",
            )
            return type("Response", (), {"content": json.dumps(payload)})()

    operation = explore(str(config().url), Browser(), Provider(), Redactor(PrivacyConfig()))
    if stale:
        with pytest.raises(ModelError, match="unknown catalog revision"):
            asyncio.run(operation)
    else:
        selected = asyncio.run(operation)
        assert selected.channel.input_locator.strategy == "role"
    assert len(requests) == (3 if stale else 1)
    assert requests[0].response_schema == INTERACTION_SCHEMA

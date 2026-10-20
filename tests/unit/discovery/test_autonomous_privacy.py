"""The actual learning request must not leak escaped constructor context."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from syntheticcx.config.models import PrivacyConfig
from syntheticcx.discovery.budget import DiscoveryBudget
from syntheticcx.discovery.contracts import DiscoveryDecision, LearningBroker
from syntheticcx.discovery.learn import learn
from syntheticcx.domain.models import LLMRequest
from syntheticcx.domain.protocols import LLMProvider
from syntheticcx.errors import ChannelError
from syntheticcx.privacy import Redactor


@pytest.mark.parametrize("ensure_ascii", [False, True])
def test_learning_request_redacts_json_escaped_private_context(ensure_ascii: bool) -> None:
    persona = 'CANARY-PERSONA-"João\\profile"'
    knowledge = 'CANARY-KNOWLEDGE-"私人\\account"'
    observation = {
        "revision": "r1",
        "catalog": {
            "candidates": [
                {
                    "id": "observed",
                    "kind": "assistant",
                    "description": json.dumps({"echo": persona}, ensure_ascii=ensure_ascii),
                    "context": json.dumps({"echo": knowledge}, ensure_ascii=ensure_ascii),
                }
            ]
        },
        "available_options": [],
    }
    browser = SimpleNamespace(observe=AsyncMock(return_value=observation), current_state=None)
    decision = {
        **dict.fromkeys(DiscoveryDecision.model_fields),
        "action": "stop",
        "revision": "r1",
        "stop_reason": "unsupported_interface",
    }
    provider = SimpleNamespace(
        generate=AsyncMock(return_value=SimpleNamespace(content=json.dumps(decision)))
    )
    redactor = Redactor(
        PrivacyConfig(redaction_keys=["description", "private_fact"]),
        sensitive_context={
            "persona": {"description": persona},
            "knowledge": {"private_fact": knowledge},
        },
    )
    with pytest.raises(ChannelError, match="learning stopped"):
        asyncio.run(
            learn(
                cast(LearningBroker, browser),
                cast(LLMProvider, provider),
                redactor,
                budget=DiscoveryBudget.create(30),
                probes={"first": "Hello"},
            )
        )
    request = cast(LLMRequest, provider.generate.call_args.args[0])
    payload = request.messages[1].content
    assert "CANARY-PERSONA" not in payload
    assert "CANARY-KNOWLEDGE" not in payload
    assert "<redacted:field>" in payload
    assert json.loads(payload)["inventory"]["revision"] == "r1"
    assert "CANARY-PERSONA" in observation["catalog"]["candidates"][0]["description"]
    assert "CANARY-KNOWLEDGE" in observation["catalog"]["candidates"][0]["context"]


def test_compiled_state_locators_are_rejected_before_model_planning():
    observation = {
        "revision": "r1",
        "catalog": {"candidates": []},
        "available_options": [],
        "state": {
            "id": "state-1",
            "kind": "composer",
            "option_labels": [],
            "input_locator": {"strategy": "css", "value": r"#CANARY\ SECRET"},
        },
    }
    browser = SimpleNamespace(observe=AsyncMock(return_value=observation), current_state=None)
    decision = {
        **dict.fromkeys(DiscoveryDecision.model_fields),
        "action": "stop",
        "revision": "r1",
        "stop_reason": "unsupported_interface",
    }
    provider = SimpleNamespace(
        generate=AsyncMock(return_value=SimpleNamespace(content=json.dumps(decision)))
    )
    with pytest.raises(ChannelError, match="native discovery observation is invalid"):
        asyncio.run(
            learn(
                cast(LearningBroker, browser),
                cast(LLMProvider, provider),
                Redactor(PrivacyConfig()),
                budget=DiscoveryBudget.create(30),
                probes={"first": "Hello"},
            )
        )
    provider.generate.assert_not_awaited()
    assert "input_locator" in observation["state"]

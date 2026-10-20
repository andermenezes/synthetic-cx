import asyncio
from datetime import UTC, datetime
from functools import wraps
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from syntheticcx.config.models import StatefulChannelConfig
from syntheticcx.discovery.budget import DiscoveryBudget
from syntheticcx.discovery.contracts import LearnResult
from syntheticcx.discovery.replay import replay
from syntheticcx.domain.models import CompletedResponse, Message, SelectionReceipt
from syntheticcx.errors import ChannelError


def run_async(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return wrapper


def setup_replay(options):
    config = StatefulChannelConfig.model_validate(
        {
            "type": "webchat_playwright_stateful",
            "url": "http://localhost:8000",
            "ownership": {
                "root_locator": {"strategy": "test_id", "value": "chat"},
                "proof": "chat_dialog",
            },
            "assistant_message_locator": {"strategy": "test_id", "value": "assistant"},
            "message_identity": {"strategy": "dom_node"},
            "response_completion": {"strategy": "quiet_interval", "quiet_interval_seconds": 0.1},
            "initial_state_id": "menu",
            "states": [
                {
                    "id": "menu",
                    "kind": "menu",
                    "option_locator": {"strategy": "role", "value": "button"},
                    "option_labels": ["Help"],
                }
            ],
            "transitions": [
                {
                    "id": "help",
                    "from_state": "menu",
                    "to_state": "menu",
                    "action_kind": "select_option",
                    "option_label": "Help",
                }
            ],
            "verification_recipe": [{"transition_id": "help"}],
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
    broker = SimpleNamespace(
        current_state=config.states[0],
        observe_initial=AsyncMock(return_value=response),
        observe=AsyncMock(
            return_value={
                "revision": "fresh",
                "ownership_bound": True,
                "reader_bound": True,
                "available_options": options,
            }
        ),
        select_option=AsyncMock(
            return_value=SelectionReceipt(
                message_id="sent",
                submitted_at=now,
                selected_option_id="fresh-id",
                selected_label="Help",
                source_message_id="reply",
            )
        ),
        receive=AsyncMock(return_value=response),
    )
    return broker, LearnResult(config=config, probes={})


@run_async
async def test_fresh_option_id_and_exact_coverage():
    broker, learned = setup_replay([{"id": "fresh-id", "label": " Help "}])
    guide = await replay(broker, learned, budget=DiscoveryBudget.create(30))
    broker.select_option.assert_awaited_once_with("fresh-id")
    assert guide.channel.type == "webchat_playwright_dynamic"
    assert "select_option" in guide.validation.observed_behaviors
    assert not hasattr(guide.channel, "states")
    assert not hasattr(guide.validation, "verified_transition_ids")


@run_async
@pytest.mark.parametrize(
    "options", [[], [{"id": "a", "label": "Help"}, {"id": "b", "label": " help "}]]
)
async def test_missing_or_ambiguous_option_never_dispatches(options):
    broker, learned = setup_replay(options)
    with pytest.raises(ChannelError):
        await replay(broker, learned, budget=DiscoveryBudget.create(30))
    broker.select_option.assert_not_awaited()


@run_async
async def test_unlearned_fresh_state_never_dispatches():
    broker, learned = setup_replay([{"id": "fresh-id", "label": "Help"}])
    broker.current_state = None
    with pytest.raises(ChannelError):
        await replay(broker, learned, budget=DiscoveryBudget.create(30))
    broker.select_option.assert_not_awaited()


@run_async
async def test_complete_three_step_recipe_uses_only_three_fresh_actions():
    broker, learned = setup_replay([{"id": "fresh-id", "label": "Help"}])
    document = learned.config.model_dump()
    document["transitions"] = [
        {**document["transitions"][0], "id": f"help_{index}"} for index in range(3)
    ]
    document["verification_recipe"] = [{"transition_id": f"help_{index}"} for index in range(3)]
    learned.config = StatefulChannelConfig.model_validate(document)
    budget = DiscoveryBudget.create(30)
    guide = await replay(broker, learned, budget=budget)
    assert budget.actions == broker.select_option.await_count == 3
    assert {item.id for item in learned.config.transitions} == {"help_0", "help_1", "help_2"}
    assert "help_0" not in guide.model_dump_json()

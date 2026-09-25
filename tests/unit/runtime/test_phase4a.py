"""Deterministic configuration, repetition, behavior, and time contracts."""

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from syntheticcx.config.loader import load_config, load_config_text
from syntheticcx.config.models import BehaviorRule, ScenarioInformation
from syntheticcx.domain.models import CompletedResponse, Message, PersonaState, SendReceipt
from syntheticcx.errors import ConfigurationError
from syntheticcx.runtime.behavior import apply_behavior
from syntheticcx.runtime.clock import (
    ManualClock,
    ManualWaitController,
    RealClock,
    RealWaitController,
)
from syntheticcx.runtime.repetition import RepetitionTracker, normalize_phrase

NOW = datetime(2026, 9, 24, tzinfo=UTC)


@pytest.mark.parametrize(
    ("information", "expected"),
    [
        ([{"id": " ", "aliases": ["account number"]}], "information ID"),
        ([{"id": "account_number", "aliases": []}], "information.0.aliases"),
        ([{"id": "account_number", "aliases": ["   "]}], "aliases"),
        ([{"id": "account_number", "aliases": ["!?-"]}], "aliases"),
        (
            [{"id": "account_number", "aliases": ["account number"], "unexpected": True}],
            "unexpected",
        ),
        ([{"id": "one", "aliases": ["first"]}, {"id": "one", "aliases": ["second"]}], "unique"),
    ],
)
def test_information_configuration_is_strict(
    information: list[dict[str, object]], expected: str
) -> None:
    document = yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())
    document["scenario"]["information"] = information
    with pytest.raises(ConfigurationError, match=expected):
        load_config_text(yaml.safe_dump(document))


def test_example_declares_information_aliases() -> None:
    config = load_config(Path("examples/telecom/scenario.yaml"))
    assert config.scenario.information[0].id == "account_number"
    assert config.scenario.information[0].aliases == ["account number", "account ID"]


def assistant(identifier: str, content: str, sequence: int) -> Message:
    return Message(
        id=identifier, actor="assistant", sequence=sequence, content=content, observed_at=NOW
    )


def response(*messages: Message) -> CompletedResponse:
    return CompletedResponse(
        messages=list(messages),
        started_at=NOW,
        completed_at=NOW,
        completion_strategy="completion_locator",
    )


def test_normalization_and_word_boundaries() -> None:
    assert normalize_phrase("  ACCOUNT—Number?  ") == ("account", "number")
    assert normalize_phrase("ＣＡＦÉ") == ("ｃａｆé",)
    tracker = RepetitionTracker([ScenarioInformation(id="account", aliases=["Account Number"])])
    tracker.record_send(
        "Here is my ACCOUNT-number: 123", SendReceipt(message_id="sent-1", submitted_at=NOW)
    )
    assert tracker.detect(response(assistant("a", "Please provide your accountnumber", 1))) == []
    matches = tracker.detect(response(assistant("a", "What is the account number?", 1)))
    assert [
        (match.information_id, match.original_message_id, match.request_message_id)
        for match in matches
    ] == [("account", "sent-1", "a")]


def test_unsent_information_is_ignored_and_each_id_matches_once_per_response() -> None:
    tracker = RepetitionTracker(
        [
            ScenarioInformation(id="account", aliases=["account number", "account ID"]),
            ScenarioInformation(id="invoice", aliases=["invoice amount"]),
        ]
    )
    request = response(assistant("a", "Please provide the account ID and invoice amount", 1))
    assert tracker.detect(request) == []
    tracker.record_send(
        "My account number and invoice amount are 123",
        SendReceipt(message_id="sent-1", submitted_at=NOW),
    )
    tracker.record_send("Another account ID", SendReceipt(message_id="sent-2", submitted_at=NOW))
    matches = tracker.detect(
        response(
            assistant("a", "Please provide the account ID and invoice amount", 1),
            assistant("b", "What is your account number?", 2),
        )
    )
    assert [
        (match.information_id, match.original_message_id, match.request_message_id)
        for match in matches
    ] == [("account", "sent-1", "a"), ("invoice", "sent-1", "a")]


def test_information_mention_without_request_does_not_trigger_repetition() -> None:
    tracker = RepetitionTracker([ScenarioInformation(id="account", aliases=["account number"])])
    tracker.record_send(
        "My account number is 123", SendReceipt(message_id="sent", submitted_at=NOW)
    )
    assert tracker.detect(response(assistant("ack", "Your account number is on file.", 1))) == []


def rule(
    identifier: str,
    priority: int,
    frustration: float,
    willingness: float,
    threshold: float | None = None,
) -> BehaviorRule:
    return BehaviorRule(
        id=identifier,
        priority=priority,
        event="repeated_information",
        frustration_delta=frustration,
        willingness_to_continue_delta=willingness,
        abandonment_threshold=threshold,
    )


def test_rules_sort_accumulate_clamp_and_abandon_without_mutation() -> None:
    state = PersonaState(frustration=0.1, willingness_to_continue=0.4)
    rules = [rule("z", 1, 0.8, -0.3, 0.2), rule("a", 1, 0.5, -0.1), rule("first", -1, -0.1, 0)]
    transition = apply_behavior(
        state,
        rules,
        event="repeated_information",
        trigger_ids=["repeated_information:account", "sent-1", "ask-1"],
    )
    assert transition.applied_rule_ids == ["first", "a", "z"]
    assert transition.before == state
    assert state.frustration == 0.1
    assert transition.after.frustration == 1
    assert transition.after.willingness_to_continue == 0
    assert transition.after.abandoned
    assert transition.trigger_ids == ["repeated_information:account", "sent-1", "ask-1"]
    assert apply_behavior(
        transition.after, rules, event="repeated_information", trigger_ids=["next"]
    ).after.abandoned


def test_nonmatching_event_preserves_state() -> None:
    state = PersonaState()
    result = apply_behavior(state, [rule("a", 1, 0.5, -0.5)], event="other", trigger_ids=["x"])
    assert result.after == state
    assert result.applied_rule_ids == []


def test_manual_clock_timeout_and_completion_without_real_sleep() -> None:
    async def scenario() -> None:
        clock = ManualClock(NOW)
        controller = ManualWaitController(clock)
        waiter = asyncio.create_task(
            controller.run(clock.wait_until(NOW + timedelta(seconds=20)), 5)
        )
        await asyncio.sleep(0)
        await clock.advance(5)
        with pytest.raises(TimeoutError):
            await waiter
        assert clock.now() == NOW + timedelta(seconds=5)
        completed = asyncio.create_task(
            controller.run(clock.wait_until(NOW + timedelta(seconds=8)), 10)
        )
        await asyncio.sleep(0)
        await clock.advance(3)
        assert await completed is None

    asyncio.run(scenario())


def test_real_clock_and_wait_controller_have_finite_deadlines() -> None:
    assert RealClock().now().tzinfo is not None

    async def scenario() -> None:
        controller = RealWaitController()
        assert await controller.run(asyncio.sleep(0, result="done"), 1) == "done"
        with pytest.raises(TimeoutError):
            await controller.run(asyncio.sleep(10), 0.01)

    asyncio.run(scenario())

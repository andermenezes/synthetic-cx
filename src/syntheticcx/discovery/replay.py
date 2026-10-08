"""One fresh, model-free replay of the complete learned path."""

from __future__ import annotations

import unicodedata
from collections.abc import Callable
from datetime import UTC, datetime
from urllib.parse import urlsplit

from syntheticcx.config.models import (
    ChatState,
    InterfaceChannelConfig,
    InterfaceChannelGuide,
    InterfaceGuideValidation,
)
from syntheticcx.discovery.budget import DiscoveryBudget
from syntheticcx.discovery.contracts import LearningBroker, LearnResult
from syntheticcx.domain.models import CompletedResponse, SelectionReceipt, SendReceipt
from syntheticcx.errors import ChannelError


def normalized_label(label: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", label).split()).casefold()


def guard_state(actual: ChatState | None, expected: ChatState) -> None:
    if actual != expected:
        raise ChannelError("discover replay", "fresh state does not match learned state")


async def replay(
    browser: LearningBroker,
    learned: LearnResult,
    *,
    budget: DiscoveryBudget,
    browser_action_timeout_seconds: float = 30,
    response_timeout_seconds: float = 30,
    on_progress: Callable[[str], None] | None = None,
) -> InterfaceChannelGuide:
    config = learned.config
    states = {state.id: state for state in config.states}
    transitions = {item.id: item for item in config.transitions}
    budget.check()
    await budget.run(browser.observe_initial(), response_timeout_seconds)
    guard_state(browser.current_state, states[config.initial_state_id])
    seen_states = {config.initial_state_id}
    seen_transitions: set[str] = set()
    behaviors = {"open", "receive", "completion"}
    for index, step in enumerate(config.verification_recipe, 1):
        transition = transitions[step.transition_id]
        if on_progress:
            on_progress(f"fresh replay step {index}: {transition.action_kind}")
        budget.consume("observations")
        inventory = await budget.run(browser.observe(), browser_action_timeout_seconds)
        guard_state(browser.current_state, states[transition.from_state])
        if not inventory.get("ownership_bound") or not inventory.get("reader_bound"):
            raise ChannelError("discover replay", "fresh ownership or reader is unbound")
        budget.consume("actions")
        if transition.action_kind == "probe":
            assert transition.probe_id is not None
            text = learned.probes.get(transition.probe_id)
            target = inventory.get("composer_id")
            if text is None or target is None:
                raise ChannelError("discover replay", "fresh probe is unavailable")
            receipt = await budget.run(
                browser.submit_probe(inventory["revision"], target, text),
                browser_action_timeout_seconds,
            )
            behaviors.add("send")
        else:
            assert transition.option_label is not None
            options = [
                item
                for item in inventory["available_options"]
                if normalized_label(item["label"]) == normalized_label(transition.option_label)
            ]
            if len(options) != 1:
                raise ChannelError(
                    "discover replay", "fresh canonical option is unavailable or ambiguous"
                )
            receipt = await budget.run(
                browser.select_option(options[0]["id"]), browser_action_timeout_seconds
            )
            if (
                not isinstance(receipt, SelectionReceipt)
                or receipt.selected_option_id != options[0]["id"]
                or normalized_label(receipt.selected_label)
                != normalized_label(transition.option_label)
            ):
                raise ChannelError("discover replay", "fresh option receipt is inconsistent")
            behaviors.add("select_option")
        if not isinstance(receipt, SendReceipt):
            raise ChannelError("discover replay", "fresh submission outcome is uncertain")
        response = await budget.run(browser.receive(), response_timeout_seconds)
        if not isinstance(response, CompletedResponse):
            raise ChannelError("discover replay", "fresh response is incomplete")
        budget.consume("observations")
        await budget.run(browser.observe(), browser_action_timeout_seconds)
        guard_state(browser.current_state, states[transition.to_state])
        seen_states.add(transition.to_state)
        seen_transitions.add(transition.id)
        if on_progress:
            on_progress(f"fresh replay step {index} completed")
    if seen_states != set(states) or seen_transitions != set(transitions):
        raise ChannelError("discover replay", "fresh coverage is incomplete")
    url = urlsplit(str(config.url))
    initial = states[config.initial_state_id]
    boundary = config.model_dump(
        mode="json",
        exclude={"type", "initial_state_id", "states", "transitions", "verification_recipe"},
    )
    return InterfaceChannelGuide(
        version="1",
        channel=InterfaceChannelConfig.model_validate(
            {
                **boundary,
                "type": "webchat_playwright_dynamic",
                "bootstrap_input_locator": initial.input_locator,
                "bootstrap_send_strategy": initial.send_strategy,
            }
        ),
        validation=InterfaceGuideValidation(
            status="verified",
            verified_at=datetime.now(UTC),
            target_origin=f"{url.scheme}://{url.netloc}",
            observed_behaviors=sorted(behaviors),
        ),
    )

"""Website planning budgets and abstention do not authorize browser effects."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from syntheticcx.config.models import PrivacyConfig
from syntheticcx.discovery.budget import DiscoveryBudget
from syntheticcx.discovery.contracts import DiscoveryDecision
from syntheticcx.discovery.learn import explore_website
from syntheticcx.errors import ChannelError, ModelError
from syntheticcx.privacy import Redactor


class Website:
    chat_hint = None

    def __init__(self, *, finish_after: int | None = None) -> None:
        self.effects: list[str] = []
        self.finish_after = finish_after
        self.revision = 0

    async def observe_website(self) -> dict[str, Any]:
        if self.finish_after is not None and len(self.effects) >= self.finish_after:
            return {"terminal": "webchat"}
        self.revision += 1
        return {
            "revision": str(self.revision),
            "terminal": None,
            "available_actions": ["observe", "wait", "open", "stop"],
            "catalog": {
                "candidates": [
                    {
                        "id": "current-control",
                        "kind": "opener",
                        "description": "Uncertain route",
                        "actions": ["open"],
                    }
                ],
                "has_more": False,
            },
        }

    async def act_website(self, revision: str, action: str, target_id: str | None = None) -> None:
        assert revision == str(self.revision)
        self.effects.append(action)


class Decisions:
    def __init__(self, action: str, stop_reason: str = "ambiguous_evidence") -> None:
        self.action = action
        self.stop_reason = stop_reason
        self.requests = 0

    async def generate(self, request: Any) -> Any:
        self.requests += 1
        payload = json.loads(request.messages[-1].content)
        fields = dict.fromkeys(DiscoveryDecision.model_fields)
        fields.update(action=self.action, revision=payload["inventory"]["revision"])
        if self.action == "stop":
            fields["stop_reason"] = self.stop_reason
        if self.action in {"open", "hover"}:
            fields["target_id"] = "current-control"
        return SimpleNamespace(content=json.dumps(fields))


def redactor() -> Redactor:
    return Redactor(PrivacyConfig())


def test_model_cannot_follow_a_portability_route_even_with_its_current_id():
    class PortabilityWebsite(Website):
        async def observe_website(self):
            observation = await super().observe_website()
            observation["catalog"]["candidates"][0]["description"] = "Portabilidade"
            return observation

    async def exercise():
        website = PortabilityWebsite()
        with pytest.raises(ModelError):
            await explore_website(
                website,
                Decisions("open"),
                redactor(),
                budget=DiscoveryBudget.create(10),
                max_discovery_attempts=2,
            )
        assert website.effects == []

    asyncio.run(exercise())


def test_website_actions_preserve_all_conversation_operation_allowances() -> None:
    async def exercise() -> None:
        website = Website(finish_after=6)
        budget = DiscoveryBudget.create(10)
        await explore_website(website, Decisions("wait"), redactor(), budget=budget)
        assert website.effects == ["wait"] * 6
        assert (budget.actions, budget.observations, budget.cycles, budget.waits) == (0, 0, 0, 0)

    asyncio.run(exercise())


def test_action_limit_prevents_the_next_browser_effect() -> None:
    async def exercise() -> None:
        website = Website()
        with pytest.raises(ChannelError, match="website action budget exhausted"):
            await explore_website(
                website,
                Decisions("wait"),
                redactor(),
                budget=DiscoveryBudget.create(10),
                website_action_limit=2,
            )
        assert website.effects == ["wait", "wait"]

    asyncio.run(exercise())


def test_observation_limit_terminates_a_model_that_only_observes() -> None:
    async def exercise() -> None:
        website = Website()
        provider = Decisions("observe")
        with pytest.raises(ChannelError, match="website observation budget exhausted"):
            await explore_website(
                website,
                provider,
                redactor(),
                budget=DiscoveryBudget.create(10),
                website_observation_limit=3,
            )
        assert website.effects == []
        assert provider.requests == 3

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "reason", ["ambiguous_evidence", "unsafe_interface", "unsupported_interface"]
)
def test_uncertain_route_can_abstain_without_being_forced_to_click(reason: str) -> None:
    async def exercise() -> None:
        website = Website()
        provider = Decisions("stop", reason)
        with pytest.raises(ChannelError, match=reason):
            await explore_website(website, provider, redactor(), budget=DiscoveryBudget.create(10))
        assert website.effects == []
        assert provider.requests == 1

    asyncio.run(exercise())


def test_unoffered_hover_cannot_dispatch_even_with_a_valid_current_control_id() -> None:
    async def exercise() -> None:
        website = Website()
        provider = Decisions("hover")
        with pytest.raises(ModelError):
            await explore_website(
                website,
                provider,
                redactor(),
                budget=DiscoveryBudget.create(10),
                max_discovery_attempts=2,
            )
        assert website.effects == []
        assert provider.requests == 2

    asyncio.run(exercise())


def test_covered_launcher_cannot_compete_with_an_available_overlay_action() -> None:
    class CoveredWebsite(Website):
        async def observe_website(self) -> dict[str, Any]:
            observation = await super().observe_website()
            if observation.get("terminal"):
                return observation
            observation["catalog"]["candidates"].insert(
                0,
                {
                    "id": "covered-launcher",
                    "kind": "opener",
                    "description": "Messaging channel",
                    "actions": [],
                    "blocked_by": "Choose your browsing region",
                },
            )
            return observation

        async def act_website(
            self, revision: str, action: str, target_id: str | None = None
        ) -> None:
            assert target_id == "current-control"
            await super().act_website(revision, action, target_id)

    class CurrentTarget:
        async def generate(self, request: Any) -> Any:
            inventory = json.loads(request.messages[-1].content)["inventory"]
            fields = dict.fromkeys(DiscoveryDecision.model_fields)
            fields.update(
                action="open",
                revision=inventory["revision"],
                target_id=inventory["catalog"]["candidates"][0]["id"],
            )
            return SimpleNamespace(content=json.dumps(fields))

    async def exercise() -> None:
        website = CoveredWebsite(finish_after=1)
        await explore_website(
            website, CurrentTarget(), redactor(), budget=DiscoveryBudget.create(10)
        )
        assert website.effects == ["open"]

    asyncio.run(exercise())


def test_pre_dispatch_readiness_change_replans_without_retrying_a_browser_effect() -> None:
    class ChangingWebsite(Website):
        refused = False

        async def act_website(
            self, revision: str, action: str, target_id: str | None = None
        ) -> None:
            if not self.refused:
                self.refused = True
                raise ChannelError("website.readiness", "control changed before dispatch")
            await super().act_website(revision, action, target_id)

    async def exercise() -> None:
        website = ChangingWebsite(finish_after=1)
        provider = Decisions("open")
        await explore_website(website, provider, redactor(), budget=DiscoveryBudget.create(10))
        assert website.effects == ["open"]
        assert provider.requests == 2

    asyncio.run(exercise())


@pytest.mark.parametrize("reason", ["no_supported_controls", "unsupported_interface"])
def test_false_absence_of_controls_is_corrected_without_forcing_a_browser_action(
    reason: str,
) -> None:
    class BrowsingPrompt(Website):
        chat_hint = "Messaging channel"

        async def observe_website(self) -> dict[str, Any]:
            observation = await super().observe_website()
            if not observation.get("terminal"):
                observation["catalog"]["candidates"].append(
                    {
                        "id": "covered",
                        "kind": "opener",
                        "description": "Messaging channel",
                        "actions": [],
                        "blocked_by": "Choose your browsing preference",
                    }
                )
            return observation

    class CorrectedDecision(Decisions):
        async def generate(self, request: Any) -> Any:
            response = await super().generate(request)
            if self.requests == 1:
                decision = json.loads(response.content)
                decision.update(action="stop", target_id=None, stop_reason=reason)
                return SimpleNamespace(content=json.dumps(decision))
            return response

    async def exercise() -> None:
        website = BrowsingPrompt(finish_after=1)
        provider = CorrectedDecision("open")
        await explore_website(website, provider, redactor(), budget=DiscoveryBudget.create(10))
        assert website.effects == ["open"]
        assert provider.requests == 2

    asyncio.run(exercise())

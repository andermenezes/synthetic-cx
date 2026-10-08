"""Consequential action permissions are explicit, scenario-scoped and bounded."""

import pytest
from pydantic import ValidationError

from syntheticcx.config.models import ActionAuthorization, ScenarioConfig


def test_payment_authorization_requires_sandbox_amount_and_currency():
    authorization = ActionAuthorization(
        action="payment", environment="sandbox", max_amount=25, currency="USD"
    )
    assert authorization.max_amount == 25
    for data in (
        {"action": "payment"},
        {"action": "payment", "environment": "sandbox", "max_amount": 25},
        {
            "action": "payment",
            "environment": "sandbox",
            "max_amount": 25,
            "currency": "usd",
        },
        {
            "action": "payment",
            "environment": "live",
            "max_amount": 25,
            "currency": "USD",
        },
    ):
        with pytest.raises(ValidationError):
            ActionAuthorization.model_validate(data)


def test_non_payment_authorization_cannot_smuggle_payment_bounds():
    with pytest.raises(ValidationError, match="payment bounds apply only"):
        ActionAuthorization(action="purchase", environment="sandbox", max_amount=25, currency="USD")


def test_scenario_rejects_duplicate_action_authorizations():
    scenario = {
        "id": "payment",
        "description": "sandbox payment test",
        "goal": "Pay the sandbox invoice",
        "action_authorizations": [
            {
                "action": "payment",
                "environment": "sandbox",
                "max_amount": 25,
                "currency": "USD",
            },
            {
                "action": "payment",
                "environment": "sandbox",
                "max_amount": 10,
                "currency": "USD",
            },
        ],
        "success_criteria": [
            {
                "id": "paid",
                "evaluator_type": "turn_count",
                "result_kind": "boolean",
                "evidence_policy": "none",
                "parameters": {"max_turns": 2},
            }
        ],
    }
    with pytest.raises(ValidationError, match="must be unique"):
        ScenarioConfig.model_validate(scenario)

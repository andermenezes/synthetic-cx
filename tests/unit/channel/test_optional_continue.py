"""Continuation needs explicit local optional-contact evidence."""

import pytest

from syntheticcx.adapters.channel.options import eligibility

SAFE = {"safe": True, "visible": True, "enabled": True}


@pytest.mark.parametrize(
    "prose,label",
    [
        ("Seu e-mail é opcional.", "Continuar"),
        ("Se não quiser informar seu email, clique em Continuar.", "Continuar"),
        ("Você pode continuar sem informar seu telefone.", "Continuar"),
        ("Your email is optional.", "Continue"),
        ("If you don't want to provide your email, click Continue.", "Continue"),
    ],
)
def test_optional_contact_continuation(prose, label):
    assert eligibility(prose, label, SAFE) is None


@pytest.mark.parametrize(
    "prose,label",
    [
        ("Qual é seu email?", "Continuar"),
        ("Choose an option", "Continue"),
        ("Your email is optional. Confirm purchase.", "Continue"),
        ("Seu email é opcional. Aceitar consentimento.", "Continuar"),
        ("Your email is optional. Sign in.", "Continue"),
        ("Seu email é opcional. Redirecionar ao site externo.", "Continuar"),
        ("Your email is optional.", "Continue purchase"),
        ("Your password is optional.", "Continue"),
    ],
)
def test_ambiguous_or_unsafe_continuation_remains_blocked(prose, label):
    assert eligibility(prose, label, SAFE) is not None


@pytest.mark.parametrize("key", ["safe", "visible", "enabled"])
def test_optional_context_never_overrides_native_guards(key):
    assert eligibility("Your email is optional.", "Continue", {**SAFE, key: False})


def test_dynamic_runtime_allows_contextual_ordinary_choices_without_keyword_whitelist():
    assert (
        eligibility(
            "Please choose how you want to proceed.", "Continuar", SAFE, dynamic_runtime=True
        )
        is None
    )
    assert eligibility("Qual é seu email?", "Continuar", SAFE, dynamic_runtime=True) is None


def test_dynamic_runtime_requires_scenario_authorization_for_consequential_action():
    assert eligibility("Choose payment method", "Pay $12.00", SAFE, dynamic_runtime=True) == (
        "scenario_action_not_authorized"
    )
    authorization = {
        "action": "payment",
        "environment": "sandbox",
        "max_amount": 20,
        "currency": "USD",
    }
    assert (
        eligibility(
            "Choose payment method",
            "Pay $12.00",
            SAFE,
            dynamic_runtime=True,
            authorizations=[authorization],
        )
        is None
    )
    assert (
        eligibility(
            "Choose payment method",
            "Pay $25.00",
            SAFE,
            dynamic_runtime=True,
            authorizations=[authorization],
        )
        == "payment_limit_exceeded"
    )
    assert (
        eligibility(
            "Choose payment method",
            "Pay now",
            SAFE,
            dynamic_runtime=True,
            authorizations=[authorization],
        )
        == "payment_amount_unbounded"
    )


def test_dynamic_runtime_requires_scenario_authorization_for_consent():
    assert eligibility("Review terms", "Accept", SAFE, dynamic_runtime=True) == (
        "scenario_action_not_authorized"
    )
    assert (
        eligibility(
            "Review terms",
            "Accept",
            SAFE,
            dynamic_runtime=True,
            authorizations=[{"action": "consent"}],
        )
        is None
    )


@pytest.mark.parametrize("label", ["Activate plan", "Deactivate account", "Transfer funds"])
def test_all_consequential_choices_require_authorization(label):
    assert eligibility("Choose", label, SAFE, dynamic_runtime=True) == (
        "scenario_action_not_authorized"
    )


def test_compound_choice_requires_every_authorization():
    assert (
        eligibility(
            "Buy and cancel old subscription",
            "Buy",
            SAFE,
            dynamic_runtime=True,
            authorizations=[{"action": "purchase"}],
        )
        == "scenario_action_not_authorized"
    )


@pytest.mark.parametrize(
    "prose,label,reason",
    [
        ("Your balance is $5. Pay $100 now.", "Pay $100", "payment_limit_exceeded"),
        ("Select payment amount: $10 or $1000", "Pay $1000", "payment_limit_exceeded"),
        ("Choose payment", "Pay $10 EUR", "payment_currency_mismatch"),
        ("Make payment in EUR", "Pay $10", "payment_currency_mismatch"),
        ("Make payment in BRL", "Pay $10 USD", "payment_currency_mismatch"),
        ("Choose payment", "Pay BRL 10 and USD 500", "payment_amount_unbounded"),
        ("Pay $10 or $100", "Pay now", "payment_amount_unbounded"),
        ("Payment due USD 10", "Pay 500", "payment_amount_unbounded"),
        ("Choose payment", "Pay $10 + 500 USD", "payment_amount_unbounded"),
        ("Choose payment", "Pay $1,000", "payment_amount_unbounded"),
        ("Choose payment", "Pay $1.000,00", "payment_amount_unbounded"),
        ("Choose payment", "Pay $1 000", "payment_amount_unbounded"),
        ("Choose payment", "Pay $-10", "payment_amount_unbounded"),
    ],
)
def test_payment_bounds_follow_selected_control_and_reject_conflicts(prose, label, reason):
    assert (
        eligibility(
            prose,
            label,
            SAFE,
            dynamic_runtime=True,
            authorizations=[
                {
                    "action": "payment",
                    "environment": "sandbox",
                    "max_amount": 25,
                    "currency": "USD",
                }
            ],
        )
        == reason
    )

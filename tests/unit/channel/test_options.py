"""Conservative EN/PT informational menu policy, independent of browsers."""

import pytest

from syntheticcx.adapters.channel.options import eligibility, normalized_label

SAFE = {"safe": True, "visible": True, "enabled": True}


@pytest.mark.parametrize(
    "prose,label",
    [
        ("Choose a topic for information", "Plans"),
        ("Which service can I help you learn about?", "Coverage"),
        ("Escolha um assunto para saber mais", "Planos"),
        ("Qual informação você deseja?", "Cobertura"),
    ],
)
def test_positive_informational_menus(prose: str, label: str) -> None:
    assert eligibility(prose, label, SAFE) is None


@pytest.mark.parametrize(
    "prose,label",
    [
        ("Proceed", "Plans"),
        ("Choose", "Blue"),
        ("Choose a payment method for your plan", "Details"),
        ("Choose an information topic", "Buy plan"),
        ("Escolha informações sobre planos", "Confirmar"),
        ("Escolha informações sobre planos", "Sim"),
        ("Choose information about coverage", "Continue"),
        ("Choose a topic to consent", "Coverage"),
        ("Choose an information topic", ""),
    ],
)
def test_ambiguous_or_unsafe_context_rejected(prose: str, label: str) -> None:
    assert eligibility(prose, label, SAFE) is not None


@pytest.mark.parametrize("field", ["safe", "visible", "enabled"])
def test_unavailable_or_structurally_unsafe_control(field: str) -> None:
    assert eligibility("Choose an information topic", "Plans", {**SAFE, field: False})


def test_duplicate_label_normalization_is_unicode_complete() -> None:
    assert normalized_label("  ＰＬＡＮＳ\n ") == normalized_label("plans")


@pytest.mark.parametrize(
    "prose,label",
    [
        (
            "Por favor escolha uma opção abaixo para direcionar melhor seu atendimento:",
            "Contratei Sky+ pelo Site",
        ),
        (
            "Por favor escolha uma opção abaixo para direcionar melhor seu atendimento:",
            "Ainda não tenho Sky+",
        ),
        ("Selecione uma das opções disponíveis", "Já sou cliente antigo"),
        ("Please choose an option below so I can route you", "I already have the service"),
        ("Select one of the options to continue our conversation", "Signed through a partner"),
        ("Pick an option", "Already a customer here"),
    ],
)
def test_choice_prompt_authorizes_conversational_routing_labels(prose: str, label: str) -> None:
    assert eligibility(prose, label, SAFE) is None


@pytest.mark.parametrize(
    "prose,label",
    [
        # No explicit choice prompt: routing vocabulary stays rejected.
        ("Entendi, já vou te ajudar com isso.", "Contratei pelo site"),
        # Transactional, action and ambiguous vocabulary stays rejected under a prompt.
        ("Por favor escolha uma opção abaixo", "Pagar fatura"),
        ("Por favor escolha uma opção abaixo", "Alterar plano"),
        ("Por favor escolha uma opção abaixo", "Confirmar"),
        ("Por favor escolha uma opção abaixo", "Sim"),
        ("Por favor escolha uma opção abaixo", ""),
        # Bubble utility controls are never conversational routing choices.
        ("Por favor escolha uma opção abaixo", "Copy"),
        ("Por favor escolha uma opção abaixo", "Copiar"),
        ("Por favor escolha uma opção abaixo", "Share"),
        ("Por favor escolha uma opção abaixo", "Avaliar atendimento"),
        # Unsafe context poisons the whole reply.
        ("Escolha uma opção para pagar sua fatura", "Já sou cliente"),
    ],
)
def test_choice_prompt_never_authorizes_transactional_or_utility_labels(
    prose: str, label: str
) -> None:
    assert eligibility(prose, label, SAFE) is not None


def test_choice_prompt_requires_option_vocabulary_not_just_a_choice_verb() -> None:
    assert eligibility("Choose carefully what happens next", "Already a customer", SAFE) is not None
    assert eligibility("Escolha um caminho", "Já sou cliente", SAFE) is not None


def test_options_only_requires_multiple_positive_informational_labels() -> None:
    assert eligibility("", "Plans", SAFE, ["Plans", "Coverage"]) is None
    assert eligibility("", "Plans", SAFE, ["Plans"]) is not None
    assert eligibility("", "Plans", SAFE, ["Plans", "Continue"]) is not None


@pytest.mark.parametrize(
    "label",
    [
        "Activate plan",
        "Withdraw funds",
        "pa\u200by now",
        "\u200b",
        "Ativar plano",
        "Sacar dinheiro",
        "Alterar plano",
        "pa\u200bgar",
        "Upgrade plan",
        "Enable service",
        "Change coverage",
        "Send funds",
        "Unknown choice",
        "Plan secret command",
        "\u202ePlans",
        "Plans\x00",
        "Plans\x08",
    ],
)
def test_positive_context_never_authorizes_unknown_action_or_spoofed_labels(label: str) -> None:
    assert eligibility("Choose a topic for service information", label, SAFE) is not None


@pytest.mark.parametrize(
    "label",
    [
        "View plan details",
        "Read about coverage",
        "Help with billing",
        "Plans and pricing",
        "Ver planos",
        "Informações sobre cobertura",
        "Detalhes da fatura",
        "Ajuda com planos",
    ],
)
def test_explicit_read_only_topic_phrasing(label: str) -> None:
    assert eligibility("Choose a topic for service information", label, SAFE) is None


def test_hidden_character_spoofs_cannot_evade_duplicates_or_context_guard() -> None:
    assert normalized_label("pa\u200by now") == normalized_label("pay now")
    assert normalized_label("Pla\u200bns") == normalized_label("Plans")
    assert normalized_label("\u200b\x00") == ""
    assert eligibility("Choose pa\u200by now for plan information", "Plans", SAFE) is not None

"""Observed send metadata remains generic and cannot promote arbitrary menu buttons."""

import json

import pytest

from syntheticcx.discovery.catalog import interaction_catalog


@pytest.mark.parametrize(
    "identifier,dialog,label,expected",
    [
        ("widget-send-message", True, "", True),
        ("widget_enviar_mensagem", True, "", True),
        ("widget-send-message", False, "", False),
        ("widget-send-purchase", True, "", False),
        ("widget-send-upload", True, "", False),
        ("widget-send-message", True, "Buy now", False),
        ("widget-menu", True, "", False),
        ("widget-send", True, "Send a message", True),
    ],
)
def test_dialog_send_candidates_require_explicit_observed_semantics(
    identifier, dialog, label, expected
):
    node = {
        "tag": "button",
        "visible": True,
        "in_dialog": dialog,
        "attributes": {"id": identifier, "type": "button", "aria-label": label},
    }
    catalog = interaction_catalog("", json.dumps({"elements": [node]}))
    assert any(item.kind == "send" for item in catalog.items.values()) is expected


def test_dialog_composer_excludes_page_search_and_newsletter_send_candidates():
    inspection = json.dumps(
        {
            "elements": [
                {
                    "tag": "input",
                    "visible": True,
                    "editable": True,
                    "in_dialog": False,
                    "attributes": {"id": "page-search", "placeholder": "Search"},
                },
                {
                    "tag": "textarea",
                    "visible": True,
                    "editable": True,
                    "in_dialog": True,
                    "attributes": {"id": "composer", "placeholder": "Message"},
                },
                {
                    "tag": "button",
                    "visible": True,
                    "in_dialog": False,
                    "attributes": {"id": "newsletter", "aria-label": "Send"},
                },
            ]
        }
    )
    snapshot = (
        '- textbox "Search"\n- button "Send"\n- dialog\n  - textbox "Message"\n  - button "Enviar"'
    )
    catalog = interaction_catalog(snapshot, inspection)
    controls = [item.value for item in catalog.items.values() if item.kind == "input"]
    assert controls and all(
        "Search" not in str(item) and "page-search" not in str(item) for item in controls
    )
    assert not any(
        item.kind == "send" and item.value.name == "Send" for item in catalog.items.values()
    )
    assert any(
        item.kind == "send" and item.value.name == "Enviar" for item in catalog.items.values()
    )


@pytest.mark.parametrize(
    "label,offered",
    [
        ("Fale conosco", True),
        ("Suporte", True),
        ("Comprar pelo chat", False),
        ("Enviar mensagem", False),
    ],
)
def test_portuguese_launcher_candidates_share_the_safe_opening_policy(label, offered):
    catalog = interaction_catalog(f'- button "{label}" [ref=e1]', "")
    assert any(item.kind == "opener" for item in catalog.items.values()) is offered

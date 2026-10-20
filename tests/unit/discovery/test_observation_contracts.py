"""Native observations fail at the owned boundary, not at planner dictionary access."""

import pytest

from syntheticcx.discovery.observations import learning_observation, website_observation
from syntheticcx.errors import ChannelError


def test_learning_inventory_preserves_json_shape():
    value = {
        "revision": "current",
        "catalog": {
            "revision": "current",
            "candidates": [{"id": "composer", "kind": "input", "description": "Customer message"}],
        },
        "available_options": [],
        "composer_id": "composer",
    }
    assert learning_observation(value) == value


@pytest.mark.parametrize(
    "catalog",
    [
        {"candidates": [{"id": "control", "kind": "invented", "description": "control"}]},
        {"candidates": [{"id": "control", "kind": "input"}]},
        {"revision": "stale", "candidates": []},
        {"candidates": [], "native_handle": object()},
    ],
)
def test_invalid_learning_catalog_fails_safely(catalog):
    with pytest.raises(ChannelError) as failure:
        learning_observation({"revision": "current", "catalog": catalog, "available_options": []})
    assert failure.value.operation == "channel.observe"
    assert "invented" not in str(failure.value)


def test_website_terminal_and_nonterminal_shapes_are_distinguished():
    assert website_observation({"terminal": "whatsapp"}) == {"terminal": "whatsapp"}
    with pytest.raises(ChannelError, match="incomplete"):
        website_observation({"terminal": None})
    with pytest.raises(ChannelError, match="invalid"):
        website_observation(
            {
                "revision": "current",
                "catalog": {"candidates": []},
                "available_actions": ["execute_javascript"],
            }
        )

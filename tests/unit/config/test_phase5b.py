"""Finite echo observation and explicit screenshot-target affirmation."""

from pathlib import Path

import pytest
import yaml

from syntheticcx.config.loader import load_config_text
from syntheticcx.errors import ConfigurationError


def document() -> dict[str, object]:
    return yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())  # type: ignore[no-any-return]


def test_echo_window_is_positive_finite_and_materialized() -> None:
    config = load_config_text(yaml.safe_dump(document()))
    assert config.execution.echo_observation_seconds == 0.25
    for value in (0, -1, float("inf"), float("nan")):
        source = document()
        source["execution"]["echo_observation_seconds"] = value  # type: ignore[index]
        with pytest.raises(ConfigurationError, match="echo_observation_seconds"):
            load_config_text(yaml.safe_dump(source))


def test_screenshot_storage_needs_three_explicit_opt_ins() -> None:
    source = document()
    source["storage"]["store_screenshots"] = True  # type: ignore[index]
    source["privacy"]["allow_screenshots"] = True  # type: ignore[index]
    with pytest.raises(ConfigurationError, match="safe-target affirmation"):
        load_config_text(yaml.safe_dump(source))
    source["privacy"]["confirm_screenshot_target_safe"] = True  # type: ignore[index]
    assert load_config_text(yaml.safe_dump(source)).storage.store_screenshots


def test_xpath_is_not_accepted_as_css() -> None:
    source = document()
    source["channel"]["input_locator"] = {  # type: ignore[index]
        "strategy": "css",
        "value": "//input",
    }
    with pytest.raises(ConfigurationError, match="XPath"):
        load_config_text(yaml.safe_dump(source))


def test_content_hash_fallback_is_limited_to_controlled_fixture_urls() -> None:
    source = document()
    source["channel"]["message_identity"] = {  # type: ignore[index]
        "strategy": "actor_sequence_content_hash"
    }
    assert load_config_text(yaml.safe_dump(source)).channel.message_identity.attribute_name is None
    source["channel"]["url"] = "https://example.com/?variant=success"  # type: ignore[index]
    with pytest.raises(ConfigurationError, match="controlled fixture URL"):
        load_config_text(yaml.safe_dump(source))

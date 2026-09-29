"""Native Chromium verification is required before publishing a guide."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from typing import Any

import pytest
import yaml

from syntheticcx.config.loader import load_config, load_discovery_draft
from syntheticcx.discovery.explore import DiscoveryCandidate
from syntheticcx.discovery.publish import publish_guide
from syntheticcx.discovery.verify import verify_candidate
from syntheticcx.errors import ChannelError, ConfigurationError

SERVER = Path("examples/telecom/webchat_fixture/server.py").resolve()
spec = importlib.util.spec_from_file_location("guide_fixture_server", SERVER)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
FixtureServer = module.FixtureServer


def draft(tmp_path: Path, url: str, *, view: str = "main") -> tuple[Path, DiscoveryCandidate]:
    document: dict[str, Any] = yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())
    channel = document.pop("channel")
    channel["url"] = url
    if view == "frame":
        channel["frame_locator"] = {"strategy": "test_id", "value": "chat-frame"}
    document["execution"]["response_timeout_seconds"] = 1
    document["execution"]["greeting_grace_seconds"] = 1
    document["execution"]["browser_action_timeout_seconds"] = 3
    path = tmp_path / "draft.yaml"
    path.write_text(yaml.safe_dump(document))
    return path, DiscoveryCandidate(channel=channel, open_steps=1)


@pytest.mark.parametrize(
    ("variant", "view"),
    [("success", "main"), ("streamed_response", "main"), ("success", "frame")],
)
def test_verified_channel_guide_is_runnable_configuration(
    tmp_path: Path, variant: str, view: str
) -> None:
    with FixtureServer() as fixture:
        url = f"{fixture.base_url}/?variant={variant}&view={view}"
        path, candidate = draft(tmp_path, url, view=view)
        limits = load_discovery_draft(path).execution
        guide = asyncio.run(verify_candidate(candidate, limits, "Synthetic test: please reply."))
        assert guide.validation.status == "verified"
        output = tmp_path / "guides" / "chat.yaml"
        completed = publish_guide(path, output, guide, in_place=False)
        config = load_config(completed)
        assert str(config.channel.url) == url
        assert path.exists() and "channel_guide" not in path.read_text()


@pytest.mark.parametrize("variant", ["missing_control", "silent_response"])
def test_failed_probe_does_not_write_a_guide(tmp_path: Path, variant: str) -> None:
    with FixtureServer() as fixture:
        path, candidate = draft(tmp_path, f"{fixture.base_url}/?variant={variant}")
        with pytest.raises(ChannelError):
            asyncio.run(
                verify_candidate(
                    candidate, load_discovery_draft(path).execution, "Synthetic test: please reply."
                )
            )
    assert not list(tmp_path.glob("*guide*"))


def test_publisher_preserves_existing_output_and_backs_up_in_place(tmp_path: Path) -> None:
    with FixtureServer() as fixture:
        path, candidate = draft(tmp_path, f"{fixture.base_url}/?variant=success")
        guide = asyncio.run(
            verify_candidate(candidate, load_discovery_draft(path).execution, "Synthetic test")
        )
    output = tmp_path / "guide.yaml"
    output.write_text("already here")
    original = path.read_bytes()
    with pytest.raises(ConfigurationError, match="exists"):
        publish_guide(path, output, guide, in_place=True)
    assert path.read_bytes() == original and output.read_text() == "already here"
    output.unlink()
    assert publish_guide(path, output, guide, in_place=True) == path
    assert (tmp_path / "draft.yaml.bak").read_bytes() == original
    assert load_config(path).channel.type == "webchat_playwright"

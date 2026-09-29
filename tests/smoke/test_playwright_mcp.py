"""Opt-in managed MCP transport smoke against the local controlled fixture."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import shutil
from importlib.util import find_spec
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from syntheticcx.adapters.discovery.playwright_mcp import ManagedPlaywrightMCP
from syntheticcx.config.loader import load_config, load_discovery_draft
from syntheticcx.config.models import PrivacyConfig
from syntheticcx.discovery.explore import explore
from syntheticcx.discovery.publish import publish_guide
from syntheticcx.discovery.verify import verify_candidate
from syntheticcx.domain.models import LLMRequest, LLMResponse
from syntheticcx.privacy import Redactor


@pytest.mark.skipif(
    os.environ.get("SYNTHETICCX_RUN_MCP_SMOKE") != "1"
    or find_spec("mcp") is None
    or shutil.which("npx") is None,
    reason="set SYNTHETICCX_RUN_MCP_SMOKE=1 and install discovery prerequisites",
)
def test_managed_mcp_opens_local_chat_without_writing_a_guide() -> None:
    server = Path(__file__).resolve().parents[2] / "examples/telecom/webchat_fixture/server.py"
    spec = importlib.util.spec_from_file_location("discovery_fixture_server", server)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixture_server = module.FixtureServer

    async def inspect(url: str) -> None:
        async with ManagedPlaywrightMCP(url, timeout_seconds=15) as browser:
            snapshot = await browser.navigate(url)
            assert 'button "Open chat" [ref=e5]' in snapshot
            opened = await browser.click("e5", "Open chat", snapshot)
            assert 'textbox "Message"' in opened
        assert browser._session is None

    with fixture_server() as fixture:
        asyncio.run(inspect(f"{fixture.base_url}/?variant=success"))


@pytest.mark.skipif(
    os.environ.get("SYNTHETICCX_RUN_MCP_SMOKE") != "1"
    or find_spec("mcp") is None
    or shutil.which("npx") is None,
    reason="set SYNTHETICCX_RUN_MCP_SMOKE=1 and install discovery prerequisites",
)
def test_mcp_model_candidate_native_verification_and_import(tmp_path: Path) -> None:
    server = Path(__file__).resolve().parents[2] / "examples/telecom/webchat_fixture/server.py"
    spec = importlib.util.spec_from_file_location("discovery_pipeline_fixture", server)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Model:
        def __init__(self, url: str) -> None:
            source: dict[str, Any] = yaml.safe_load(
                Path("examples/telecom/scenario.yaml").read_text()
            )
            source["channel"]["url"] = "__SYNTHETICCX_TARGET_URL__"
            self.channel = source["channel"]
            self.turn = 0

        async def generate(self, request: LLMRequest) -> LLMResponse:
            self.turn += 1
            assert request.response_schema is not None
            if self.turn == 1:
                payload = {
                    "action": "open_chat",
                    "target": "e5",
                    "element": "Open chat",
                    "channel_json": "",
                }
            else:
                assert "textbox" in request.messages[-1].content
                payload = {
                    "action": "propose",
                    "target": "",
                    "element": "",
                    "channel_json": json.dumps(self.channel),
                }
            return cast(LLMResponse, type("Response", (), {"content": json.dumps(payload)})())

    async def inspect(url: str, path: Path) -> None:
        async with ManagedPlaywrightMCP(url, timeout_seconds=15) as browser:
            candidate = await explore(url, browser, Model(url), Redactor(PrivacyConfig()))
        guide = await verify_candidate(
            candidate, load_discovery_draft(path).execution, "Synthetic test: please reply."
        )
        saved = tmp_path / "guide.yaml"
        completed = publish_guide(path, saved, guide, in_place=False)
        assert load_config(completed).channel == candidate.channel

    with module.FixtureServer() as fixture:
        url = f"{fixture.base_url}/?variant=success"
        document: dict[str, Any] = yaml.safe_load(
            Path("examples/telecom/scenario.yaml").read_text()
        )
        document.pop("channel")
        draft = tmp_path / "draft.yaml"
        draft.write_text(yaml.safe_dump(document))
        asyncio.run(inspect(url, draft))

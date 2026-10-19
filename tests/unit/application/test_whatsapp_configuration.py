"""Reject unsafe WhatsApp operational setup before any journey is started."""

import asyncio
from pathlib import Path

import pytest
import yaml

from syntheticcx.application.run import execute_run
from syntheticcx.errors import ConfigurationError


def whatsapp_document():
    document = yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())
    document["channel"] = {
        "type": "whatsapp_web_playwright",
        "url": "https://telecom.example/support",
        "navigation_steps": [
            {
                "action": "click",
                "locator": {"strategy": "test_id", "value": "whatsapp-support"},
            }
        ],
    }
    return document


def test_whatsapp_requires_an_explicit_profile(tmp_path):
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(whatsapp_document()))
    with pytest.raises(ConfigurationError, match="browser-profile"):
        asyncio.run(execute_run(path))


def test_whatsapp_requires_recipient_binding_even_with_profile(tmp_path, monkeypatch):
    monkeypatch.delenv("SYNTHETICCX_WHATSAPP_RECIPIENT", raising=False)
    profile = tmp_path / "dedicated-profile"
    profile.mkdir()
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(whatsapp_document()))
    with pytest.raises(ConfigurationError) as caught:
        asyncio.run(execute_run(path, browser_profile=profile))
    assert caught.value.category == "configuration"
    assert "recipient" in str(caught.value).lower()


def test_whatsapp_refuses_account_wide_recording(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNTHETICCX_WHATSAPP_RECIPIENT", "15551234567")
    profile = tmp_path / "dedicated-profile"
    profile.mkdir()
    document = whatsapp_document()
    document["privacy"].update(allow_screenshots=True, confirm_screenshot_target_safe=True)
    document["storage"]["store_screenshots"] = True
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ConfigurationError, match="screenshots"):
        asyncio.run(execute_run(path, browser_profile=profile))


@pytest.mark.parametrize("recipient", ["", "PRIVATE-INVALID-RECIPIENT"])
def test_cli_rejects_invalid_override_without_environment_fallback_or_evidence(
    tmp_path,
    monkeypatch,
    recipient,
):
    from typer.testing import CliRunner

    from syntheticcx.cli import app

    monkeypatch.setenv("SYNTHETICCX_WHATSAPP_RECIPIENT", "15551234567")
    profile = tmp_path / "dedicated-profile"
    profile.mkdir()
    document = whatsapp_document()
    root = tmp_path / "runs"
    document["storage"]["run_root"] = str(root)
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(document))
    result = CliRunner().invoke(
        app,
        [
            "run",
            str(path),
            "--browser-profile",
            str(profile),
            "--recipient",
            recipient,
        ],
    )
    assert result.exit_code == 2
    assert not root.exists()
    if recipient:
        assert recipient not in result.output


@pytest.mark.parametrize("command", ["discover", "run"])
def test_cli_rejects_recipient_for_webchat(tmp_path, command):
    from typer.testing import CliRunner

    from syntheticcx.cli import app

    path = tmp_path / "scenario.yaml"
    document = yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())
    root = tmp_path / "runs"
    document["storage"]["run_root"] = str(root)
    path.write_text(yaml.safe_dump(document))
    args = [command, str(path), "--recipient", "15551234567"]
    output = tmp_path / "guide.yaml"
    if command == "discover":
        args += ["--url", "https://service.example/", "--output", str(output)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 2
    assert "15551234567" not in result.output
    assert not root.exists()
    assert not output.exists()

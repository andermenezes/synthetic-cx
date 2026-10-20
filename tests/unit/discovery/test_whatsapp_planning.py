"""WhatsApp operational prerequisites fail before browser or model allocation."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from syntheticcx.discovery import whatsapp as discovery
from syntheticcx.errors import ConfigurationError


@pytest.fixture
def draft(tmp_path: Path) -> Path:
    document = yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())
    document.pop("channel")
    path = tmp_path / "draft.yaml"
    path.write_text(yaml.safe_dump(document))
    return path


def test_credentials_in_target_are_rejected_without_resources(
    draft: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        discovery, "OpenAIProvider", lambda *_args, **_kwargs: pytest.fail("model allocated")
    )
    with pytest.raises(ConfigurationError, match="URL credentials") as error:
        asyncio.run(
            discovery.discover_whatsapp_guide(
                draft, "https://user:PRIVATE-CANARY@service.example", browser_profile=tmp_path
            )
        )
    assert "PRIVATE-CANARY" not in str(error.value)


@pytest.mark.parametrize("recipient", [None, "123", "1555-not-phone", "１２３４５６"])
def test_invalid_recipient_fails_before_preflight_or_resource_allocation(
    draft: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recipient: str | None
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    if recipient is None:
        monkeypatch.delenv("SYNTHETICCX_WHATSAPP_RECIPIENT", raising=False)
    else:
        monkeypatch.setenv("SYNTHETICCX_WHATSAPP_RECIPIENT", recipient)
    monkeypatch.setattr(
        discovery, "OpenAIProvider", lambda *_args, **_kwargs: pytest.fail("model allocated")
    )
    with pytest.raises(ConfigurationError):
        asyncio.run(
            discovery.discover_whatsapp_guide(
                draft, "https://service.example/", browser_profile=tmp_path
            )
        )


@pytest.mark.parametrize("capture", ["store_screenshots", "store_video"])
def test_account_capture_is_rejected_before_resources(
    draft: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capture: str
) -> None:
    document = yaml.safe_load(draft.read_text())
    document.setdefault("storage", {})[capture] = True
    document.setdefault("privacy", {}).update(
        allow_screenshots=True, confirm_screenshot_target_safe=True, allow_video=True
    )
    draft.write_text(yaml.safe_dump(document))
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("SYNTHETICCX_WHATSAPP_RECIPIENT", "15551234567")
    monkeypatch.setattr(
        discovery, "OpenAIProvider", lambda *_args, **_kwargs: pytest.fail("model allocated")
    )
    with pytest.raises(ConfigurationError, match="account-wide"):
        asyncio.run(
            discovery.discover_whatsapp_guide(
                draft, "https://service.example/", browser_profile=tmp_path
            )
        )


@pytest.mark.parametrize("attempts", [0, 6])
def test_attempt_bound_fails_before_loading_scenario(tmp_path: Path, attempts: int) -> None:
    with pytest.raises(ConfigurationError, match="between 1 and 5"):
        asyncio.run(
            discovery.discover_whatsapp_guide(
                tmp_path / "not-a-scenario.yaml",
                "https://service.example/",
                browser_profile=tmp_path,
                max_discovery_attempts=attempts,
            )
        )


@pytest.mark.parametrize("invalid", ["missing-profile", "empty-probe"])
def test_operational_prerequisites_fail_before_model_allocation(
    draft: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("SYNTHETICCX_WHATSAPP_RECIPIENT", "15551234567")
    monkeypatch.setattr(
        discovery, "OpenAIProvider", lambda *_args, **_kwargs: pytest.fail("model allocated")
    )
    profile = tmp_path / "does-not-exist" if invalid == "missing-profile" else tmp_path
    with pytest.raises(ConfigurationError):
        asyncio.run(
            discovery.discover_whatsapp_guide(
                draft,
                "https://service.example/",
                browser_profile=profile,
                probe_message=" " if invalid == "empty-probe" else "Owned probe",
            )
        )

"""Discovery composition keeps credentials and preflight outside the browser."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import yaml

from syntheticcx.discovery import run as discovery
from syntheticcx.discovery.explore import DiscoveryCandidate
from syntheticcx.errors import ConfigurationError


def draft(tmp_path: Path) -> Path:
    source: dict[str, Any] = yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())
    source.pop("channel")
    path = tmp_path / "draft.yaml"
    path.write_text(yaml.safe_dump(source))
    return path


def test_missing_selected_credential_never_starts_browser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = draft(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        discovery,
        "ManagedPlaywrightMCP",
        lambda *_args, **_kwargs: pytest.fail("browser constructed before credential resolution"),
    )
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        asyncio.run(discovery.discover_candidate(path, "http://127.0.0.1:8000/"))


def test_override_uses_only_explicit_credential_and_preflights_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = draft(tmp_path)
    override = tmp_path / "model.yaml"
    override.write_text(
        yaml.safe_dump(
            {
                "provider": "openai_compatible",
                "model": "local",
                "base_url": "http://127.0.0.1:5555/v1",
                "credential_environment_variable": "LOCAL_DISCOVERY_KEY",
                "capabilities": dict.fromkeys(
                    (
                        "chat_completions",
                        "structured_output",
                        "embeddings",
                        "token_usage",
                        "model_version",
                    ),
                    True,
                ),
            }
        )
    )
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-credential")
    monkeypatch.delenv("LOCAL_DISCOVERY_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="LOCAL_DISCOVERY_KEY"):
        asyncio.run(
            discovery.discover_candidate(
                path, "http://127.0.0.1:8000/", discovery_model_config=override
            )
        )

    monkeypatch.setenv("LOCAL_DISCOVERY_KEY", "selected-credential")
    events: list[str] = []

    async def probe(_model: Any, settings: Any, **_kwargs: Any) -> None:
        assert settings.credential.get_secret_value() == "selected-credential"
        events.append("probe")

    class Provider:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            events.append("provider")

        async def close(self) -> None:
            events.append("close")

    class Browser:
        def __init__(self, *_args: Any, **kwargs: Any) -> None:
            assert kwargs["headed"] is True
            assert kwargs["chat_hint"] is None
            events.append("browser")

        async def __aenter__(self) -> Browser:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            events.append("browser.close")

    async def explore(
        _url: str, _browser: Any, _provider: Any, _redactor: Any, **_kwargs: Any
    ) -> DiscoveryCandidate:
        events.append("explore")
        source: dict[str, Any] = yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())
        return DiscoveryCandidate(channel=source["channel"], open_steps=1)

    monkeypatch.setattr(discovery, "_probe_role", probe)
    monkeypatch.setattr(discovery, "OpenAICompatibleProvider", Provider)
    monkeypatch.setattr(discovery, "ManagedPlaywrightMCP", Browser)
    monkeypatch.setattr(discovery, "explore", explore)
    asyncio.run(
        discovery.discover_candidate(
            path,
            "http://127.0.0.1:8000/",
            discovery_model_config=override,
            headed=True,
        )
    )
    assert events == ["probe", "provider", "browser", "explore", "browser.close", "close"]


def test_invalid_model_override_is_safe(tmp_path: Path) -> None:
    override = tmp_path / "model.yaml"
    override.write_text("provider: unknown\ncredential: canary-secret\n")
    with pytest.raises(ConfigurationError) as caught:
        discovery._selected_model(override, yaml.safe_load("provider: openai\nmodel: x\n"))  # type: ignore[arg-type]
    assert "canary-secret" not in str(caught.value)

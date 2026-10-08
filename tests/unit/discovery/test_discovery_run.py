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


def test_autonomous_owns_one_learning_and_one_fresh_broker(tmp_path, monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from syntheticcx.config import sanitize
    from syntheticcx.discovery import learn as learning
    from syntheticcx.discovery import replay as replaying

    path = draft(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "selected-secret")
    provider = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(discovery, "OpenAIProvider", lambda *args, **kwargs: provider)
    monkeypatch.setattr(discovery, "_probe_role", AsyncMock())
    first = SimpleNamespace(start=AsyncMock(), close=AsyncMock())
    fresh = SimpleNamespace(start=AsyncMock(), close=AsyncMock())
    factory = Mock(side_effect=[first, fresh])
    module = ModuleType("syntheticcx.adapters.channel.broker")
    module.NativeChatBroker = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    contract = object()
    learned = SimpleNamespace(config=contract)
    learner = AsyncMock(return_value=learned)
    guide = object()
    replayer = AsyncMock(return_value=guide)
    publication_check = Mock()
    monkeypatch.setattr(learning, "learn", learner)
    monkeypatch.setattr(replaying, "replay", replayer)
    monkeypatch.setattr(sanitize, "sanitized_channel_guide", publication_check)
    result = asyncio.run(
        discovery.discover_guide(
            path, "http://localhost:8000", language="pt-BR", probe_message="private override"
        )
    )
    assert result is guide
    assert factory.call_count == 2
    assert factory.call_args_list[0].kwargs["contract"] is None
    assert factory.call_args_list[1].kwargs["contract"] is contract
    assert learner.call_args.kwargs["probes"] == {
        "explicit": "private override",
    }
    assert learner.call_args.args[0] is first
    assert replayer.call_args.args[0] is fresh
    assert "selected-secret" in publication_check.call_args.kwargs["resolved_secrets"]
    first.close.assert_awaited_once()
    fresh.close.assert_awaited_once()
    provider.close.assert_awaited_once()


def test_autonomous_uncertain_learning_never_starts_replay(tmp_path, monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from syntheticcx.discovery import learn as learning
    from syntheticcx.errors import ChannelError

    path = draft(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "selected-secret")
    provider = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(discovery, "OpenAIProvider", lambda *args, **kwargs: provider)
    monkeypatch.setattr(discovery, "_probe_role", AsyncMock())
    first = SimpleNamespace(start=AsyncMock(), close=AsyncMock())
    factory = Mock(return_value=first)
    module = ModuleType("syntheticcx.adapters.channel.broker")
    module.NativeChatBroker = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(learning, "learn", AsyncMock(side_effect=ChannelError("send", "uncertain")))
    with pytest.raises(ChannelError):
        asyncio.run(discovery.discover_guide(path, "http://localhost:8000"))
    assert factory.call_count == 1
    first.close.assert_awaited_once()
    provider.close.assert_awaited_once()


def test_autonomous_total_deadline_clips_slow_browser_and_provider_cleanup(tmp_path, monkeypatch):
    import sys
    from time import monotonic
    from types import ModuleType, SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from syntheticcx.errors import ChannelError

    path = draft(tmp_path)
    document = yaml.safe_load(path.read_text())
    document["execution"].update(journey_timeout_seconds=0.06, close_timeout_seconds=0.1)
    path.write_text(yaml.safe_dump(document))
    monkeypatch.setenv("OPENAI_API_KEY", "selected-secret")

    async def slow():
        await asyncio.sleep(10)

    provider = SimpleNamespace(close=AsyncMock(side_effect=slow))
    browser = SimpleNamespace(start=AsyncMock(side_effect=slow), close=AsyncMock(side_effect=slow))
    monkeypatch.setattr(discovery, "OpenAIProvider", lambda *_args, **_kwargs: provider)
    monkeypatch.setattr(discovery, "_probe_role", AsyncMock())
    module = ModuleType("syntheticcx.adapters.channel.broker")
    factory = Mock(return_value=browser)
    module.NativeChatBroker = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    started = monotonic()
    with pytest.raises(ChannelError, match="budget exhausted"):
        asyncio.run(discovery.discover_guide(path, "http://localhost:8000"))
    assert monotonic() - started < 0.13
    browser.close.assert_awaited_once()
    provider.close.assert_awaited_once()
    assert factory.call_count == 1


def test_credential_url_rejected_before_model_or_browser_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        discovery, "OpenAIProvider", lambda *_args, **_kwargs: pytest.fail("model created")
    )
    with pytest.raises(ConfigurationError, match="URL credentials") as error:
        asyncio.run(discovery.discover_guide(draft(tmp_path), "https://user:CANARY@chat.example"))
    assert "CANARY" not in str(error.value)


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
        source: dict[str, Any] = yaml.safe_load(
            Path("tests/fixtures/discovery-sample.yaml").read_text()
        )
        return DiscoveryCandidate(channel=source["channel"], open_steps=1)

    async def calibrate(
        candidate: DiscoveryCandidate, *_args: Any, **kwargs: Any
    ) -> DiscoveryCandidate:
        assert kwargs["headed"] is True
        events.append("calibrate")
        return candidate

    monkeypatch.setattr(discovery, "_probe_role", probe)
    monkeypatch.setattr(discovery, "OpenAICompatibleProvider", Provider)
    monkeypatch.setattr(discovery, "ManagedPlaywrightMCP", Browser)
    monkeypatch.setattr(discovery, "explore", explore)
    monkeypatch.setattr(discovery, "calibrate_candidate", calibrate)
    asyncio.run(
        discovery.discover_candidate(
            path,
            "http://127.0.0.1:8000/",
            discovery_model_config=override,
            headed=True,
        )
    )
    assert events == [
        "probe",
        "provider",
        "browser",
        "explore",
        "browser.close",
        "calibrate",
        "close",
    ]


def test_invalid_model_override_is_safe(tmp_path: Path) -> None:
    override = tmp_path / "model.yaml"
    override.write_text("provider: unknown\ncredential: canary-secret\n")
    with pytest.raises(ConfigurationError) as caught:
        discovery._selected_model(override, yaml.safe_load("provider: openai\nmodel: x\n"))  # type: ignore[arg-type]
    assert "canary-secret" not in str(caught.value)


def test_id_hint_raw_intent_survives_redaction_for_exploration(tmp_path, monkeypatch):
    source = yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())
    source.pop("channel")
    source["scenario"]["knowledge"]["customer_id"] = "customer-123"
    source["privacy"]["redaction_keys"].append("customer_id")
    path = tmp_path / "draft.yaml"
    path.write_text(yaml.safe_dump(source))
    monkeypatch.setenv("OPENAI_API_KEY", "canary-provider-credential")

    class Browser:
        def __init__(self, *args, **kwargs):
            assert kwargs["chat_hint"] == "id=chat-customer-123"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class Provider:
        def __init__(self, *args, **kwargs):
            pass

        async def close(self):
            pass

    async def probe(*args, **kwargs):
        pass

    async def explored(*args, **kwargs):
        assert kwargs["id_launcher_requested"] is True
        assert "customer-123" not in kwargs["chat_hint"]
        channel = yaml.safe_load(Path("tests/fixtures/discovery-sample.yaml").read_text())[
            "channel"
        ]
        return DiscoveryCandidate(channel=channel, open_steps=1)

    async def calibrated(candidate, *args, **kwargs):
        return candidate

    monkeypatch.setattr(discovery, "_probe_role", probe)
    monkeypatch.setattr(discovery, "ManagedPlaywrightMCP", Browser)
    monkeypatch.setattr(discovery, "OpenAIProvider", Provider)
    monkeypatch.setattr(discovery, "explore", explored)
    monkeypatch.setattr(discovery, "calibrate_candidate", calibrated)
    asyncio.run(
        discovery.discover_candidate(path, "https://example.com/", chat_hint="id=chat-customer-123")
    )

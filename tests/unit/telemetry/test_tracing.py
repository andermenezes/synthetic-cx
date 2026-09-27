"""Local, disabled, and isolated operational traces without external transport."""

import asyncio
import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from syntheticcx.config.models import PrivacyConfig, TelemetryConfig
from syntheticcx.privacy import Redactor
from syntheticcx.telemetry.run import RunTelemetry, span


def test_disabled_run_does_not_create_span_artifacts(tmp_path: Path) -> None:
    async def exercise() -> None:
        telemetry = RunTelemetry(
            TelemetryConfig(enabled=False),
            "run-a",
            tmp_path / "run-a",
            Redactor(PrivacyConfig()),
            remote_timeout=1,
        )
        with telemetry.activate(), span("model.call", operation="sensitive-input"):
            pass
        await telemetry.close(created=True, timeout=1)

    asyncio.run(exercise())
    assert not (tmp_path / "run-a").exists()


def test_concurrent_run_contexts_cannot_share_spans(tmp_path: Path) -> None:
    async def exercise(run_id: str) -> None:
        directory = tmp_path / run_id
        directory.mkdir()
        telemetry = RunTelemetry(
            TelemetryConfig(), run_id, directory, Redactor(PrivacyConfig()), remote_timeout=1
        )
        with telemetry.activate(), span("run.initialize"):
            await asyncio.sleep(0)
            with span("model.call", operation="secret prompt should not be recorded"):
                await asyncio.sleep(0)
        await telemetry.close(created=True, timeout=1)

    async def both() -> None:
        await asyncio.gather(exercise("run-a"), exercise("run-b"))

    asyncio.run(both())
    for run_id in ("run-a", "run-b"):
        records = [
            json.loads(line)
            for line in (tmp_path / run_id / "artifacts/telemetry/spans.jsonl")
            .read_text()
            .splitlines()
        ]
        assert [record["name"] for record in records] == ["model.call", "run.initialize"]
        assert len({record["trace_id"] for record in records}) == 1
        assert all(record["attributes"]["run.id"] == run_id for record in records)
        assert "secret prompt" not in json.dumps(records)


def test_local_attributes_redact_selected_secret(tmp_path: Path) -> None:
    async def exercise() -> None:
        directory = tmp_path / "run-a"
        directory.mkdir()
        telemetry = RunTelemetry(
            TelemetryConfig(),
            "run-a",
            directory,
            Redactor(PrivacyConfig(), resolved_secrets=("token-canary",)),
            remote_timeout=1,
        )
        with telemetry.activate(), span("model.call", operation="token-canary"):
            pass
        await telemetry.close(created=True, timeout=1)

    asyncio.run(exercise())
    content = (tmp_path / "run-a/artifacts/telemetry/spans.jsonl").read_text()
    assert "token-canary" not in content
    assert "operation" not in json.loads(content)["attributes"]


def test_local_and_disabled_modes_never_import_optional_otlp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_import = builtins.__import__

    def deny_otlp(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("opentelemetry.exporter.otlp"):
            raise AssertionError("OTLP imported without remote export")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", deny_otlp)

    async def exercise() -> None:
        for enabled in (False, True):
            telemetry = RunTelemetry(
                TelemetryConfig(enabled=enabled),
                f"run-{enabled}",
                tmp_path / f"run-{enabled}",
                Redactor(PrivacyConfig()),
                remote_timeout=1,
            )
            await telemetry.close(created=False, timeout=1)

    asyncio.run(exercise())


def test_remote_exporter_is_loaded_only_after_explicit_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_import = builtins.__import__
    constructed: list[dict[str, object]] = []

    class FakeOTLP:
        def __init__(self, **kwargs: object) -> None:
            constructed.append(kwargs)

        def export(self, _spans: object) -> object:
            from opentelemetry.sdk.trace.export import SpanExportResult

            return SpanExportResult.SUCCESS

        def shutdown(self) -> None:
            pass

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            return True

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "opentelemetry.exporter.otlp.proto.http.trace_exporter":
            return SimpleNamespace(OTLPSpanExporter=FakeOTLP)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    async def exercise() -> None:
        directory = tmp_path / "run-a"
        directory.mkdir()
        telemetry = RunTelemetry(
            TelemetryConfig(
                enabled=True, remote_export=True, otlp_endpoint="http://localhost:4318/v1/traces"
            ),
            "run-a",
            directory,
            Redactor(PrivacyConfig()),
            remote_timeout=1,
        )
        with telemetry.activate(), span("run.initialize"):
            pass
        await telemetry.close(created=True, timeout=1)

    asyncio.run(exercise())
    assert constructed == [
        {
            "endpoint": "http://localhost:4318/v1/traces",
            "headers": {"x-syntheticcx-client": "syntheticcx"},
            "timeout": 1,
        }
    ]

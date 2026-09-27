"""CLI composition through real storage and evaluation with controlled adapters."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from syntheticcx.adapters.llm.preflight import CapabilityPreflight
from syntheticcx.application import run as application
from syntheticcx.cli import app
from syntheticcx.config import settings as config_settings
from syntheticcx.domain.models import (
    CompletedResponse,
    ConversationObservation,
    CustomerScenarioContext,
    EvaluationResult,
    Message,
    PersonaAction,
    PersonaDefinition,
    PersonaState,
    SendReceipt,
)
from syntheticcx.errors import ConfigurationError
from syntheticcx.evaluation.service import EvaluationService


class FakeChannel:
    def __init__(self, *_args: Any) -> None:
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def send(self, _message: str) -> SendReceipt:
        return SendReceipt(message_id="sent-1", submitted_at=application.RealClock().now())

    async def receive(self) -> CompletedResponse:
        now = application.RealClock().now()
        return CompletedResponse(
            messages=[
                Message(
                    id="assistant-1",
                    sequence=1,
                    actor="assistant",
                    content="A plan change.",
                    observed_at=now,
                )
            ],
            started_at=now,
            completed_at=now,
            completion_strategy="completion_locator",
        )

    async def close(self) -> None:
        self.closed = True


class FakePersona:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.closed = False

    async def initialize(
        self, definition: PersonaDefinition, _context: CustomerScenarioContext
    ) -> PersonaState:
        return definition.fresh_state()

    async def respond(
        self, _observation: ConversationObservation, state: PersonaState
    ) -> PersonaAction:
        if state.turn_number:
            return PersonaAction(state=state, signal="goal_believed_complete")
        return PersonaAction(message="Why did my invoice increase?", state=state, signal="continue")

    async def close(self) -> None:
        self.closed = True


class FakeProvider:
    async def close(self) -> None:
        pass


class GoalEvaluator:
    def __init__(self, status: str) -> None:
        self.status = status

    async def evaluate(self, context: Any) -> EvaluationResult:
        assistant = next(
            event
            for event in context.sanitized_trace
            if event.type == "message_observed" and event.data.message.actor == "assistant"
        )
        return EvaluationResult(
            evaluator_id="semantic_goal_completion",
            criterion_id=context.criterion.id,
            required=True,
            status=self.status,
            explanation="Controlled billing verdict.",
            evidence_event_ids=[assistant.id],
            evidence_message_ids=[assistant.data.message.id],
        )


def scenario_file(tmp_path: Path, *, report: bool = True) -> Path:
    source = yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())
    source["execution"]["greeting_grace_seconds"] = 0
    source["storage"]["run_root"] = str(tmp_path / "runs")
    source["storage"]["materialize_report"] = report
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("verdict", "code", "report_enabled"),
    [("passed", 0, True), ("failed", 1, True), ("error", 3, True), ("passed", 0, False)],
)
def test_cli_composes_run_and_renders_committed_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verdict: str, code: int, report_enabled: bool
) -> None:
    path = scenario_file(tmp_path, report=report_enabled)
    monkeypatch.setenv("OPENAI_API_KEY", "canary-credential")

    async def preflight(*_args: Any) -> CapabilityPreflight:
        return CapabilityPreflight()

    channel = FakeChannel()
    persona = FakePersona()
    monkeypatch.setattr(application, "preflight_models", preflight)
    monkeypatch.setattr(application, "PlaywrightChannel", lambda *_args: channel)
    monkeypatch.setattr(
        application, "TinyTroupePersonaSimulator", lambda *_args, **_kwargs: persona
    )
    monkeypatch.setattr(application, "OpenAIProvider", lambda *_args, **_kwargs: FakeProvider())

    def service(**kwargs: Any) -> EvaluationService:
        return EvaluationService(
            **kwargs, evaluators={"semantic_goal_completion": GoalEvaluator(verdict)}
        )

    monkeypatch.setattr(application, "EvaluationService", service)
    output = CliRunner().invoke(app, ["run", str(path)])
    assert output.exit_code == code, output.output
    assert channel.started and channel.closed and persona.closed
    directories = list((tmp_path / "runs").iterdir())
    assert len(directories) == 1
    directory = directories[0]
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["outcome"] == verdict
    assert f"Run ID: {directory.name}" in output.output
    assert f"Outcome: {verdict}" in output.output
    assert "Turns: 1" in output.output
    assert "canary-credential" not in output.output
    spans = [
        json.loads(line)
        for line in (directory / "artifacts/telemetry/spans.jsonl").read_text().splitlines()
    ]
    names = {item["name"] for item in spans}
    assert {
        "run.initialize",
        "persistence.create",
        "browser.start",
        "conversation.run",
        "conversation.turn",
        "persona.initialize",
        "persona.update",
        "model.call",
        "evaluation.run",
        "evaluation.criterion",
        "persistence.append",
        "persistence.finalize",
        "resource.cleanup",
    } <= names
    assert len({item["trace_id"] for item in spans}) == 1
    assert all(item["attributes"]["run.id"] == directory.name for item in spans)
    assert "canary-credential" not in (directory / "artifacts/telemetry/spans.jsonl").read_text()
    assert "trace_id=" in output.output and f"run_id={directory.name}" in output.output
    if report_enabled:
        report = json.loads((directory / "report.json").read_text())
        errors = ", ".join(
            f"{error['category']} ({error['operation']}): {error['message']}"
            for error in report["errors"]
        )
        for field, value in [
            ("Outcome", report["outcome"]),
            ("Termination", report["termination_reason"]),
            ("Goal", report["goal_result"]["status"]),
            ("Turns", report["turn_count"]),
            ("Duration (seconds)", report["duration_seconds"]),
            ("Repetition findings", report["repetition_count"]),
            ("Errors", errors or "none"),
            ("Artifacts", report["artifact_location"]),
        ]:
            assert f"{field}: {value}" in output.output
    else:
        assert not (directory / "report.json").exists()
        assert "report.json: omitted" in output.output


def test_invalid_configuration_does_not_probe_or_create_a_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = scenario_file(tmp_path)
    source = yaml.safe_load(path.read_text())
    source["execution"]["max_turns"] = 0
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    monkeypatch.setattr(
        application, "preflight_models", lambda *_args: pytest.fail("unexpected provider probe")
    )
    output = CliRunner().invoke(app, ["run", str(path)])
    assert output.exit_code == 2
    assert "execution.max_turns" in output.output
    assert not (tmp_path / "runs").exists()


def test_capability_failure_is_reported_before_browser_or_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = scenario_file(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "canary-credential")

    async def unavailable(*_args: Any) -> CapabilityPreflight:
        raise ConfigurationError(
            "probe model capabilities", "persona.structured_output is unavailable"
        )

    monkeypatch.setattr(application, "preflight_models", unavailable)
    monkeypatch.setattr(
        application, "PlaywrightChannel", lambda *_args: pytest.fail("browser constructed")
    )
    output = CliRunner().invoke(app, ["run", str(path)])
    assert output.exit_code == 2
    assert "persona.structured_output" in output.output
    assert "Run ID:" not in output.output
    assert not (tmp_path / "runs").exists()


def test_run_id_is_announced_only_after_initial_metadata_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = scenario_file(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "canary-credential")

    async def preflight(*_args: Any) -> CapabilityPreflight:
        return CapabilityPreflight()

    class FailingChannel(FakeChannel):
        async def start(self) -> None:
            assert list((tmp_path / "runs").glob("*/metadata.json"))
            raise OSError("hostile secret: canary-credential")

    channel = FailingChannel()
    monkeypatch.setattr(application, "preflight_models", preflight)
    monkeypatch.setattr(application, "PlaywrightChannel", lambda *_args: channel)
    monkeypatch.setattr(
        application, "TinyTroupePersonaSimulator", lambda *_args, **_kwargs: FakePersona()
    )
    monkeypatch.setattr(application, "OpenAIProvider", lambda *_args, **_kwargs: FakeProvider())
    output = CliRunner().invoke(app, ["run", str(path)])
    assert output.exit_code == 3
    assert channel.closed
    assert "Run ID:" in output.output
    assert "canary-credential" not in output.output
    assert "Outcome: error" in output.output
    directory = next((tmp_path / "runs").iterdir())
    assert json.loads((directory / "metadata.json").read_text())["outcome"] == "error"


@pytest.mark.parametrize(
    ("failure", "expected_code", "expected_text", "finalized"),
    [
        ("cancel", 130, "Outcome: cancelled", True),
        ("storage", 3, "storage failure; final report unavailable", False),
    ],
)
def test_cli_reports_cancelled_and_storage_failure_without_claiming_a_final_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    expected_code: int,
    expected_text: str,
    finalized: bool,
) -> None:
    path = scenario_file(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "canary-credential")

    async def preflight(*_args: Any) -> CapabilityPreflight:
        return CapabilityPreflight()

    class InterruptingChannel(FakeChannel):
        async def start(self) -> None:
            if failure == "cancel":
                raise asyncio.CancelledError
            await super().start()

    class FailingStorage(application.FilesystemStorage):
        async def create_run(self, *args: Any, **kwargs: Any) -> None:
            if failure == "storage":
                raise OSError("hostile secret: canary-credential")
            await super().create_run(*args, **kwargs)

    monkeypatch.setattr(application, "preflight_models", preflight)
    monkeypatch.setattr(application, "PlaywrightChannel", lambda *_args: InterruptingChannel())
    monkeypatch.setattr(
        application, "TinyTroupePersonaSimulator", lambda *_args, **_kwargs: FakePersona()
    )
    monkeypatch.setattr(application, "OpenAIProvider", lambda *_args, **_kwargs: FakeProvider())
    monkeypatch.setattr(application, "FilesystemStorage", FailingStorage)
    output = CliRunner().invoke(app, ["run", str(path)])
    assert output.exit_code == expected_code, output.output
    assert expected_text in output.output
    assert ("Outcome:" in output.output) is finalized
    assert "canary-credential" not in output.output


def test_telemetry_shutdown_failure_does_not_overwrite_final_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = scenario_file(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "canary-credential")

    async def preflight(*_args: Any) -> CapabilityPreflight:
        return CapabilityPreflight()

    async def broken_close(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("hostile token: canary-credential")

    monkeypatch.setattr(application, "preflight_models", preflight)
    monkeypatch.setattr(application, "PlaywrightChannel", lambda *_args: FakeChannel())
    monkeypatch.setattr(
        application, "TinyTroupePersonaSimulator", lambda *_args, **_kwargs: FakePersona()
    )
    monkeypatch.setattr(application, "OpenAIProvider", lambda *_args, **_kwargs: FakeProvider())
    monkeypatch.setattr(application.RunTelemetry, "close", broken_close)

    def service(**kwargs: Any) -> EvaluationService:
        return EvaluationService(
            **kwargs, evaluators={"semantic_goal_completion": GoalEvaluator("passed")}
        )

    monkeypatch.setattr(application, "EvaluationService", service)
    output = CliRunner().invoke(app, ["run", str(path)])
    assert output.exit_code == 0
    assert "Outcome: passed" in output.output
    assert "Telemetry shutdown failed" in output.output
    assert "canary-credential" not in output.output
    directory = next((tmp_path / "runs").iterdir())
    assert json.loads((directory / "metadata.json").read_text())["outcome"] == "passed"


def test_remote_telemetry_setup_failure_is_a_configuration_error_before_browser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = scenario_file(tmp_path)
    source = yaml.safe_load(path.read_text())
    source["telemetry"] = {
        "enabled": True,
        "remote_export": True,
        "otlp_endpoint": "http://localhost:4318/v1/traces",
    }
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "canary-credential")
    monkeypatch.setattr(config_settings, "version", lambda _package: "1.44.0")

    async def preflight(*_args: Any) -> CapabilityPreflight:
        return CapabilityPreflight()

    def unavailable_telemetry(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("unavailable telemetry exporter")

    monkeypatch.setattr(application, "preflight_models", preflight)
    monkeypatch.setattr(application, "RunTelemetry", unavailable_telemetry)
    monkeypatch.setattr(
        application, "PlaywrightChannel", lambda *_args: pytest.fail("browser constructed")
    )
    output = CliRunner().invoke(app, ["run", str(path)])
    assert output.exit_code == 2
    assert "remote telemetry exporter could not be configured" in output.output
    assert not (tmp_path / "runs").exists()


def test_unexpected_owner_failure_returns_safe_partial_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = scenario_file(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "canary-credential")

    async def preflight(*_args: Any) -> CapabilityPreflight:
        return CapabilityPreflight()

    async def unexpected_owner(**_kwargs: Any) -> None:
        raise RuntimeError("hostile owner failure: canary-credential")

    channel = FakeChannel()
    persona = FakePersona()
    monkeypatch.setattr(application, "preflight_models", preflight)
    monkeypatch.setattr(application, "PlaywrightChannel", lambda *_args: channel)
    monkeypatch.setattr(
        application, "TinyTroupePersonaSimulator", lambda *_args, **_kwargs: persona
    )
    monkeypatch.setattr(application, "OpenAIProvider", lambda *_args, **_kwargs: FakeProvider())
    monkeypatch.setattr(application, "run_evaluated", unexpected_owner)
    output = CliRunner().invoke(app, ["run", str(path)])
    assert output.exit_code == 3
    assert "Status: final report unavailable" in output.output
    assert "canary-credential" not in output.output
    assert channel.closed and persona.closed

"""CLI composition through real storage and evaluation with controlled adapters."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from syntheticcx.adapters.channel import dynamic as channel_module
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
    ModelIdentity,
    ModelInteraction,
    PersonaAction,
    PersonaDefinition,
    PersonaState,
    SendReceipt,
    TokenUsage,
)
from syntheticcx.errors import ConfigurationError
from syntheticcx.evaluation.service import EvaluationService


class FakeChannel:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
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


def offline_adapters(monkeypatch):
    async def preflight(*_args):
        return CapabilityPreflight()

    monkeypatch.setenv("OPENAI_API_KEY", "offline-credential")
    monkeypatch.setattr(application, "preflight_models", preflight)
    monkeypatch.setattr(channel_module, "DynamicPlaywrightChannel", FakeChannel)
    monkeypatch.setattr(application, "TinyTroupePersonaSimulator", FakePersona)


@pytest.mark.parametrize("kind", ["boolean", "scored"])
def test_registered_evaluator_runs_with_its_own_identity(tmp_path, monkeypatch, kind):
    from pydantic import BaseModel, Field

    from syntheticcx.config.models import StrictModel
    from syntheticcx.evaluation.registry import EvaluatorRegistration

    class Parameters(StrictModel):
        minimum: int = Field(default=1, ge=1, strict=True)

    constructed = []

    class CustomEvaluator:
        def __init__(self, parameters: BaseModel):
            constructed.append(parameters)

        async def evaluate(self, context):
            assert context.criterion.parameters == {"minimum": 1}
            assistant = next(
                event
                for event in context.sanitized_trace
                if event.type == "message_observed" and event.data.message.actor == "assistant"
            )
            return EvaluationResult(
                evaluator_id="answer_policy",
                criterion_id=context.criterion.id,
                required=context.criterion.required,
                status="passed",
                score=0.9 if kind == "scored" else None,
                explanation="Current answer satisfies the registered policy.",
                evidence_event_ids=[assistant.id],
                evidence_message_ids=[assistant.data.message.id],
            )

    offline_adapters(monkeypatch)
    path = scenario_file(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["scenario"]["success_criteria"] = [
        {
            "id": "answer-policy",
            "evaluator_type": "answer_policy",
            "result_kind": kind,
            "evidence_policy": "required",
            **({"threshold": 0.8} if kind == "scored" else {}),
        }
    ]
    path.write_text(yaml.safe_dump(data))
    registration = EvaluatorRegistration(
        "answer_policy",
        Parameters,
        lambda parameters, dependencies: CustomEvaluator(parameters),
        ("boolean", "scored"),
    )
    result = asyncio.run(
        application.execute_run(path, evaluator_registry={"answer_policy": registration})
    )
    assert result.exit_code == 0 and result.report.outcome == "passed"
    assert len(constructed) == 1
    directory = Path(result.artifact_location)
    evaluation = json.loads((directory / "evaluation.json").read_text())
    assert "answer_policy" in json.dumps(evaluation)
    config = yaml.safe_load((directory / "config.yaml").read_text())
    assert config["effective_config"]["scenario"]["success_criteria"][0]["parameters"] == {
        "minimum": 1
    }


def test_storage_shutdown_failure_is_visible_without_rewriting_report(tmp_path, monkeypatch):
    offline_adapters(monkeypatch)
    path = scenario_file(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["scenario"]["success_criteria"] = [
        {
            "id": "turns",
            "evaluator_type": "turn_count",
            "result_kind": "boolean",
            "evidence_policy": "required",
            "parameters": {"max_turns": 5},
        }
    ]
    path.write_text(yaml.safe_dump(data))
    original = application.FilesystemStorage

    class FailingClose(original):
        async def close(self):
            await super().close()
            raise OSError("private-shutdown-value")

    monkeypatch.setattr(application, "FilesystemStorage", FailingClose)
    result = CliRunner().invoke(app, ["run", str(path)])
    assert result.exit_code == 3
    assert "Infrastructure shutdown failure (storage.close)" in result.output
    assert "before this failure" in result.output
    assert "private-shutdown-value" not in result.output
    directory = next((tmp_path / "runs").iterdir())
    assert json.loads((directory / "report.json").read_text())["outcome"] == "passed"
    assert json.loads((directory / "metadata.json").read_text())["lifecycle_status"] == "finalized"


@pytest.mark.parametrize("markdown_failure", [False, True])
def test_cli_redacts_scenario_fact_recalled_only_in_feedback(
    tmp_path, monkeypatch, markdown_failure
):
    from syntheticcx.domain.models import PersonaFeedbackResponse

    path = scenario_file(tmp_path)
    data = yaml.safe_load(path.read_text())
    sensitive = "private-account-908172"
    data["scenario"]["knowledge"]["account_number"] = sensitive
    data["privacy"]["redaction_keys"].append("account_number")
    data["scenario"]["observation_topics"] = [{"id": "clarity", "description": "Was it clear?"}]
    data["reporting"] = {"markdown": {"enabled": True}}
    data["persona"]["description"] = f"Customer remembers {sensitive}"
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setenv("OPENAI_API_KEY", "canary-credential")

    class ReflectingPersona(FakePersona):
        reflection_available = True

        async def reflect(self, request):
            return PersonaFeedbackResponse.model_validate(
                {
                    "summary": f"I remember {sensitive}.",
                    "topics": [
                        {
                            "topic_id": "clarity",
                            "status": "observed",
                            "impression": f"It was clear for {sensitive}.",
                            "evidence_message_ids": [request.messages[-1].id],
                        }
                    ],
                }
            )

    async def preflight(*args):
        return CapabilityPreflight()

    monkeypatch.setattr(application, "preflight_models", preflight)
    monkeypatch.setattr(channel_module, "DynamicPlaywrightChannel", FakeChannel)
    monkeypatch.setattr(application, "TinyTroupePersonaSimulator", ReflectingPersona)
    monkeypatch.setattr(application, "OpenAIProvider", lambda *args, **kwargs: FakeProvider())
    monkeypatch.setattr(
        application,
        "EvaluationService",
        lambda **kwargs: EvaluationService(
            **kwargs,
            evaluators={"semantic_goal_completion": GoalEvaluator("passed")},
        ),
    )
    if markdown_failure:
        original_write = application.FilesystemStorage._atomic_write

        def fail_markdown(cls, destination, content):
            if destination.name == "report.md":
                raise OSError("sensitive filesystem error")
            original_write(destination, content)

        monkeypatch.setattr(
            application.FilesystemStorage, "_atomic_write", classmethod(fail_markdown)
        )
    output = CliRunner().invoke(app, ["run", str(path)])
    if markdown_failure:
        assert output.exit_code == 3, output.output
        directory = next((tmp_path / "runs").iterdir())
        assert (
            json.loads((directory / "metadata.json").read_text())["lifecycle_status"] != "finalized"
        )
        assert not (directory / "report.md").exists()
        assert "Markdown report:" not in output.output
        return
    assert output.exit_code == 0, output.output
    directory = next((tmp_path / "runs").iterdir())
    for name in ["trace.jsonl", "config.yaml", "report.json", "report.md"]:
        assert sensitive not in (directory / name).read_text()
    assert "Markdown report:" in output.output
    events = [json.loads(line) for line in (directory / "trace.jsonl").read_text().splitlines()]
    feedback = next(
        event["data"] for event in events if event["type"] == "persona_feedback_recorded"
    )
    assert feedback["status"] == "available"


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
    monkeypatch.setattr(
        channel_module, "DynamicPlaywrightChannel", lambda *_args, **_kwargs: channel
    )
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


def test_cli_infrastructure_error_takes_precedence_over_failed_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = scenario_file(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "canary-credential")

    async def preflight(*_args: Any) -> CapabilityPreflight:
        return CapabilityPreflight()

    class FailingCloseChannel(FakeChannel):
        async def close(self) -> None:
            await super().close()
            raise OSError("hostile close: canary-credential")

    channel = FailingCloseChannel()
    monkeypatch.setattr(application, "preflight_models", preflight)
    monkeypatch.setattr(
        channel_module, "DynamicPlaywrightChannel", lambda *_args, **_kwargs: channel
    )
    monkeypatch.setattr(
        application, "TinyTroupePersonaSimulator", lambda *_args, **_kwargs: FakePersona()
    )
    monkeypatch.setattr(application, "OpenAIProvider", lambda *_args, **_kwargs: FakeProvider())

    def service(**kwargs: Any) -> EvaluationService:
        return EvaluationService(
            **kwargs, evaluators={"semantic_goal_completion": GoalEvaluator("failed")}
        )

    monkeypatch.setattr(application, "EvaluationService", service)
    output = CliRunner().invoke(app, ["run", str(path)])
    assert channel.closed
    assert output.exit_code == 3, output.output
    directory = next((tmp_path / "runs").iterdir())
    report = json.loads((directory / "report.json").read_text())
    assert report["outcome"] == "error"
    assert report["goal_result"]["status"] == "failed"
    assert "Outcome: error" in output.output
    assert "Goal: failed" in output.output
    assert "canary-credential" not in output.output


def test_cli_enabled_prompt_capture_is_redacted_in_committed_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = scenario_file(tmp_path)
    source = yaml.safe_load(path.read_text())
    source["privacy"]["capture_prompts"] = True
    source["privacy"]["redaction_keys"].extend(["description", "writing_description"])
    account = 'private-"account\\id'
    credential = 'canary-"credential\\id'
    narrative = 'private-persona-"background\\id'
    writing = 'private-writing-"voice\\id'
    source["persona"]["description"] = narrative
    source["persona"]["communication"].update(
        writing_description=writing, message_length="short", grammar_error_probability=0.15
    )
    source["scenario"]["knowledge"]["account_number"] = account
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", credential)

    async def preflight(*_args: Any) -> CapabilityPreflight:
        return CapabilityPreflight()

    class CapturingPersona(FakePersona):
        def __init__(self, prompt_capture: Any) -> None:
            super().__init__()
            self.capture = prompt_capture
            self.interactions: list[ModelInteraction] = []

        async def initialize(
            self, definition: PersonaDefinition, context: CustomerScenarioContext
        ) -> PersonaState:
            assert definition.description == narrative
            assert definition.communication.writing_description == writing
            return await super().initialize(definition, context)

        async def respond(
            self, observation: ConversationObservation, state: PersonaState
        ) -> PersonaAction:
            now = application.RealClock().now()
            self.interactions.append(
                ModelInteraction(
                    id=f"persona-{len(self.interactions)}",
                    identity=ModelIdentity(provider="openai", model="fake"),
                    started_at=now,
                    ended_at=now,
                    usage=TokenUsage(unavailable_reason="offline fixture"),
                    captured_prompt=self.capture.sanitize(
                        [
                            {
                                "role": "user",
                                "content": json.dumps(
                                    {
                                        "knowledge": {"account_number": account},
                                        "secret": credential,
                                        "background": narrative,
                                        "voice": writing,
                                    }
                                ),
                            }
                        ]
                    ),
                )
            )
            result = await super().respond(observation, state)
            return (
                result.model_copy(update={"message": narrative + " " + writing})
                if result.message
                else result
            )

        def take_interactions(self) -> tuple[ModelInteraction, ...]:
            values = tuple(self.interactions)
            self.interactions.clear()
            return values

    monkeypatch.setattr(application, "preflight_models", preflight)
    monkeypatch.setattr(
        channel_module, "DynamicPlaywrightChannel", lambda *_args, **_kwargs: FakeChannel()
    )
    monkeypatch.setattr(
        application,
        "TinyTroupePersonaSimulator",
        lambda *_args, **kwargs: CapturingPersona(kwargs["prompt_capture"]),
    )
    monkeypatch.setattr(application, "OpenAIProvider", lambda *_args, **_kwargs: FakeProvider())

    def service(**kwargs: Any) -> EvaluationService:
        return EvaluationService(
            **kwargs, evaluators={"semantic_goal_completion": GoalEvaluator("passed")}
        )

    monkeypatch.setattr(application, "EvaluationService", service)
    output = CliRunner().invoke(app, ["run", str(path)])
    assert output.exit_code == 0, output.output
    directory = next((tmp_path / "runs").iterdir())
    trace = [json.loads(line) for line in (directory / "trace.jsonl").read_text().splitlines()]
    prompts = [
        event["data"]["captured_prompt"]
        for event in trace
        if event["type"] == "model_interaction_completed"
    ]
    assert len(prompts) == 2
    assert all("<redacted:account_number>" in prompt for prompt in prompts)
    assert all("<redacted:secret>" in prompt for prompt in prompts)
    assert "private-" not in (directory / "trace.jsonl").read_text()
    assert "canary-" not in (directory / "trace.jsonl").read_text()
    for artifact in directory.rglob("*"):
        if artifact.is_file() and artifact.suffix in {".json", ".jsonl", ".yaml", ".log"}:
            assert "private-persona" not in artifact.read_text()
            assert "private-writing" not in artifact.read_text()
    effective = yaml.safe_load((directory / "config.yaml").read_text())["effective_config"]
    assert effective["persona"]["communication"]["message_length"] == "short"
    assert effective["persona"]["communication"]["grammar_error_probability"] == 0.15
    assert "prompts_not_captured" not in json.dumps(trace)
    metadata = json.loads((directory / "metadata.json").read_text())
    assert "prompts_not_captured" not in metadata["metadata"]["capture_omissions"]


@pytest.mark.parametrize("blocked", ["model", "browser", "evaluator", "persistence"])
def test_cli_deadline_matrix_retains_evidence_and_reports_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blocked: str
) -> None:
    path = scenario_file(tmp_path)
    source = yaml.safe_load(path.read_text())
    source["execution"].update(
        model_timeout_seconds=0.1,
        response_timeout_seconds=0.1,
        evaluator_timeout_seconds=0.1,
        persistence_timeout_seconds=0.1,
        journey_timeout_seconds=3,
    )
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "canary-credential")

    async def preflight(*_args: Any) -> CapabilityPreflight:
        return CapabilityPreflight()

    class SlowPersona(FakePersona):
        async def respond(
            self, observation: ConversationObservation, state: PersonaState
        ) -> PersonaAction:
            if blocked == "model":
                await asyncio.Event().wait()
            return await super().respond(observation, state)

    class SlowChannel(FakeChannel):
        async def receive(self) -> CompletedResponse:
            if blocked == "browser":
                await asyncio.Event().wait()
            return await super().receive()

    class SlowEvaluator(GoalEvaluator):
        async def evaluate(self, context: Any) -> EvaluationResult:
            if blocked == "evaluator":
                await asyncio.Event().wait()
            return await super().evaluate(context)

    class SlowStorage(application.FilesystemStorage):
        async def append_event(self, run_id: str, event: Any) -> None:
            if blocked == "persistence" and event.type == "message_observed":
                await asyncio.Event().wait()
            await super().append_event(run_id, event)

    channel = SlowChannel()
    persona = SlowPersona()
    monkeypatch.setattr(application, "preflight_models", preflight)
    monkeypatch.setattr(
        channel_module, "DynamicPlaywrightChannel", lambda *_args, **_kwargs: channel
    )
    monkeypatch.setattr(
        application, "TinyTroupePersonaSimulator", lambda *_args, **_kwargs: persona
    )
    monkeypatch.setattr(application, "OpenAIProvider", lambda *_args, **_kwargs: FakeProvider())
    monkeypatch.setattr(application, "FilesystemStorage", SlowStorage)

    def service(**kwargs: Any) -> EvaluationService:
        return EvaluationService(
            **kwargs, evaluators={"semantic_goal_completion": SlowEvaluator("passed")}
        )

    monkeypatch.setattr(application, "EvaluationService", service)
    started = time.monotonic()
    output = CliRunner().invoke(app, ["run", str(path)])
    assert time.monotonic() - started < 3
    assert output.exit_code == 3, output.output
    assert channel.closed and persona.closed
    assert "canary-credential" not in output.output
    directory = next((tmp_path / "runs").iterdir())
    events = [json.loads(line) for line in (directory / "trace.jsonl").read_text().splitlines()]
    assert events and [event["sequence"] for event in events] == list(range(len(events)))
    metadata = json.loads((directory / "metadata.json").read_text())
    if blocked == "persistence":
        assert metadata["lifecycle_status"] != "finalized"
        assert not (directory / "report.json").exists()
        assert "final report unavailable" in output.output
    else:
        report = json.loads((directory / "report.json").read_text())
        assert metadata["outcome"] == report["outcome"] == "error"
        assert "Outcome: error" in output.output
        if blocked in {"model", "browser"}:
            category = "model" if blocked == "model" else "channel"
            assert any(error["category"] == category for error in report["errors"])
            assert report["termination_reason"] == f"{category}_error"
        else:
            assert report["goal_result"]["status"] == "error"


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
    monkeypatch.setattr(
        channel_module,
        "DynamicPlaywrightChannel",
        lambda *_args, **_kwargs: pytest.fail("unexpected browser construction"),
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
        channel_module,
        "DynamicPlaywrightChannel",
        lambda *_args, **_kwargs: pytest.fail("browser constructed"),
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
    monkeypatch.setattr(
        channel_module, "DynamicPlaywrightChannel", lambda *_args, **_kwargs: channel
    )
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
        ("append", 3, "final report unavailable", False),
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

        async def append_event(self, *args: Any, **kwargs: Any) -> None:
            event = kwargs.get("event") if kwargs else args[1]
            if failure == "append" and event.type == "message_observed":
                raise OSError("hostile secret: canary-credential")
            await super().append_event(*args, **kwargs)

    monkeypatch.setattr(application, "preflight_models", preflight)
    monkeypatch.setattr(
        channel_module, "DynamicPlaywrightChannel", lambda *_args, **_kwargs: InterruptingChannel()
    )
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
    if failure == "append":
        directory = next((tmp_path / "runs").iterdir())
        assert not (directory / "report.json").exists()
        assert not (directory / "evaluation.json").exists()
        assert "Outcome: passed" not in output.output


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
    monkeypatch.setattr(
        channel_module, "DynamicPlaywrightChannel", lambda *_args, **_kwargs: FakeChannel()
    )
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
        channel_module,
        "DynamicPlaywrightChannel",
        lambda *_args, **_kwargs: pytest.fail("browser constructed"),
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
    monkeypatch.setattr(
        channel_module, "DynamicPlaywrightChannel", lambda *_args, **_kwargs: channel
    )
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

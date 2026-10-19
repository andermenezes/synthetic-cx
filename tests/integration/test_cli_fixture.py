"""Installed CLI composition through Chromium and the controlled fixture."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
import yaml
from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from syntheticcx import cli as cli_module
from syntheticcx.adapters.channel import dynamic as dynamic_channel
from syntheticcx.adapters.channel.dynamic import DynamicChatBroker
from syntheticcx.adapters.llm.preflight import CapabilityPreflight
from syntheticcx.application import run as application
from syntheticcx.cli import app
from syntheticcx.domain.models import (
    ConversationObservation,
    CustomerScenarioContext,
    LLMRequest,
    LLMResponse,
    ModelIdentity,
    ModelInteraction,
    PersonaAction,
    PersonaDefinition,
    PersonaState,
    TokenUsage,
)

EXAMPLE = Path("examples/telecom").resolve()
SCHEMAS = Path("schemas")


class DeterministicPersona:
    """Supply the configured identifier; stop after the first assistant turn."""

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
        return PersonaAction(
            message="My account number is SYNTHETIC-001. Why is this invoice higher?",
            state=state,
            signal="continue",
        )

    async def close(self) -> None:
        self.closed = True


class DeterministicJudge:
    """Exercise the real semantic evaluator and evidence validation offline."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.closed = False

    async def generate(self, request: LLMRequest) -> LLMResponse:
        transcript = json.loads(request.messages[-1].content)["transcript"]
        answers = [item for item in transcript if item["actor"] == "assistant"]
        answer = answers[-1]
        content = " ".join(item["content"] for item in answers)
        passed = "plan changed" in content and "50 more" in content
        now = application.RealClock().now()
        return LLMResponse(
            content=json.dumps(
                {
                    "passed": passed,
                    "explanation": (
                        "The assistant explained the plan change and 50-unit increase."
                        if passed
                        else (
                            "The assistant repeated the account request without explaining "
                            "the increase."
                        )
                    ),
                    "evidence": [
                        {"event_id": answer["event_id"], "message_id": answer["message_id"]}
                    ],
                }
            ),
            interaction=ModelInteraction(
                id=str(uuid4()),
                identity=ModelIdentity(provider="fake", model="deterministic-judge"),
                started_at=now,
                ended_at=now,
                usage=TokenUsage(unavailable_reason="offline test provider"),
            ),
        )

    async def close(self) -> None:
        self.closed = True


def validate_artifacts(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    artifacts = {
        "metadata": "metadata.json",
        "config": "config.yaml",
        "conversation": "conversation.json",
        "evaluation": "evaluation.json",
        "report": "report.json",
    }
    values: dict[str, Any] = {}
    for kind, filename in artifacts.items():
        path = directory / filename
        value = (
            yaml.safe_load(path.read_text()) if kind == "config" else json.loads(path.read_text())
        )
        schema = json.loads((SCHEMAS / f"{kind}.schema.json").read_text())
        Draft202012Validator(schema).validate(value)
        values[kind] = value
    event_schema = Draft202012Validator(
        json.loads((SCHEMAS / "trace-event.schema.json").read_text())
    )
    events = [json.loads(line) for line in (directory / "trace.jsonl").read_text().splitlines()]
    for event in events:
        event_schema.validate(event)
    event_ids = {event["id"] for event in events}
    message_ids = {message["id"] for message in values["conversation"]}
    for finding in values["evaluation"]["results"]:
        assert set(finding["evidence_event_ids"]) <= event_ids
        assert set(finding["evidence_message_ids"]) <= message_ids
    return values["metadata"], values["report"]


@pytest.mark.parametrize(
    ("variant", "persona", "expected_code", "expected_reason"),
    [
        ("success", "standard", 0, "goal_believed_complete"),
        ("repeated_question", "standard", 1, "goal_believed_complete"),
        ("repeated_question", "impatient", 1, "abandonment"),
    ],
)
def test_cli_browser_fixture_produces_explained_schema_valid_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    persona: str,
    expected_code: int,
    expected_reason: str,
) -> None:
    monkeypatch.syspath_prepend(str(EXAMPLE))
    demo = importlib.import_module("run_demo")
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-secret")
    judge = DeterministicJudge()
    customer = DeterministicPersona()

    async def preflight(*_args: Any) -> CapabilityPreflight:
        return CapabilityPreflight()

    monkeypatch.setattr(application, "preflight_models", preflight)
    monkeypatch.setattr(
        application, "TinyTroupePersonaSimulator", lambda *_args, **_kwargs: customer
    )
    monkeypatch.setattr(application, "OpenAIProvider", lambda *_args, **_kwargs: judge)
    with demo.FixtureServer() as fixture:
        configured = demo.configured_example(
            fixture.base_url, variant=variant, persona=persona, run_root=tmp_path / "runs"
        )
        configured["execution"]["greeting_grace_seconds"] = 1
        source = tmp_path / "generated.yaml"
        source.write_text(yaml.safe_dump(configured), encoding="utf-8")
        original = (EXAMPLE / "scenario.yaml").read_bytes()
        output = CliRunner().invoke(app, ["run", str(source)])
        assert (EXAMPLE / "scenario.yaml").read_bytes() == original
    assert output.exit_code == expected_code, output.output
    assert judge.closed and customer.closed
    directory = next((tmp_path / "runs").iterdir())
    metadata, report = validate_artifacts(directory)
    conversation = json.loads((directory / "conversation.json").read_text())
    assert [message["sequence"] for message in conversation] == list(range(len(conversation)))
    assert (
        metadata["outcome"] == report["outcome"] == ("passed" if expected_code == 0 else "failed")
    )
    assert metadata["termination_reason"] == report["termination_reason"] == expected_reason
    goal = report["goal_result"]
    expected_console = {
        "Run ID": report["run_id"],
        "Outcome": report["outcome"],
        "Termination": report["termination_reason"],
        "Goal": goal["status"] if goal else "unavailable",
        "Turns": report["turn_count"],
        "Duration (seconds)": report["duration_seconds"],
        "Repetition findings": report["repetition_count"],
        "Errors": ", ".join(
            f"{error['category']} ({error['operation']}): {error['message']}"
            for error in report["errors"]
        )
        or "none",
        "Artifacts": report["artifact_location"],
    }
    for label, value in expected_console.items():
        assert f"{label}: {value}\n" in output.output + "\n"
    assert "offline-test-secret" not in output.output

    for path in directory.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".yaml", ".txt"}:
            assert "offline-test-secret" not in path.read_text(), path
    assert report["goal_result"]["evidence_event_ids"]
    evaluation = json.loads((directory / "evaluation.json").read_text())
    assert all(finding["explanation"] for finding in evaluation["results"])
    if expected_code == 0:
        assert all(
            finding["status"] == "passed"
            for finding in evaluation["results"]
            if finding["required"]
        )
    if variant == "repeated_question":
        assert report["goal_result"]["status"] == "failed"
        assert report["repetition_count"] == 1
        if persona == "impatient":
            assert any(
                event["data"].get("after", {}).get("abandoned")
                for event in [
                    json.loads(line)
                    for line in (directory / "trace.jsonl").read_text().splitlines()
                ]
                if event["type"] == "behavioral_state_changed"
            )


def test_two_personas_keep_scenario_identical_and_runs_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(EXAMPLE))
    demo = importlib.import_module("run_demo")
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-secret")

    async def preflight(*_args: Any) -> CapabilityPreflight:
        return CapabilityPreflight()

    monkeypatch.setattr(application, "preflight_models", preflight)
    monkeypatch.setattr(
        application, "TinyTroupePersonaSimulator", lambda *_args, **_kwargs: DeterministicPersona()
    )
    monkeypatch.setattr(
        application, "OpenAIProvider", lambda *_args, **_kwargs: DeterministicJudge()
    )
    canonical = yaml.safe_load((EXAMPLE / "scenario.yaml").read_text())["scenario"]
    with demo.FixtureServer() as fixture:
        for persona in ("standard", "impatient"):
            document = demo.configured_example(
                fixture.base_url,
                variant="repeated_question",
                persona=persona,
                run_root=tmp_path / "runs",
            )
            assert document["scenario"] == canonical
            document["execution"]["greeting_grace_seconds"] = 1
            source = tmp_path / f"{persona}.yaml"
            source.write_text(yaml.safe_dump(document), encoding="utf-8")
            output = CliRunner().invoke(app, ["run", str(source)])
            assert output.exit_code == 1, output.output
    directories = list((tmp_path / "runs").iterdir())
    assert len(directories) == len({directory.name for directory in directories}) == 2
    by_persona = {
        json.loads((directory / "metadata.json").read_text())["persona_id"]: directory
        for directory in directories
    }
    assert set(by_persona) == {"standard-customer", "impatient-customer"}
    assert (
        json.loads((by_persona["standard-customer"] / "report.json").read_text())[
            "termination_reason"
        ]
        == "goal_believed_complete"
    )
    assert (
        json.loads((by_persona["impatient-customer"] / "report.json").read_text())[
            "termination_reason"
        ]
        == "abandonment"
    )


def test_identical_config_creates_distinct_runs_without_overwriting_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(EXAMPLE))
    demo = importlib.import_module("run_demo")
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-secret")

    async def preflight(*_args: Any) -> CapabilityPreflight:
        return CapabilityPreflight()

    monkeypatch.setattr(application, "preflight_models", preflight)
    monkeypatch.setattr(
        application, "TinyTroupePersonaSimulator", lambda *_args, **_kwargs: DeterministicPersona()
    )
    monkeypatch.setattr(
        application, "OpenAIProvider", lambda *_args, **_kwargs: DeterministicJudge()
    )
    with demo.FixtureServer() as fixture:
        configured = demo.configured_example(
            fixture.base_url, variant="success", persona="standard", run_root=tmp_path / "runs"
        )
        configured["execution"]["greeting_grace_seconds"] = 1
        configured["model"]["seed"] = 123
        source = tmp_path / "identical.yaml"
        source.write_text(yaml.safe_dump(configured), encoding="utf-8")
        first = CliRunner().invoke(app, ["run", str(source)])
        assert first.exit_code == 0, first.output
        first_dir = next((tmp_path / "runs").iterdir())
        before = {
            path.relative_to(first_dir): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in first_dir.rglob("*")
            if path.is_file()
        }
        second = CliRunner().invoke(app, ["run", str(source)])
        assert second.exit_code == 0, second.output
    directories = list((tmp_path / "runs").iterdir())
    assert len(directories) == len({directory.name for directory in directories}) == 2
    second_dir = next(directory for directory in directories if directory != first_dir)
    assert before == {
        path.relative_to(first_dir): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in first_dir.rglob("*")
        if path.is_file()
    }
    for directory in (first_dir, second_dir):
        metadata, report = validate_artifacts(directory)
        assert metadata["id"] == report["run_id"] == directory.name
        for role in ("persona_model", "evaluation_model"):
            model = metadata["metadata"][role]
            assert model["provider"] == "openai"
            assert model["model"] == configured["model"]["model"]
            assert model["requested_seed"] == 123
            assert model["sampling_parameters"]["requested_temperature"] == 0


def test_demo_launcher_uses_and_removes_temporary_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(EXAMPLE))
    demo = importlib.import_module("run_demo")
    source_path: Path | None = None

    def run_command(args: list[str], *, check: bool) -> SimpleNamespace:
        nonlocal source_path
        assert args[:4] == [sys.executable, "-m", "syntheticcx", "run"]
        assert not check
        source_path = Path(args[4])
        document = yaml.safe_load(source_path.read_text())
        assert document["channel"]["url"].startswith("http://127.0.0.1:")
        assert document["persona"]["id"] == "impatient-customer"
        assert document["storage"]["run_root"] == str(tmp_path)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(demo.subprocess, "run", run_command)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_demo.py",
            "--variant",
            "repeated_question",
            "--persona",
            "impatient",
            "--run-root",
            str(tmp_path),
        ],
    )
    assert demo.main() == 1
    assert source_path is not None and not source_path.exists()


@pytest.mark.parametrize("variant", ["missing_control", "silent_response"])
def test_cli_fixture_channel_failure_keeps_evidence_and_closes_browser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    monkeypatch.syspath_prepend(str(EXAMPLE))
    demo = importlib.import_module("run_demo")
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-secret")

    async def preflight(*_args: Any) -> CapabilityPreflight:
        return CapabilityPreflight()

    channels: list[DynamicChatBroker] = []

    class ObservedChannel(DynamicChatBroker):
        closed = 0

        async def close(self) -> None:
            self.closed += 1
            await super().close()

    def new_channel(*args: Any, **kwargs: Any) -> ObservedChannel:
        channel = ObservedChannel(*args, **kwargs)
        channels.append(channel)
        return channel

    monkeypatch.setattr(application, "preflight_models", preflight)
    monkeypatch.setattr(dynamic_channel, "DynamicChatBroker", new_channel)
    monkeypatch.setattr(
        application, "TinyTroupePersonaSimulator", lambda *_args, **_kwargs: DeterministicPersona()
    )
    monkeypatch.setattr(
        application, "OpenAIProvider", lambda *_args, **_kwargs: DeterministicJudge()
    )
    with demo.FixtureServer() as fixture:
        document = demo.configured_example(
            fixture.base_url,
            variant=variant,
            persona="standard",
            run_root=tmp_path / "runs",
        )
        document["execution"]["greeting_grace_seconds"] = 1
        document["execution"]["browser_action_timeout_seconds"] = 1
        document["execution"]["response_timeout_seconds"] = 0.5
        source = tmp_path / f"{variant}.yaml"
        source.write_text(yaml.safe_dump(document), encoding="utf-8")
        output = CliRunner().invoke(app, ["run", str(source)])
    assert output.exit_code == 3, output.output
    assert len(channels) == 1 and channels[0].closed == 1
    directory = next((tmp_path / "runs").iterdir())
    metadata, report = validate_artifacts(directory)
    assert metadata["outcome"] == report["outcome"] == "error"
    assert report["errors"]
    assert "channel" in json.dumps(report["errors"])
    assert "offline-test-secret" not in output.output


def test_discovered_guide_runs_through_normal_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only discovery exploration is doubled; verification and execution use Chromium."""
    monkeypatch.syspath_prepend(str(EXAMPLE))
    demo = importlib.import_module("run_demo")
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-secret")

    async def proposed(scenario: Path, url: str, **_kwargs: Any):
        from syntheticcx.adapters.channel.broker import NativeChatBroker
        from syntheticcx.config.loader import load_discovery_draft
        from syntheticcx.config.models import StatefulChannelConfig
        from syntheticcx.discovery.budget import DiscoveryBudget
        from syntheticcx.discovery.contracts import LearnResult
        from syntheticcx.discovery.replay import replay

        document = yaml.safe_load(scenario.read_text())
        channel = yaml.safe_load((EXAMPLE / "scenario.yaml").read_text())["channel"]
        channel["url"] = url
        assert "channel" not in document
        # Double only the learned private sample; exact replay and runtime are native.
        sample = StatefulChannelConfig.model_validate(
            {
                **channel,
                "type": "webchat_playwright_stateful",
                "initial_state_id": "composer",
                "states": [
                    {
                        "id": "composer",
                        "kind": "composer",
                        "input_locator": {"strategy": "test_id", "value": "chat-input"},
                        "send_strategy": {
                            "type": "click",
                            "locator": {"strategy": "test_id", "value": "send-message"},
                        },
                    }
                ],
                "transitions": [
                    {
                        "id": "probe",
                        "from_state": "composer",
                        "to_state": "composer",
                        "action_kind": "probe",
                        "probe_id": "explicit",
                    }
                ],
                "verification_recipe": [{"transition_id": "probe"}],
            }
        )
        limits = load_discovery_draft(scenario).execution
        broker = NativeChatBroker(contract=sample, limits=limits)
        try:
            await broker.start()
            return await replay(
                broker,
                LearnResult(config=sample, probes={"explicit": "Synthetic test: please reply."}),
                budget=DiscoveryBudget.create(limits.journey_timeout_seconds),
            )
        finally:
            await broker.close()

    async def preflight(*_args: Any) -> CapabilityPreflight:
        return CapabilityPreflight()

    monkeypatch.setattr(cli_module, "discover_guide", proposed)
    monkeypatch.setattr(application, "preflight_models", preflight)
    monkeypatch.setattr(
        application, "TinyTroupePersonaSimulator", lambda *_args, **_kwargs: DeterministicPersona()
    )
    monkeypatch.setattr(
        application, "OpenAIProvider", lambda *_args, **_kwargs: DeterministicJudge()
    )
    with demo.FixtureServer() as fixture:
        document = demo.configured_example(
            fixture.base_url, variant="success", persona="standard", run_root=tmp_path / "runs"
        )
        document.pop("channel")
        document["execution"]["greeting_grace_seconds"] = 1
        draft = tmp_path / "draft.yaml"
        draft.write_text(yaml.safe_dump(document))
        guide = tmp_path / "chat-guide.yaml"
        discovered = CliRunner().invoke(
            app,
            [
                "discover",
                str(draft),
                "--url",
                f"{fixture.base_url}/?variant=success&view=frame",
                "--output",
                str(guide),
            ],
        )
        assert discovered.exit_code == 0, discovered.output
        completed = tmp_path / "draft.completed.yaml"
        assert guide.exists() and completed.exists()
        published = yaml.safe_load(guide.read_text())
        assert "version" not in published
        assert published["channel"]["type"] == "webchat_playwright_dynamic"
        assert "states" not in published["channel"]
        output = CliRunner().invoke(app, ["run", str(completed), "--language", "pt-BR"])
        assert output.exit_code == 0, output.output
    directory = next((tmp_path / "runs").iterdir())
    metadata, report = validate_artifacts(directory)
    assert metadata["outcome"] == report["outcome"] == "passed"
    config_artifact = yaml.safe_load((directory / "config.yaml").read_text())
    assert (
        config_artifact["channel_guide_source_hash"]
        == hashlib.sha256(guide.read_bytes()).hexdigest()
    )
    assert "channel_guide" not in config_artifact["effective_config"]
    assert config_artifact["effective_config"]["persona"]["communication"]["language"] == "pt-BR"

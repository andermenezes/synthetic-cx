"""Deterministic runner boundaries: persistence, terminal policy, and cleanup."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from syntheticcx.adapters.storage.filesystem import FilesystemStorage
from syntheticcx.config.loader import load_config
from syntheticcx.config.models import BehaviorConfig, Criterion, PrivacyConfig, ProjectConfig
from syntheticcx.domain.artifacts import ConfigArtifact, Run
from syntheticcx.domain.events import (
    EvaluationCompletedEvent,
    OperationFailure,
    RunLifecycleEvent,
    TypedTraceEvent,
)
from syntheticcx.domain.models import (
    ArtifactRecord,
    CompletedResponse,
    ConversationObservation,
    CustomerScenarioContext,
    EvaluationResult,
    LLMResponse,
    Message,
    ModelIdentity,
    ModelInteraction,
    PersonaAction,
    PersonaDefinition,
    PersonaState,
    ReplyOption,
    Scenario,
    SelectionReceipt,
    SendReceipt,
    TokenUsage,
)
from syntheticcx.errors import ModelError
from syntheticcx.evaluation.service import EvaluationService
from syntheticcx.privacy import Redactor
from syntheticcx.privacy.prompt_capture import PromptCapture
from syntheticcx.runtime.behavior_signals import (
    DetectorRegistration,
    EmptyDetectorParameters,
    prepare_behavior,
)
from syntheticcx.runtime.clock import ManualClock, ManualWaitController
from syntheticcx.runtime.journal import RunJournal
from syntheticcx.runtime.owner import run_evaluated
from syntheticcx.runtime.result import RuntimeResult
from syntheticcx.runtime.runner import _failure, run_conversation


class GoalEvaluator:
    def __init__(self, status: str = "passed") -> None:
        self.status = status

    async def evaluate(self, context):
        assistant = next(
            (
                event
                for event in context.sanitized_trace
                if event.type == "message_observed" and event.data.message.actor == "assistant"
            ),
            None,
        )
        if assistant is None:
            return None
        return EvaluationResult(
            evaluator_id=context.criterion.evaluator_type,
            criterion_id=context.criterion.id,
            required=context.criterion.required,
            status=self.status,
            explanation="The assistant explained the plan change."
            if self.status == "passed"
            else "The explanation is insufficient.",
            evidence_event_ids=[assistant.id],
            evidence_message_ids=[assistant.data.message.id],
        )


@pytest.mark.parametrize(
    ("verdict", "expected", "exit_code"),
    [
        ("passed", "passed", 0),
        ("failed", "failed", 1),
        ("error", "error", 3),
    ],
)
def test_owner_commits_evaluation_and_final_lifecycle_before_closing_storage(
    verdict: str, expected: str, exit_code: int
) -> None:
    async def exercise() -> None:
        config = load_config(Path("examples/telecom/scenario.yaml"))
        storage = Storage()
        clock = ManualClock(NOW)
        wait = ManualWaitController(clock)
        redactor = Redactor(PrivacyConfig())
        journal = RunJournal("run-1", storage, redactor, clock)
        service = EvaluationService(
            journal=journal,
            wait=wait,
            evaluator_timeout_seconds=5,
            redactor=redactor,
            evaluators={"semantic_goal_completion": GoalEvaluator(verdict)},
        )
        result = await run_evaluated(
            run=run(),
            config_artifact=ConfigArtifact(source_content_hash="hash", effective_config={}),
            scenario=Scenario.model_validate(config.scenario.model_dump()),
            persona=PersonaDefinition.model_validate(config.persona.model_dump()),
            channel=Channel(),
            simulator=Persona(),
            storage=storage,
            journal=journal,
            clock=clock,
            wait=wait,
            limits=config.execution.model_copy(update={"greeting_grace_seconds": 0}),
            evaluation=service,
            redactor=redactor,
            project_config=config,
        )
        assert result.finalized and result.outcome == expected and result.exit_code == exit_code
        assert storage.finalized == storage.closed == 1
        assert (
            len([event for event in storage.events if isinstance(event, EvaluationCompletedEvent)])
            == 1
        )
        assert isinstance(storage.events[-1], RunLifecycleEvent)
        assert storage.events[-1].data.outcome == expected
        assert result.conversation.committed_events[-1].data.current == "in_progress"

    asyncio.run(exercise())


def test_owner_finalizes_consistent_filesystem_artifacts(tmp_path: Path) -> None:
    async def exercise() -> None:
        config = load_config(Path("examples/telecom/scenario.yaml"))
        redactor = Redactor(config.privacy)
        storage = FilesystemStorage(tmp_path, redactor, timeout_seconds=5, project_config=config)
        clock = ManualClock(NOW)
        wait = ManualWaitController(clock)
        journal = RunJournal("run-1", storage, redactor, clock)
        service = EvaluationService(
            journal=journal,
            wait=wait,
            evaluator_timeout_seconds=5,
            redactor=redactor,
            evaluators={"semantic_goal_completion": GoalEvaluator()},
        )
        result = await run_evaluated(
            run=run(),
            config_artifact=ConfigArtifact(source_content_hash="hash", effective_config={}),
            scenario=Scenario.model_validate(config.scenario.model_dump()),
            persona=PersonaDefinition.model_validate(config.persona.model_dump()),
            channel=Channel(),
            simulator=Persona(),
            storage=storage,
            journal=journal,
            clock=clock,
            wait=wait,
            limits=config.execution.model_copy(update={"greeting_grace_seconds": 0}),
            evaluation=service,
            redactor=redactor,
            project_config=config,
        )
        assert result.finalized and result.outcome == "passed"
        directory = tmp_path / "run-1"
        metadata = json.loads((directory / "metadata.json").read_text())
        report = json.loads((directory / "report.json").read_text())
        evaluation = json.loads((directory / "evaluation.json").read_text())
        trace = [
            (json.loads(line)) for line in (directory / "trace.jsonl").read_text().splitlines()
        ]
        assert metadata["outcome"] == report["outcome"] == "passed"
        assert (
            metadata["termination_reason"]
            == report["termination_reason"]
            == "goal_believed_complete"
        )
        assert evaluation["results"][0] == report["findings"][0]
        assert trace[-1]["data"]["outcome"] == "passed"

    asyncio.run(exercise())


def test_cancellation_during_acknowledged_finalization_keeps_committed_outcome() -> None:
    class GatedStorage(Storage):
        def __init__(self) -> None:
            super().__init__()
            self.finalizing = asyncio.Event()
            self.release = asyncio.Event()

        async def finalize_run(self, run: Run) -> None:
            self.finalizing.set()
            await self.release.wait()
            await super().finalize_run(run)

    async def exercise() -> None:
        config = load_config(Path("examples/telecom/scenario.yaml"))
        backend = GatedStorage()
        clock = ManualClock(NOW)
        wait = ManualWaitController(clock)
        redactor = Redactor(PrivacyConfig())
        journal = RunJournal("run-1", backend, redactor, clock)
        evaluation = EvaluationService(
            journal=journal,
            wait=wait,
            evaluator_timeout_seconds=5,
            redactor=redactor,
            evaluators={"semantic_goal_completion": GoalEvaluator()},
        )
        task = asyncio.create_task(
            run_evaluated(
                run=run(),
                config_artifact=ConfigArtifact(source_content_hash="hash", effective_config={}),
                scenario=Scenario.model_validate(config.scenario.model_dump()),
                persona=PersonaDefinition.model_validate(config.persona.model_dump()),
                channel=Channel(),
                simulator=Persona(),
                storage=backend,
                journal=journal,
                clock=clock,
                wait=wait,
                limits=config.execution.model_copy(update={"greeting_grace_seconds": 0}),
                evaluation=evaluation,
                redactor=redactor,
            )
        )
        await backend.finalizing.wait()
        task.cancel()
        backend.release.set()
        result = await task
        assert result.finalized and result.outcome == "passed" and result.exit_code == 0
        assert backend.finalized == backend.closed == 1
        assert backend.events[-1].data.outcome == "passed"

    asyncio.run(exercise())


@pytest.mark.parametrize(
    (
        "channel_failure",
        "close_failure",
        "storage_options",
        "expected_outcome",
        "expected_finalized",
        "expected_exit",
    ),
    [
        ("receive", False, {}, "error", True, 3),
        (None, True, {}, "error", True, 3),
        (None, False, {"fail_evaluation": True}, None, False, 3),
        (None, False, {"fail_finalize": True}, None, False, 3),
        (None, False, {"fail_close": True}, "passed", True, 3),
    ],
)
def test_owner_preserves_partial_evidence_on_failure(
    channel_failure: str | None,
    close_failure: bool,
    storage_options: dict[str, bool],
    expected_outcome: str | None,
    expected_finalized: bool,
    expected_exit: int,
) -> None:
    async def exercise() -> None:
        config = load_config(Path("examples/telecom/scenario.yaml"))
        redactor = Redactor(PrivacyConfig())
        storage = Storage(**storage_options)
        clock = ManualClock(NOW)
        wait = ManualWaitController(clock)
        journal = RunJournal("run-1", storage, redactor, clock)
        service = EvaluationService(
            journal=journal,
            wait=wait,
            evaluator_timeout_seconds=5,
            redactor=redactor,
            evaluators={"semantic_goal_completion": GoalEvaluator()},
        )
        result = await run_evaluated(
            run=run(),
            config_artifact=ConfigArtifact(source_content_hash="hash", effective_config={}),
            scenario=Scenario.model_validate(config.scenario.model_dump()),
            persona=PersonaDefinition.model_validate(config.persona.model_dump()),
            channel=Channel(failure=channel_failure, close_failure=close_failure),
            simulator=Persona(),
            storage=storage,
            journal=journal,
            clock=clock,
            wait=wait,
            limits=config.execution.model_copy(update={"greeting_grace_seconds": 0}),
            evaluation=service,
            redactor=redactor,
            project_config=config,
        )
        assert result.finalized is expected_finalized
        assert result.outcome == expected_outcome
        assert result.exit_code == expected_exit
        assert storage.closed == 1
        assert result.cleanup_completed is (not storage.fail_close and not close_failure)
        if storage.fail_close:
            assert result.shutdown_errors[0].operation == "storage.close"
            assert "sensitive" not in result.shutdown_errors[0].message
        assert storage.events == list(result.committed_events)
        if expected_finalized:
            assert result.evaluations[0].status == ("unavailable" if channel_failure else "passed")
            assert storage.events[-1].data.outcome == expected_outcome

    asyncio.run(exercise())


def test_owner_cancellation_records_cancelled_outcome() -> None:
    async def exercise() -> None:
        config = load_config(Path("examples/telecom/scenario.yaml"))
        redactor = Redactor(PrivacyConfig())
        storage = Storage()
        channel = Channel(failure="hang")
        clock = ManualClock(NOW)
        wait = ManualWaitController(clock)
        journal = RunJournal("run-1", storage, redactor, clock)
        service = EvaluationService(
            journal=journal,
            wait=wait,
            evaluator_timeout_seconds=5,
            redactor=redactor,
            evaluators={"semantic_goal_completion": GoalEvaluator()},
        )
        task = asyncio.create_task(
            run_evaluated(
                run=run(),
                config_artifact=ConfigArtifact(source_content_hash="hash", effective_config={}),
                scenario=Scenario.model_validate(config.scenario.model_dump()),
                persona=PersonaDefinition.model_validate(config.persona.model_dump()),
                channel=channel,
                simulator=Persona(),
                storage=storage,
                journal=journal,
                clock=clock,
                wait=wait,
                limits=config.execution.model_copy(update={"greeting_grace_seconds": 0}),
                evaluation=service,
                redactor=redactor,
                project_config=config,
            )
        )
        await channel.receiving.wait()
        task.cancel()
        result = await task
        assert result.finalized and result.outcome == "cancelled" and result.exit_code == 130
        assert result.evaluations[0].status == "unavailable"
        assert channel.closed == storage.closed == 1

    asyncio.run(exercise())


def test_owner_journey_timeout_is_finalized_as_error_with_partial_evidence() -> None:
    async def exercise() -> None:
        config = load_config(Path("examples/telecom/scenario.yaml"))
        storage = Storage()
        channel = Channel(failure="hang")
        clock = ManualClock(NOW)
        wait = ManualWaitController(clock)
        redactor = Redactor(PrivacyConfig())
        journal = RunJournal("run-1", storage, redactor, clock)
        service = EvaluationService(
            journal=journal,
            wait=wait,
            evaluator_timeout_seconds=5,
            redactor=redactor,
            evaluators={"semantic_goal_completion": GoalEvaluator()},
        )
        task = asyncio.create_task(
            run_evaluated(
                run=run(),
                config_artifact=ConfigArtifact(source_content_hash="hash", effective_config={}),
                scenario=Scenario.model_validate(config.scenario.model_dump()),
                persona=PersonaDefinition.model_validate(config.persona.model_dump()),
                channel=channel,
                simulator=Persona(),
                storage=storage,
                journal=journal,
                clock=clock,
                wait=wait,
                limits=config.execution.model_copy(
                    update={
                        "greeting_grace_seconds": 0,
                        "journey_timeout_seconds": 2,
                    }
                ),
                evaluation=service,
                redactor=redactor,
                project_config=config,
            )
        )
        await channel.receiving.wait()
        await clock.advance(2)
        result = await task
        assert result.finalized and result.outcome == "error" and result.exit_code == 3
        assert result.conversation.termination_reason == "journey_timeout"
        assert result.evaluations[0].status == "unavailable"
        assert storage.events[-1].data.termination_reason == "journey_timeout"
        assert channel.closed == storage.closed == 1

    asyncio.run(exercise())


def test_evaluation_cancellation_preserves_conversation_stop_reason() -> None:
    class WaitingEvaluator:
        def __init__(self) -> None:
            self.started = asyncio.Event()

        async def evaluate(self, context):
            self.started.set()
            await asyncio.Event().wait()

    async def exercise() -> None:
        config = load_config(Path("examples/telecom/scenario.yaml"))
        redactor = Redactor(PrivacyConfig())
        storage = Storage()
        clock = ManualClock(NOW)
        wait = ManualWaitController(clock)
        journal = RunJournal("run-1", storage, redactor, clock)
        evaluator = WaitingEvaluator()
        service = EvaluationService(
            journal=journal,
            wait=wait,
            evaluator_timeout_seconds=5,
            redactor=redactor,
            evaluators={"semantic_goal_completion": evaluator},
        )
        task = asyncio.create_task(
            run_evaluated(
                run=run(),
                config_artifact=ConfigArtifact(source_content_hash="hash", effective_config={}),
                scenario=Scenario.model_validate(config.scenario.model_dump()),
                persona=PersonaDefinition.model_validate(config.persona.model_dump()),
                channel=Channel(),
                simulator=Persona(),
                storage=storage,
                journal=journal,
                clock=clock,
                wait=wait,
                limits=config.execution.model_copy(update={"greeting_grace_seconds": 0}),
                evaluation=service,
                redactor=redactor,
            )
        )
        await evaluator.started.wait()
        task.cancel()
        result = await task
        assert result.finalized and result.outcome == "cancelled" and result.exit_code == 130
        assert result.conversation.termination_reason == "goal_believed_complete"
        assert storage.events[-1].data.termination_reason == "cancelled"
        assert any(
            event.data.termination_reason == "goal_believed_complete"
            for event in storage.events
            if isinstance(event, RunLifecycleEvent) and event.data.current == "in_progress"
        )
        assert result.evaluations[0].status == "unavailable"

    asyncio.run(exercise())


def test_optional_evaluation_error_is_visible_without_gating_outcome() -> None:
    class TransientEvaluator:
        def __init__(self) -> None:
            self.calls = 0

        async def evaluate(self, context):
            self.calls += 1
            raise ModelError("evaluation.generate", "temporary", retryable=True)

    async def exercise() -> None:
        config = load_config(Path("examples/telecom/scenario.yaml"))
        scenario = Scenario.model_validate(config.scenario.model_dump()).model_copy(
            update={
                "success_criteria": [
                    *config.scenario.success_criteria,
                    Criterion(
                        id="optional-timeout",
                        evaluator_type="journey_timeout",
                        result_kind="boolean",
                        evidence_policy="required",
                        required=False,
                    ),
                ]
            }
        )
        redactor = Redactor(PrivacyConfig())
        storage = Storage()
        clock = ManualClock(NOW)
        wait = ManualWaitController(clock)
        journal = RunJournal("run-1", storage, redactor, clock)
        transient = TransientEvaluator()
        service = EvaluationService(
            journal=journal,
            wait=wait,
            evaluator_timeout_seconds=5,
            redactor=redactor,
            evaluators={"semantic_goal_completion": GoalEvaluator(), "journey_timeout": transient},
        )
        result = await run_evaluated(
            run=run(),
            config_artifact=ConfigArtifact(source_content_hash="hash", effective_config={}),
            scenario=scenario,
            persona=PersonaDefinition.model_validate(config.persona.model_dump()),
            channel=Channel(),
            simulator=Persona(),
            storage=storage,
            journal=journal,
            clock=clock,
            wait=wait,
            limits=config.execution.model_copy(update={"greeting_grace_seconds": 0}),
            evaluation=service,
            redactor=redactor,
        )
        assert result.outcome == "passed" and result.exit_code == 0
        assert [finding.status for finding in result.evaluations] == ["passed", "error"]
        assert transient.calls == 2

    asyncio.run(exercise())


def test_evaluation_provider_close_failure_is_recorded_before_finalization() -> None:
    class Provider:
        def __init__(self) -> None:
            self.closed = 0

        async def close(self) -> None:
            self.closed += 1
            raise OSError("sensitive-provider-token")

    async def exercise() -> None:
        config = load_config(Path("examples/telecom/scenario.yaml"))
        redactor = Redactor(PrivacyConfig())
        storage = Storage()
        clock = ManualClock(NOW)
        wait = ManualWaitController(clock)
        journal = RunJournal("run-1", storage, redactor, clock)
        service = EvaluationService(
            journal=journal,
            wait=wait,
            evaluator_timeout_seconds=5,
            redactor=redactor,
            evaluators={"semantic_goal_completion": GoalEvaluator()},
        )
        provider = Provider()
        result = await run_evaluated(
            run=run(),
            config_artifact=ConfigArtifact(source_content_hash="hash", effective_config={}),
            scenario=Scenario.model_validate(config.scenario.model_dump()),
            persona=PersonaDefinition.model_validate(config.persona.model_dump()),
            channel=Channel(),
            simulator=Persona(),
            storage=storage,
            journal=journal,
            clock=clock,
            wait=wait,
            limits=config.execution.model_copy(update={"greeting_grace_seconds": 0}),
            evaluation=service,
            evaluation_provider=provider,
            redactor=redactor,
        )
        assert provider.closed == 1
        assert result.finalized and result.outcome == "error" and result.exit_code == 3
        assert result.errors[-1].operation == "evaluation.close"
        assert result.errors[-1].message == "model operation failed"

    asyncio.run(exercise())


def test_owner_semantic_provider_persists_usage_and_redacted_fact(tmp_path: Path) -> None:
    class Provider:
        def __init__(self) -> None:
            self.closed = 0
            self.requests = []
            self.capture: PromptCapture | None = None

        async def generate(self, request):
            self.requests.append(request)
            transcript = json.loads(request.messages[1].content)["transcript"]
            answer = next(item for item in transcript if item["actor"] == "assistant")
            return LLMResponse(
                content=json.dumps(
                    {
                        "passed": True,
                        "explanation": "The plan increase was explained.",
                        "evidence": [
                            {"event_id": answer["event_id"], "message_id": answer["message_id"]}
                        ],
                    }
                ),
                interaction=ModelInteraction(
                    id="evaluation-call",
                    identity=ModelIdentity(
                        provider="openai", model="fake", model_version="fake-v2"
                    ),
                    started_at=NOW,
                    ended_at=NOW,
                    usage=TokenUsage(input_tokens=12, output_tokens=9),
                    captured_prompt=self.capture.sanitize(
                        [message.model_dump(mode="json") for message in request.messages]
                    )
                    if self.capture is not None
                    else None,
                ),
            )

        async def close(self):
            self.closed += 1

    async def exercise() -> None:
        config = load_config(Path("examples/telecom/scenario.yaml"))
        source = config.model_dump(mode="json")
        source["scenario"]["starting_context"]["account_number"] = "secret-marker"
        source["scenario"]["success_criteria"][0]["parameters"]["expected_facts"] = [
            "The plan changed for secret-marker."
        ]
        source["privacy"]["redaction_keys"] = ["account_number"]
        source["privacy"]["capture_prompts"] = True
        config = ProjectConfig.model_validate(source)
        redactor = Redactor(config.privacy, resolved_secrets=["secret-marker"])
        storage = FilesystemStorage(tmp_path, redactor, timeout_seconds=5, project_config=config)
        clock = ManualClock(NOW)
        wait = ManualWaitController(clock)
        journal = RunJournal("run-1", storage, redactor, clock)
        provider = Provider()
        provider.capture = PromptCapture(redactor, config.scenario)
        service = EvaluationService(
            journal=journal,
            wait=wait,
            evaluator_timeout_seconds=5,
            redactor=redactor,
            provider=provider,
        )
        result = await run_evaluated(
            run=run(),
            config_artifact=ConfigArtifact(source_content_hash="hash", effective_config=source),
            scenario=Scenario.model_validate(config.scenario.model_dump()),
            persona=PersonaDefinition.model_validate(config.persona.model_dump()),
            channel=Channel(),
            simulator=Persona(),
            storage=storage,
            journal=journal,
            clock=clock,
            wait=wait,
            limits=config.execution.model_copy(update={"greeting_grace_seconds": 0}),
            evaluation=service,
            evaluation_provider=provider,
            redactor=redactor,
            project_config=config,
        )
        assert result.finalized and result.outcome == "passed" and provider.closed == 1
        trace = [
            json.loads(line) for line in (tmp_path / "run-1/trace.jsonl").read_text().splitlines()
        ]
        interaction = next(item for item in trace if item["type"] == "model_interaction_completed")
        assert interaction["data"]["usage"]["input_tokens"] == 12
        assert interaction["data"]["usage"]["output_tokens"] == 9
        metadata = json.loads((tmp_path / "run-1/metadata.json").read_text())
        assert metadata["metadata"]["usage"]["input_tokens"] == 12
        assert metadata["metadata"]["usage"]["output_tokens"] == 9
        assert metadata["metadata"]["provider_model_version"] == "fake-v2"
        assert interaction["data"]["captured_prompt"] is not None
        assert "<redacted:account_number>" in interaction["data"]["captured_prompt"]
        assert "prompts_not_captured" not in interaction["data"]["capture_omissions"]
        assert trace.index(interaction) < next(
            i for i, item in enumerate(trace) if item["type"] == "evaluation_completed"
        )
        assert "secret-marker" not in (tmp_path / "run-1/trace.jsonl").read_text()
        assert "secret-marker" not in (tmp_path / "run-1/evaluation.json").read_text()
        assert "secret-marker" not in (tmp_path / "run-1/report.json").read_text()
        assert "secret-marker" not in (tmp_path / "run-1/config.yaml").read_text()
        assert "secret-marker" not in provider.requests[0].messages[1].content

    asyncio.run(exercise())


NOW = datetime(2026, 9, 24, tzinfo=UTC)


@pytest.mark.parametrize(
    "mode", ["success", "failure", "invalid", "timeout", "unsupported", "partial", "cancelled"]
)
def test_persona_feedback_is_supplemental_and_markdown_is_independent(tmp_path, mode):
    from syntheticcx.domain.models import PersonaFeedbackResponse

    async def exercise_feedback():
        data = load_config(Path("examples/telecom/scenario.yaml")).model_dump()
        data["scenario"]["observation_topics"] = [{"id": "clarity", "description": "Was it clear?"}]
        data["reporting"] = {"language": "pt-BR", "markdown": {"enabled": True}}
        config = ProjectConfig.model_validate(data)
        clock = ManualClock(NOW)
        wait = ManualWaitController(clock)
        channel = Channel()

        class ReflectingPersona(Persona):
            reflection_available = True

            def __init__(self):
                super().__init__(complete=mode != "partial")
                self.reflections = 0
                self.closed = False

            async def close(self):
                self.closed = True

            async def reflect(self, request):
                assert channel.closed == 1 and not self.closed
                self.reflections += 1
                assert request.language == "pt-BR"
                assert "success_criteria" not in request.model_dump_json()
                await clock.advance(2)
                if mode == "failure":
                    raise ModelError("persona.reflect", "sensitive-token")
                if mode == "timeout":
                    raise TimeoutError
                if mode == "cancelled":
                    raise asyncio.CancelledError
                ref = request.messages[-1].id if mode != "invalid" else "invented-message"
                return PersonaFeedbackResponse.model_validate(
                    {
                        "summary": "Eu entendi a explicação.",
                        "topics": [
                            {
                                "topic_id": "clarity",
                                "status": "observed",
                                "impression": "Achei claro, mas não é um resultado formal.",
                                "evidence_message_ids": [ref],
                            }
                        ],
                    }
                )

            def take_interactions(self):
                if not self.reflections:
                    return ()
                return (
                    ModelInteraction(
                        id="reflection-call",
                        identity=run().metadata.persona_model,
                        started_at=NOW,
                        ended_at=clock.now(),
                        usage=TokenUsage(input_tokens=2, output_tokens=3),
                    ),
                )

        class IsolatedEvaluator(GoalEvaluator):
            async def evaluate(self, context):
                assert not any(
                    event.type == "persona_feedback_recorded" for event in context.sanitized_trace
                )
                assert not any(
                    event.type == "model_interaction_completed"
                    and event.data.id == "reflection-call"
                    for event in context.sanitized_trace
                )
                return await super().evaluate(context)

        simulator = ReflectingPersona() if mode != "unsupported" else Persona()
        redactor = Redactor(config.privacy)
        storage = FilesystemStorage(
            tmp_path,
            redactor,
            timeout_seconds=5,
            project_config=config,
            materialize_report=False,
            materialize_conversation=False,
        )
        journal = RunJournal("run-1", storage, redactor, clock)
        evaluation = EvaluationService(
            journal=journal,
            wait=wait,
            evaluator_timeout_seconds=5,
            redactor=redactor,
            evaluators={"semantic_goal_completion": IsolatedEvaluator()},
        )
        result = await run_evaluated(
            run=run(),
            config_artifact=ConfigArtifact(source_content_hash="hash", effective_config={}),
            scenario=Scenario.model_validate(config.scenario.model_dump()),
            persona=PersonaDefinition.model_validate(config.persona.model_dump()),
            channel=channel,
            simulator=simulator,
            storage=storage,
            journal=journal,
            clock=clock,
            wait=wait,
            limits=config.execution.model_copy(
                update={"greeting_grace_seconds": 0, "max_turns": 1 if mode == "partial" else 2}
            ),
            evaluation=evaluation,
            redactor=redactor,
            project_config=config,
        )
        assert result.finalized
        assert result.outcome == ("cancelled" if mode == "cancelled" else "passed")
        assert len(channel.sent) == 1
        feedback_event = next(
            event for event in journal.committed_events if event.type == "persona_feedback_recorded"
        )
        feedback = feedback_event.data
        assert feedback.status == ("available" if mode in {"success", "partial"} else "unavailable")
        assert feedback.partial_experience == (mode == "partial")
        assert not (tmp_path / "run-1/report.json").exists()
        assert not (tmp_path / "run-1/conversation.json").exists()
        markdown = (tmp_path / "run-1/report.md").read_text()
        assert "Impressões do cliente" in markdown
        assert "Conclusões formais do teste" in markdown
        assert "sensitive-token" not in markdown
        metadata = json.loads((tmp_path / "run-1/metadata.json").read_text())
        assert any(item["path"] == "report.md" for item in metadata["artifacts"])
        stop = next(
            event
            for event in journal.committed_events
            if isinstance(event, RunLifecycleEvent)
            and event.data.conversation_duration_seconds is not None
        )
        assert (
            stop.data.conversation_duration_seconds < (clock.now() - NOW).total_seconds()
            or mode == "unsupported"
        )

    asyncio.run(exercise_feedback())


PNG = b"\x89PNG\r\n\x1a\nsynthetic-fixture-bytes"


def capture_config() -> ProjectConfig:
    source = load_config(Path("examples/telecom/scenario.yaml")).model_dump(mode="json")
    source["privacy"].update(allow_screenshots=True, confirm_screenshot_target_safe=True)
    source["storage"]["store_screenshots"] = True
    return ProjectConfig.model_validate(source)


def run() -> Run:
    return Run.model_validate(
        {
            "id": "run-1",
            "scenario_id": "billing",
            "persona_id": "standard",
            "started_at": NOW,
            "lifecycle_status": "in_progress",
            "metadata": {
                "configuration_version": "1",
                "framework_version": "0.1.0",
                "persona_model": {"provider": "fake", "model": "fake"},
                "evaluation_model": {"provider": "fake", "model": "fake"},
                "usage": {"unavailable_reason": "none"},
                "runtime_platform": "test",
                "owner": {
                    "owner_uuid": "owner",
                    "hostname": "test",
                    "pid": 1,
                    "process_start_evidence": "start",
                },
            },
        }
    )


class Storage:
    def __init__(
        self,
        *,
        fail_append: bool = False,
        fail_at_sequence: int | None = None,
        fail_customer: bool = False,
        fail_create: bool = False,
        fail_evaluation: bool = False,
        fail_finalize: bool = False,
        fail_close: bool = False,
    ) -> None:
        self.created = 0
        self.finalized = 0
        self.closed = 0
        self.events: list[TypedTraceEvent] = []
        self.fail_append = fail_append
        self.fail_at_sequence = fail_at_sequence
        self.fail_customer = fail_customer
        self.fail_create = fail_create
        self.fail_evaluation = fail_evaluation
        self.fail_finalize = fail_finalize
        self.fail_close = fail_close

    async def create_run(self, run: Run, config_artifact: ConfigArtifact) -> None:
        assert run.id == "run-1"
        self.created += 1
        if self.fail_create:
            raise OSError("sensitive-storage-path")

    async def append_event(self, run_id: str, event: TypedTraceEvent) -> None:
        if (
            self.fail_append
            or event.sequence == self.fail_at_sequence
            or self.fail_customer
            and event.type == "message_observed"
            and event.data.message.actor == "customer"
            or self.fail_evaluation
            and isinstance(event, EvaluationCompletedEvent)
        ):
            raise OSError("sensitive-storage-path")
        self.events.append(event)

    async def finalize_run(self, run: Run) -> None:
        self.finalized += 1
        if self.fail_finalize:
            raise OSError("sensitive-storage-path")

    async def close(self) -> None:
        self.closed += 1
        if self.fail_close:
            raise OSError("sensitive-storage-path")


def reply(content: str) -> CompletedResponse:
    return CompletedResponse(
        messages=[
            Message(
                id="assistant-1", actor="assistant", sequence=1, content=content, observed_at=NOW
            )
        ],
        started_at=NOW,
        completed_at=NOW,
        completion_strategy="completion_locator",
    )


class Channel:
    def __init__(
        self,
        *,
        failure: str | None = None,
        close_failure: bool = False,
        close_hangs: bool = False,
        echo: str | None = None,
    ) -> None:
        self.failure = failure
        self.close_failure = close_failure
        self.close_hangs = close_hangs
        self.echo = echo
        self.started = 0
        self.closed = 0
        self.sent: list[str] = []
        self.receiving = asyncio.Event()
        self.closing = asyncio.Event()

    async def start(self) -> None:
        self.started += 1
        if self.failure == "start":
            raise OSError("sensitive-channel-token")

    async def send(self, message: str) -> SendReceipt:
        if self.failure == "send":
            raise OSError("sensitive-channel-token")
        self.sent.append(message)
        return SendReceipt(
            message_id=f"sent-{len(self.sent)}",
            submitted_at=NOW,
            evidence={"echo": self.echo} if self.echo is not None else {},
        )

    async def receive(self) -> CompletedResponse:
        self.receiving.set()
        if self.failure == "receive":
            raise OSError("sensitive-channel-token")
        if self.failure == "hang":
            await asyncio.Event().wait()
        response = reply("Your bill increased due to a plan change")
        return response.model_copy(
            update={
                "messages": [
                    message.model_copy(update={"id": f"assistant-{len(self.sent)}"})
                    for message in response.messages
                ]
            }
        )

    async def close(self) -> None:
        self.closed += 1
        self.closing.set()
        if self.close_hangs:
            await asyncio.Event().wait()
        if self.close_failure:
            raise OSError("sensitive-channel-token")


class Persona:
    def __init__(
        self,
        *,
        failure: bool = False,
        complete: bool = True,
        goal_initial: bool = False,
    ) -> None:
        self.failure = failure
        self.complete = complete
        self.goal_initial = goal_initial
        self.calls = 0

    async def initialize(
        self, definition: PersonaDefinition, scenario_context: CustomerScenarioContext
    ) -> PersonaState:
        return definition.fresh_state()

    async def respond(
        self, observation: ConversationObservation, state: PersonaState
    ) -> PersonaAction:
        self.calls += 1
        if self.failure:
            raise OSError("sensitive-model-prompt")
        if self.goal_initial or (self.complete and state.turn_number):
            return PersonaAction(state=state, signal="goal_believed_complete")
        return PersonaAction(message="My account number is 123", state=state, signal="continue")


async def exercise(
    channel: Channel,
    persona: Persona,
    storage: Storage,
    *,
    clock: ManualClock | None = None,
    max_turns: int = 4,
    timeout: float = 20,
    greeting_grace: float = 0,
    definition: PersonaDefinition | None = None,
    project_config: ProjectConfig | None = None,
    redactor: Redactor | None = None,
    assistant_update_limit: int = 64,
) -> RuntimeResult:
    config = load_config(Path("examples/telecom/scenario.yaml"))
    logical_clock = clock or ManualClock(NOW)
    journal = RunJournal("run-1", storage, redactor or Redactor(PrivacyConfig()), logical_clock)
    result = await run_conversation(
        run=run(),
        config_artifact=ConfigArtifact(source_content_hash="hash", effective_config={}),
        scenario=Scenario.model_validate(config.scenario.model_dump()),
        persona=definition or PersonaDefinition.model_validate(config.persona.model_dump()),
        channel=channel,
        simulator=persona,
        storage=storage,
        journal=journal,
        clock=logical_clock,
        wait=ManualWaitController(logical_clock),
        limits=config.execution.model_copy(
            update={
                "greeting_grace_seconds": greeting_grace,
                "max_turns": max_turns,
                "journey_timeout_seconds": timeout,
                "assistant_update_limit": assistant_update_limit,
            }
        ),
        project_config=project_config or config,
    )
    assert storage.finalized == 0
    assert storage.closed == 1
    assert storage.events == list(result.committed_events)
    return result


@pytest.mark.parametrize(
    "update_limit,reason", [(6, "goal_believed_complete"), (5, "channel_error")]
)
def test_supplemental_updates_have_their_own_budget(update_limit, reason):
    class UpdatingChannel(Channel):
        updates = 0

        async def observe_pending(self):
            if not self.sent or self.updates >= 6:
                return None
            self.updates += 1
            response = reply(f"Settled update {self.updates}")
            return response.model_copy(
                update={
                    "messages": [
                        message.model_copy(update={"id": f"update-{self.updates}"})
                        for message in response.messages
                    ]
                }
            )

    channel = UpdatingChannel()
    result = asyncio.run(
        exercise(channel, Persona(), Storage(), max_turns=2, assistant_update_limit=update_limit)
    )
    assert result.termination_reason == reason
    assert result.turn_count == 1 and len(channel.sent) == 1
    completed = [
        event
        for event in result.committed_events
        if event.type == "behavioral_signal_recorded"
        and event.data.signal.signal == "customer_turn_completed"
    ]
    assert len(completed) == 1
    updates = [
        event
        for event in result.committed_events
        if event.type == "message_observed" and event.data.message.id.startswith("update-")
    ]
    assert len(updates) == 6  # Even the over-budget observation remains evidence.
    if reason == "channel_error":
        assert result.error.operation == "channel.observe"


def test_invalid_persona_output_still_journals_completed_call():
    class InvalidPersona(Persona):
        def __init__(self):
            super().__init__()
            self.pending = []

        async def respond(self, observation, state):
            self.pending.append(
                ModelInteraction(
                    id="completed-invalid",
                    identity=ModelIdentity(provider="fake", model="fake"),
                    started_at=NOW,
                    ended_at=NOW,
                    usage=TokenUsage(input_tokens=100, output_tokens=20),
                )
            )
            from syntheticcx.errors import ModelError

            raise ModelError("persona.respond", "invalid customer action")

        def take_interactions(self):
            interactions, self.pending = tuple(self.pending), []
            return interactions

    channel = Channel()
    result = asyncio.run(exercise(channel, InvalidPersona(), Storage()))
    assert result.termination_reason == "model_error" and not channel.sent
    interactions = [
        event.data
        for event in result.committed_events
        if event.type == "model_interaction_completed"
    ]
    assert len(interactions) == 1
    assert interactions[0].usage.input_tokens == 100
    assert interactions[0].usage.output_tokens == 20
    assert interactions[0].input_event_ids


def test_success_returns_detached_evidence_without_finalizing() -> None:
    async def scenario() -> None:
        storage = Storage()
        channel = Channel()
        result = await exercise(channel, Persona(), storage)
        assert result.termination_reason == "goal_believed_complete"
        assert result.turn_count == result.persona_state.turn_number == 1
        assert result.error is None and result.cleanup_completed
        assert channel.started == channel.closed == storage.created == 1
        assert [event.type for event in result.committed_events][-1] == "run_lifecycle_changed"
        assert result.committed_events[-1].data.current == "incomplete"
        assert result.committed_events[-1].data.outcome is None
        customer = next(
            event.data.message
            for event in result.committed_events
            if event.type == "message_observed" and event.data.message.actor == "customer"
        )
        assert customer.id == "sent-1"
        assert customer.content == "My account number is 123"
        assert not hasattr(result, "outcome")

    asyncio.run(scenario())


class SelectingChannel(Channel):
    def __init__(self, *, uncertain: bool = False, bad_receipt: bool = False) -> None:
        super().__init__()
        self.selected: list[str] = []
        self.uncertain = uncertain
        self.bad_receipt = bad_receipt

    async def select_option(self, option_id: str) -> SelectionReceipt:
        self.selected.append(option_id)
        if self.uncertain:
            raise OSError("uncertain sensitive click result")
        return SelectionReceipt(
            message_id="selected-1",
            submitted_at=NOW,
            selected_option_id=option_id,
            selected_label="Account number information",
            source_message_id="wrong" if self.bad_receipt else "assistant-1",
            evidence={"activation": "native_click"},
        )

    async def receive(self) -> CompletedResponse:
        if self.selected:
            result = reply("Please provide your account number")
            return result.model_copy(
                update={
                    "messages": [
                        result.messages[0].model_copy(update={"id": "assistant-2", "sequence": 2})
                    ]
                }
            )
        result = reply("")
        return result.model_copy(
            update={
                "messages": [
                    result.messages[0].model_copy(
                        update={"options": [ReplyOption(id="opaque:1", label="Displayed label")]}
                    )
                ]
            }
        )


class SelectingPersona(Persona):
    def __init__(self, option_id: str | None = "opaque:1") -> None:
        super().__init__()
        self.option_id = option_id

    async def respond(
        self, observation: ConversationObservation, state: PersonaState
    ) -> PersonaAction:
        self.calls += 1
        if state.turn_number:
            return PersonaAction(state=state, signal="goal_believed_complete")
        return PersonaAction(selected_option_id=self.option_id, state=state, signal="continue")


def test_selection_is_one_turn_with_adapter_label_evidence_and_no_information_supply() -> None:
    async def scenario() -> None:
        channel = SelectingChannel()
        persona = SelectingPersona()
        result = await exercise(channel, persona, Storage(), greeting_grace=1)
        assert result.error is None and result.termination_reason == "goal_believed_complete"
        assert result.turn_count == result.persona_state.turn_number == 1
        assert channel.selected == ["opaque:1"] and channel.sent == []
        assert persona.calls == 2
        customer = next(
            e
            for e in result.committed_events
            if e.type == "message_observed" and e.data.message.actor == "customer"
        )
        assert customer.data.message.content == "Account number information"
        assert customer.data.message.metadata == {
            "action_kind": "select_option",
            "option_id": "opaque:1",
            "source_assistant_message_id": "assistant-1",
        }
        assert customer.data.identity_evidence == {"activation": "native_click"}
        assert not any(e.type == "repeated_information_observed" for e in result.committed_events)
        assert result.persona_state.frustration == 0.1

    asyncio.run(scenario())


@pytest.mark.parametrize("option_id", [None, "unknown", "opaque:old"])
def test_graph_rejects_missing_unknown_or_stale_selection_before_dispatch(
    option_id: str | None,
) -> None:
    async def scenario() -> None:
        channel = SelectingChannel()
        result = await exercise(channel, SelectingPersona(option_id), Storage(), greeting_grace=1)
        assert result.termination_reason == "model_error" and result.turn_count == 0
        assert channel.selected == channel.sent == []

    asyncio.run(scenario())


@pytest.mark.parametrize("bad_receipt", [False, True])
def test_failed_selection_never_retries_or_falls_back_to_text(bad_receipt: bool) -> None:
    async def scenario() -> None:
        channel = SelectingChannel(uncertain=not bad_receipt, bad_receipt=bad_receipt)
        result = await exercise(channel, SelectingPersona(), Storage(), greeting_grace=1)
        assert result.termination_reason == "channel_error"
        assert result.error is not None and result.error.operation == "channel.send"
        assert channel.selected == ["opaque:1"] and channel.sent == []
        assert "sensitive" not in result.model_dump_json()

    asyncio.run(scenario())


def test_selection_requires_optional_channel_capability_without_text_fallback() -> None:
    class MenuTextChannel(Channel):
        async def receive(self) -> CompletedResponse:
            return await SelectingChannel().receive()

    async def scenario() -> None:
        channel = MenuTextChannel()
        result = await exercise(channel, SelectingPersona(), Storage(), greeting_grace=1)
        assert result.termination_reason == "channel_error" and result.turn_count == 0
        assert channel.sent == []

    asyncio.run(scenario())


def test_selection_label_uses_committed_redaction_not_persona_display_text() -> None:
    class PrivateMenu(SelectingChannel):
        async def receive(self) -> CompletedResponse:
            result = await super().receive()
            if not self.selected:
                result = result.model_copy(
                    update={
                        "messages": [
                            result.messages[0].model_copy(
                                update={
                                    "options": [
                                        ReplyOption(
                                            id="opaque:1", label="Account number information"
                                        )
                                    ],
                                }
                            )
                        ]
                    }
                )
            return result

    class RedactedPersona(SelectingPersona):
        async def respond(
            self, observation: ConversationObservation, state: PersonaState
        ) -> PersonaAction:
            if not state.turn_number:
                assert observation.messages[-1].options[0].label == "<redacted:secret>"
            return await super().respond(observation, state)

    async def scenario() -> None:
        channel = PrivateMenu()
        result = await exercise(
            channel,
            RedactedPersona(),
            Storage(),
            greeting_grace=1,
            redactor=Redactor(PrivacyConfig(), resolved_secrets=["Account number information"]),
        )
        assert result.error is None and result.turn_count == 1
        customer = next(
            e.data.message
            for e in result.committed_events
            if e.type == "message_observed" and e.data.message.actor == "customer"
        )
        assert customer.content == "<redacted:secret>"
        assert customer.metadata["action_kind"] == "select_option"
        assert "Account number information" not in result.model_dump_json()
        assert channel.sent == [] and channel.selected == ["opaque:1"]

    asyncio.run(scenario())


@pytest.mark.parametrize("bad_source", [False, True])
def test_selection_receipt_binds_to_redacted_committed_source(bad_source: bool) -> None:
    class PrivateSourceChannel(SelectingChannel):
        async def receive(self) -> CompletedResponse:
            response = await super().receive()
            if not self.selected:
                response = response.model_copy(
                    update={
                        "messages": [
                            response.messages[0].model_copy(
                                update={"id": "assistant-private-secret"}
                            )
                        ]
                    }
                )
            return response

        async def select_option(self, option_id: str) -> SelectionReceipt:
            receipt = await super().select_option(option_id)
            return receipt.model_copy(
                update={
                    "source_message_id": "wrong-source"
                    if bad_source
                    else "assistant-private-secret",
                }
            )

    class CommittedSourcePersona(SelectingPersona):
        async def respond(
            self, observation: ConversationObservation, state: PersonaState
        ) -> PersonaAction:
            if not state.turn_number:
                assert observation.messages[-1].id == "assistant-<redacted:secret>"
            return await super().respond(observation, state)

    async def scenario() -> None:
        channel = PrivateSourceChannel()
        result = await exercise(
            channel,
            CommittedSourcePersona(),
            Storage(),
            greeting_grace=1,
            redactor=Redactor(PrivacyConfig(), resolved_secrets=["private-secret"]),
        )
        assert channel.selected == ["opaque:1"] and channel.sent == []
        assert "private-secret" not in result.model_dump_json()
        customers = [
            e.data.message
            for e in result.committed_events
            if e.type == "message_observed" and e.data.message.actor == "customer"
        ]
        if bad_source:
            assert result.termination_reason == "channel_error" and result.turn_count == 0
            assert customers == []
        else:
            assert result.error is None and result.termination_reason == "goal_believed_complete"
            assert result.turn_count == result.persona_state.turn_number == 1
            assert len(customers) == 1
            source = next(
                e.data.message
                for e in result.committed_events
                if e.type == "message_observed" and e.data.message.options
            )
            assert customers[0].metadata["source_assistant_message_id"] == source.id
            assert source.id == "assistant-<redacted:secret>"
            assert customers[0].content == "Account number information"
            assert customers[0].metadata["action_kind"] == "select_option"

    asyncio.run(scenario())


def test_selection_acknowledged_before_journal_failure_still_counts_one_turn() -> None:
    async def scenario() -> None:
        channel = SelectingChannel()
        result = await exercise(
            channel, SelectingPersona(), Storage(fail_customer=True), greeting_grace=1
        )
        assert result.termination_reason == "persistence_error"
        assert result.turn_count == result.persona_state.turn_number == 1
        assert channel.selected == ["opaque:1"] and channel.sent == []

    asyncio.run(scenario())


def test_graph_rejects_reusing_option_from_previous_observation() -> None:
    class ReusingPersona(SelectingPersona):
        async def respond(
            self, observation: ConversationObservation, state: PersonaState
        ) -> PersonaAction:
            return PersonaAction(selected_option_id="opaque:1", state=state, signal="continue")

    async def scenario() -> None:
        channel = SelectingChannel()
        result = await exercise(channel, ReusingPersona(), Storage(), greeting_grace=1)
        assert result.termination_reason == "model_error" and result.turn_count == 1
        assert channel.selected == ["opaque:1"] and channel.sent == []

    asyncio.run(scenario())


def test_selection_obeys_existing_max_turn_limit() -> None:
    async def scenario() -> None:
        channel = SelectingChannel()
        persona = SelectingPersona()
        result = await exercise(channel, persona, Storage(), greeting_grace=1, max_turns=1)
        assert result.termination_reason == "max_turns" and result.turn_count == 1
        assert persona.calls == 1 and channel.selected == ["opaque:1"] and channel.sent == []

    asyncio.run(scenario())


def test_persona_close_is_owned_and_bounded_on_success_and_failure() -> None:
    class CloseablePersona(Persona):
        def __init__(self, *, fail: bool = False) -> None:
            super().__init__(failure=fail)
            self.closed = 0

        async def close(self) -> None:
            self.closed += 1

    async def scenario() -> None:
        for failed in (False, True):
            persona = CloseablePersona(fail=failed)
            result = await exercise(Channel(), persona, Storage())
            assert persona.closed == 1
            assert result.cleanup_completed
            assert result.termination_reason == (
                "model_error" if failed else "goal_believed_complete"
            )

    asyncio.run(scenario())


def test_persona_close_failure_is_sanitized_and_does_not_hide_primary_error() -> None:
    class FailingClose(Persona):
        async def close(self) -> None:
            raise RuntimeError("secret model endpoint")

    async def scenario() -> None:
        result = await exercise(Channel(), FailingClose(failure=True), Storage())
        assert result.termination_reason == "model_error"
        assert result.error is not None and result.error.operation == "persona.respond"
        assert not result.cleanup_completed
        failures = [
            event.data.error
            for event in result.committed_events
            if event.type == "operation_failed"
        ]
        assert {failure.operation for failure in failures} == {"persona.respond", "persona.close"}
        assert "secret model endpoint" not in str(result.model_dump())

    asyncio.run(scenario())


def test_persona_cleanup_on_runner_cancellation_and_close_deadline() -> None:
    class ClosingPersona(Persona):
        def __init__(self, *, hangs: bool = False) -> None:
            super().__init__()
            self.hangs = hangs
            self.closed = 0
            self.closing = asyncio.Event()

        async def close(self) -> None:
            self.closed += 1
            self.closing.set()
            if self.hangs:
                await asyncio.Event().wait()

    async def scenario() -> None:
        channel = Channel(failure="hang")
        persona = ClosingPersona()
        task = asyncio.create_task(exercise(channel, persona, Storage()))
        await channel.receiving.wait()
        task.cancel()
        cancelled = await task
        assert cancelled.termination_reason == "cancelled"
        assert cancelled.cleanup_completed
        assert persona.closed == 1

        clock = ManualClock(NOW)
        hanging = ClosingPersona(hangs=True)
        task = asyncio.create_task(exercise(Channel(), hanging, Storage(), clock=clock))
        await hanging.closing.wait()
        await clock.advance(5)
        result = await task
        assert not result.cleanup_completed
        assert result.error is not None and result.error.operation == "persona.close"
        assert hanging.closed == 1

    asyncio.run(scenario())


def test_failed_interaction_append_prevents_customer_send() -> None:
    class EvidencePersona(Persona):
        def __init__(self) -> None:
            super().__init__()
            self.pending = False

        async def respond(
            self, observation: ConversationObservation, state: PersonaState
        ) -> PersonaAction:
            result = await super().respond(observation, state)
            self.pending = True
            return result

        def take_interactions(self) -> tuple[ModelInteraction, ...]:
            if not self.pending:
                return ()
            self.pending = False
            return (
                ModelInteraction(
                    id="persona-call-1",
                    identity=ModelIdentity(provider="fake", model="model"),
                    started_at=NOW,
                    ended_at=NOW,
                    usage=TokenUsage(unavailable_reason="not supplied"),
                ),
            )

    async def scenario() -> None:
        channel = Channel()
        result = await exercise(channel, EvidencePersona(), Storage(fail_at_sequence=2))
        assert result.termination_reason == "persistence_error"
        assert channel.sent == []
        assert [event.type for event in result.committed_events] == [
            "run_lifecycle_changed",
            "conversation_started",
        ]

    asyncio.run(scenario())


def test_interaction_drain_failure_reports_the_specific_persona_operation() -> None:
    class BrokenEvidencePersona(Persona):
        def take_interactions(self) -> tuple[ModelInteraction, ...]:
            raise RuntimeError("secret interaction failure")

    async def scenario() -> None:
        result = await exercise(Channel(), BrokenEvidencePersona(), Storage())
        assert result.termination_reason == "model_error"
        assert result.error is not None
        assert result.error.operation == "persona.interactions"
        assert "secret interaction failure" not in str(result.model_dump())

    asyncio.run(scenario())


def test_timed_out_persona_result_does_not_publish_model_interaction() -> None:
    class DelayedPersona(Persona):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()
            self.pending: list[ModelInteraction] = []
            self.closed = 0

        async def respond(
            self, observation: ConversationObservation, state: PersonaState
        ) -> PersonaAction:
            self.entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.pending.append(
                    ModelInteraction(
                        id="late-call",
                        identity=ModelIdentity(provider="fake", model="fake"),
                        started_at=NOW,
                        ended_at=NOW,
                        usage=TokenUsage(unavailable_reason="not reported"),
                    )
                )
            raise AssertionError("unreachable")

        def take_interactions(self) -> tuple[ModelInteraction, ...]:
            return tuple(self.pending)

        async def close(self) -> None:
            self.closed += 1

    async def scenario() -> None:
        clock = ManualClock(NOW)
        persona = DelayedPersona()
        channel = Channel()
        task = asyncio.create_task(exercise(channel, persona, Storage(), clock=clock, timeout=100))
        await persona.entered.wait()
        await asyncio.sleep(0)
        await clock.advance(30)
        result = await task
        assert result.termination_reason == "model_error"
        assert result.error is not None and result.error.operation == "persona.respond"
        assert persona.closed == 1
        assert persona.pending  # late model metadata existed, but was never acknowledged
        assert channel.sent == []
        assert not any(
            event.type == "model_interaction_completed" for event in result.committed_events
        )

    asyncio.run(scenario())


def test_max_turns_respects_large_graph_recursion_budget() -> None:
    async def scenario() -> None:
        channel = Channel()
        persona = Persona(complete=False)
        result = await exercise(channel, persona, Storage(), max_turns=10)
        assert result.termination_reason == "max_turns"
        assert result.turn_count == 10
        assert len(channel.sent) == 10

    asyncio.run(scenario())


def test_channel_and_model_failures_are_sanitized_and_closed() -> None:
    async def scenario() -> None:
        for channel, persona, expected in [
            (Channel(failure="start"), Persona(), "channel_error"),
            (Channel(failure="send"), Persona(), "channel_error"),
            (Channel(), Persona(failure=True), "model_error"),
        ]:
            storage = Storage()
            result = await exercise(channel, persona, storage)
            assert result.termination_reason == expected
            assert result.turn_count == 0
            assert channel.closed == 1
            assert result.error is not None
            assert "sensitive" not in str(result.model_dump())
            if expected == "model_error":
                assert result.error.reason_code == "unknown_model_failure"
            assert [event.type for event in result.committed_events][-2:] == [
                "operation_failed",
                "run_lifecycle_changed",
            ]

    asyncio.run(scenario())


def test_model_failure_diagnostic_preserves_only_allowlisted_code() -> None:
    from syntheticcx.adapters.persona.tinytroupe import _model_failure_reason

    assert _model_failure_reason(TimeoutError("credential-bearing text")) == "provider_timeout"
    assert _model_failure_reason(ValueError("private response text")) == "persona_adapter_error"
    _, detail = _failure(
        ModelError(
            "persona.respond",
            "persona operation failed",
            reason_code="provider_http_4xx",
        ),
        "runtime.execute",
    )
    assert detail.reason_code == "provider_http_4xx"
    assert "private" not in str(detail.model_dump())


def test_screenshot_capture_requires_policy_and_is_acknowledged_before_event() -> None:
    class CapturingChannel(Channel):
        captured = 0

        async def capture_failure_screenshot(self) -> bytes:
            assert self.closed == 0
            self.captured += 1
            return PNG

    class ScreenshotStorage(Storage):
        saved = 0

        async def write_screenshot(
            self, run_id: str, relative_path: str, content: bytes
        ) -> ArtifactRecord:
            assert self.created == 1 and run_id == "run-1"
            assert relative_path == "artifacts/screenshots/channel-failure.png"
            assert content == PNG
            self.saved += 1
            return ArtifactRecord(
                path=relative_path,
                media_type="image/png",
                content_hash="ack-hash",
                status="captured",
            )

        async def append_event(self, run_id: str, event: TypedTraceEvent) -> None:
            if event.type == "artifact_recorded" and event.data.status == "captured":
                assert self.saved == 1
            await super().append_event(run_id, event)

    async def scenario() -> None:
        for enabled, status, captures, writes in [
            (False, "omitted", 0, 0),
            (True, "captured", 1, 1),
        ]:
            channel = CapturingChannel(failure="send")
            storage = ScreenshotStorage()
            result = await exercise(
                channel,
                Persona(),
                storage,
                project_config=capture_config() if enabled else None,
            )
            artifacts = [
                event.data for event in result.committed_events if event.type == "artifact_recorded"
            ]
            assert len(artifacts) == 1 and artifacts[0].status == status
            assert channel.captured == captures and storage.saved == writes
            assert channel.closed == storage.closed == 1
            assert b"synthetic-fixture-bytes" not in str(result.model_dump()).encode()
            assert [event.type for event in result.committed_events][-3:] == [
                "artifact_recorded",
                "operation_failed",
                "run_lifecycle_changed",
            ]

    asyncio.run(scenario())


@pytest.mark.parametrize("enabled", [False, True])
def test_capture_policy_projects_real_failure_artifacts_and_omissions(
    tmp_path: Path, enabled: bool
) -> None:
    class FailingChannel(Channel):
        async def capture_failure_screenshot(self) -> bytes:
            assert self.closed == 0
            return PNG

    class CapturingPersona(Persona):
        def __init__(self, capture: PromptCapture | None) -> None:
            super().__init__()
            self.capture = capture
            self.interactions: list[ModelInteraction] = []

        async def respond(
            self, observation: ConversationObservation, state: PersonaState
        ) -> PersonaAction:
            self.interactions.append(
                ModelInteraction(
                    id="capture-call",
                    identity=ModelIdentity(provider="fake", model="fake"),
                    started_at=NOW,
                    ended_at=NOW,
                    usage=TokenUsage(unavailable_reason="offline test"),
                    captured_prompt=(
                        self.capture.sanitize(
                            [{"role": "user", "content": "canary-credential account SYNTHETIC-001"}]
                        )
                        if self.capture is not None
                        else None
                    ),
                    capture_omissions=["prompts_not_captured"] if self.capture is None else [],
                )
            )
            return await super().respond(observation, state)

        def take_interactions(self) -> tuple[ModelInteraction, ...]:
            values = tuple(self.interactions)
            self.interactions.clear()
            return values

    async def scenario() -> None:
        config = (
            capture_config() if enabled else load_config(Path("examples/telecom/scenario.yaml"))
        )
        if enabled:
            source = config.model_dump(mode="json")
            source["privacy"]["capture_prompts"] = True
            config = ProjectConfig.model_validate(source)
        redactor = Redactor(config.privacy, resolved_secrets=["canary-credential"])
        backend = FilesystemStorage(tmp_path, redactor, timeout_seconds=5, project_config=config)
        clock = ManualClock(NOW)
        journal = RunJournal("run-1", backend, redactor, clock)
        service = EvaluationService(
            journal=journal,
            wait=ManualWaitController(clock),
            evaluator_timeout_seconds=5,
            redactor=redactor,
            evaluators={"semantic_goal_completion": GoalEvaluator()},
        )
        channel = FailingChannel(failure="send")
        capture = PromptCapture(redactor, config.scenario) if enabled else None
        result = await run_evaluated(
            run=run(),
            config_artifact=ConfigArtifact(
                source_content_hash="hash", effective_config={"credential": "canary-credential"}
            ),
            scenario=Scenario.model_validate(config.scenario.model_dump()),
            persona=PersonaDefinition.model_validate(config.persona.model_dump()),
            channel=channel,
            simulator=CapturingPersona(capture),
            storage=backend,
            journal=journal,
            clock=clock,
            wait=ManualWaitController(clock),
            limits=config.execution.model_copy(update={"greeting_grace_seconds": 0}),
            evaluation=service,
            redactor=redactor,
            project_config=config,
        )
        directory = tmp_path / "run-1"
        metadata = json.loads((directory / "metadata.json").read_text())
        report = json.loads((directory / "report.json").read_text())
        assert result.outcome == report["outcome"] == "error"
        assert channel.closed == 1
        image = directory / "artifacts/screenshots/channel-failure.png"
        records = [record for record in report["artifacts"] if record["media_type"] == "image/png"]
        assert len(records) == 1 and records == [
            record for record in metadata["artifacts"] if record["media_type"] == "image/png"
        ]
        if enabled:
            assert image.read_bytes() == PNG
            assert records[0]["status"] == "captured"
            assert records[0]["content_hash"] == sha256(PNG).hexdigest()
            trace = [
                json.loads(line) for line in (directory / "trace.jsonl").read_text().splitlines()
            ]
            captured = [
                event["data"]["captured_prompt"]
                for event in trace
                if event["type"] == "model_interaction_completed"
            ]
            assert len(captured) == 1
            assert "<redacted:secret>" in captured[0]
            assert "<redacted:account_number>" in captured[0]
            assert "prompts_not_captured" not in metadata["metadata"]["capture_omissions"]
        else:
            assert not image.exists()
            assert records[0]["status"] == "omitted"
            assert "screenshot:screenshots_disabled" in metadata["metadata"]["capture_omissions"]
            assert "screenshot:screenshots_disabled" in report["omissions"]
            assert "prompts_not_captured" in metadata["metadata"]["capture_omissions"]
        assert all(
            "canary-credential" not in path.read_text()
            for path in directory.rglob("*")
            if path.is_file() and path.suffix in {".json", ".jsonl", ".yaml"}
        )

    asyncio.run(scenario())


def test_model_failure_does_not_capture_browser_image() -> None:
    class CapturingChannel(Channel):
        async def capture_failure_screenshot(self) -> bytes:
            raise AssertionError("model failures do not request browser screenshots")

    async def scenario() -> None:
        result = await exercise(
            CapturingChannel(),
            Persona(failure=True),
            Storage(),
            project_config=capture_config(),
        )
        assert result.termination_reason == "model_error"
        assert not any(event.type == "artifact_recorded" for event in result.committed_events)

    asyncio.run(scenario())


def test_browser_journey_timeout_captures_before_close() -> None:
    class HangingChannel(Channel):
        async def capture_failure_screenshot(self) -> bytes:
            assert self.closed == 0
            return PNG

    class ScreenshotStorage(Storage):
        async def write_screenshot(
            self, run_id: str, relative_path: str, content: bytes
        ) -> ArtifactRecord:
            assert content == PNG
            return ArtifactRecord(
                path=relative_path,
                media_type="image/png",
                content_hash="ack-hash",
                status="captured",
            )

    async def scenario() -> None:
        clock = ManualClock(NOW)
        channel = HangingChannel(failure="hang")
        task = asyncio.create_task(
            exercise(
                channel,
                Persona(),
                ScreenshotStorage(),
                clock=clock,
                timeout=2,
                project_config=capture_config(),
            )
        )
        await channel.receiving.wait()
        await clock.advance(2)
        result = await task
        assert result.termination_reason == "journey_timeout"
        assert channel.closed == 1
        assert any(
            event.type == "artifact_recorded" and event.data.status == "captured"
            for event in result.committed_events
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["capture", "write"])
def test_failed_screenshot_capture_records_truthful_failure(failure: str) -> None:
    class ChannelWithCapture(Channel):
        async def capture_failure_screenshot(self) -> bytes:
            if failure == "capture":
                raise OSError("secret pixels")
            return PNG

    class StorageWithWriter(Storage):
        async def write_screenshot(
            self, run_id: str, relative_path: str, content: bytes
        ) -> ArtifactRecord:
            if failure == "write":
                raise OSError("secret path")
            raise AssertionError("write must not follow failed capture")

    async def scenario() -> None:
        channel = ChannelWithCapture(failure="send")
        result = await exercise(
            channel,
            Persona(),
            StorageWithWriter(),
            project_config=capture_config(),
        )
        artifacts = [
            event.data for event in result.committed_events if event.type == "artifact_recorded"
        ]
        assert len(artifacts) == 1
        assert artifacts[0].status == "failed"
        assert artifacts[0].reason == (
            "capture_failed" if failure == "capture" else "storage_failed"
        )
        assert result.termination_reason == "channel_error"
        assert "secret" not in str(result.model_dump())
        assert channel.closed == 1

    asyncio.run(scenario())


def test_cancellation_during_failure_capture_finishes_bounded_cleanup() -> None:
    class SlowCapture(Channel):
        def __init__(self) -> None:
            super().__init__(failure="send")
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def capture_failure_screenshot(self) -> bytes:
            self.entered.set()
            await self.release.wait()
            return PNG

    class ScreenshotStorage(Storage):
        async def write_screenshot(
            self, run_id: str, relative_path: str, content: bytes
        ) -> ArtifactRecord:
            return ArtifactRecord(
                path=relative_path,
                media_type="image/png",
                content_hash="ack-hash",
                status="captured",
            )

    async def scenario() -> None:
        channel = SlowCapture()
        result_task = asyncio.create_task(
            exercise(
                channel,
                Persona(),
                ScreenshotStorage(),
                project_config=capture_config(),
            )
        )
        await channel.entered.wait()
        result_task.cancel()
        channel.release.set()
        result = await result_task
        assert result.termination_reason == "cancelled"
        assert result.cleanup_completed
        assert channel.closed == 1
        assert [event.type for event in result.committed_events][-1] == "run_lifecycle_changed"
        assert any(event.type == "artifact_recorded" for event in result.committed_events)

    asyncio.run(scenario())


def test_persistence_error_and_secondary_close_error_do_not_hide_primary() -> None:
    async def scenario() -> None:
        storage = Storage(fail_append=True)
        channel = Channel(close_failure=True)
        result = await exercise(channel, Persona(), storage)
        assert result.termination_reason == "persistence_error"
        assert result.error is not None and result.error.category == "persistence"
        assert not result.cleanup_completed
        assert channel.closed == 1
        assert result.committed_events == ()

    asyncio.run(scenario())


def test_owner_append_failure_stops_evaluation_and_attempts_one_bounded_failure_record() -> None:
    class CountingJournal(RunJournal):
        def __init__(self, storage: Storage, clock: ManualClock) -> None:
            super().__init__("run-1", storage, Redactor(PrivacyConfig()), clock)
            self.failure_attempts = 0

        async def append(self, data):
            if isinstance(data, OperationFailure):
                self.failure_attempts += 1
            return await super().append(data)

    class NeverEvaluate:
        async def evaluate(self, scenario: Scenario):
            raise AssertionError("evaluation must not run after failed evidence append")

    async def scenario() -> None:
        config = load_config(Path("examples/telecom/scenario.yaml"))
        storage = Storage(fail_at_sequence=2)
        channel = Channel(close_failure=True)
        clock = ManualClock(NOW)
        journal = CountingJournal(storage, clock)
        result = await run_evaluated(
            run=run(),
            config_artifact=ConfigArtifact(source_content_hash="hash", effective_config={}),
            scenario=Scenario.model_validate(config.scenario.model_dump()),
            persona=PersonaDefinition.model_validate(config.persona.model_dump()),
            channel=channel,
            simulator=Persona(),
            storage=storage,
            journal=journal,
            clock=clock,
            wait=ManualWaitController(clock),
            limits=config.execution.model_copy(update={"greeting_grace_seconds": 0}),
            evaluation=NeverEvaluate(),
            redactor=Redactor(PrivacyConfig()),
            project_config=config,
        )
        assert not result.finalized and result.outcome is None and result.exit_code == 3
        assert result.conversation.termination_reason == "persistence_error"
        assert result.evaluations == () and storage.finalized == 0
        assert journal.failure_attempts == 1
        assert [event.type for event in result.committed_events] == [
            "run_lifecycle_changed",
            "conversation_started",
        ]
        assert channel.closed == storage.closed == 1
        assert "sensitive-storage-path" not in str(result.model_dump())

    asyncio.run(asyncio.wait_for(scenario(), timeout=1))


def test_successful_receipt_remains_counted_when_customer_evidence_fails() -> None:
    async def scenario() -> None:
        storage = Storage(fail_customer=True)
        channel = Channel()
        result = await exercise(channel, Persona(), storage)
        assert channel.sent == ["My account number is 123"]
        assert result.termination_reason == "persistence_error"
        assert result.turn_count == result.persona_state.turn_number == 1
        assert result.error is not None and result.error.category == "persistence"

    asyncio.run(scenario())


def test_close_failure_is_secondary_to_successful_graph_termination() -> None:
    async def scenario() -> None:
        result = await exercise(Channel(close_failure=True), Persona(), Storage())
        assert result.termination_reason == "goal_believed_complete"
        assert not result.cleanup_completed
        assert result.error is not None and result.error.operation == "channel.close"
        assert any(event.type == "operation_failed" for event in result.committed_events)

    asyncio.run(scenario())


def test_close_deadline_is_bounded_and_attempted_once() -> None:
    async def scenario() -> None:
        clock = ManualClock(NOW)
        channel = Channel(close_hangs=True)
        task = asyncio.create_task(exercise(channel, Persona(), Storage(), clock=clock))
        await channel.closing.wait()
        await clock.advance(5)
        result = await task
        assert result.termination_reason == "goal_believed_complete"
        assert result.error is not None and result.error.operation == "channel.close"
        assert result.cleanup_completed is False
        assert channel.closed == 1

    asyncio.run(scenario())


def test_journey_timeout_and_cancellation_close_started_resources_once() -> None:
    async def scenario() -> None:
        clock = ManualClock(NOW)
        channel = Channel(failure="hang")
        task = asyncio.create_task(exercise(channel, Persona(), Storage(), clock=clock, timeout=2))
        await channel.receiving.wait()
        await clock.advance(2)
        timed_out = await task
        assert timed_out.termination_reason == "journey_timeout"
        assert timed_out.turn_count == 1
        assert channel.sent == ["My account number is 123"]
        assert channel.closed == 1

        other = Channel(failure="hang")
        task = asyncio.create_task(exercise(other, Persona(), Storage()))
        await other.receiving.wait()
        task.cancel()
        cancelled = await task
        assert cancelled.termination_reason == "cancelled"
        assert other.closed == 1
        assert cancelled.cleanup_completed

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("mode", "reason", "turns", "sends", "error_category"),
    [
        ("goal_initial", "goal_believed_complete", 0, 0, None),
        ("goal_after_response", "goal_believed_complete", 1, 1, None),
        ("max_turns", "max_turns", 1, 1, None),
        ("abandoned_initial", "abandonment", 0, 0, None),
        ("start_failure", "channel_error", 0, 0, "channel"),
        ("send_failure", "channel_error", 0, 0, "channel"),
        ("receive_failure", "channel_error", 1, 1, "channel"),
        ("model_failure", "model_error", 0, 0, "model"),
        ("journal_failure", "persistence_error", 1, 1, "persistence"),
    ],
)
def test_terminal_state_matrix(
    mode: str, reason: str, turns: int, sends: int, error_category: str | None
) -> None:
    async def scenario() -> None:
        config = load_config(Path("examples/telecom/scenario.yaml"))
        definition = PersonaDefinition.model_validate(config.persona.model_dump())
        if mode == "abandoned_initial":
            definition = definition.model_copy(
                update={
                    "initial_state": definition.initial_state.model_copy(update={"abandoned": True})
                }
            )
        channel = Channel(
            failure={
                "start_failure": "start",
                "send_failure": "send",
                "receive_failure": "receive",
            }.get(mode)
        )
        persona = Persona(
            failure=mode == "model_failure",
            complete=mode != "max_turns",
            goal_initial=mode == "goal_initial",
        )
        storage = Storage(fail_customer=mode == "journal_failure")
        result = await exercise(
            channel,
            persona,
            storage,
            max_turns=2 if mode == "goal_after_response" else 1,
            definition=definition,
        )
        assert result.termination_reason == reason
        assert result.turn_count == result.persona_state.turn_number == turns
        assert len(channel.sent) == sends
        assert channel.closed == storage.closed == 1
        assert result.cleanup_completed
        if error_category is None:
            assert result.error is None
        else:
            assert result.error is not None
            assert result.error.category == error_category
        if mode == "abandoned_initial":
            assert persona.calls == 0
        if mode != "journal_failure":
            assert result.committed_events[-1].type == "run_lifecycle_changed"
            assert result.committed_events[-1].data.termination_reason == reason
            assert result.committed_events[-1].data.outcome is None

    asyncio.run(scenario())


@pytest.mark.parametrize("echo", [None, "My account number is 123", "different echo"])
def test_canonical_customer_content_does_not_depend_on_echo(echo: str | None) -> None:
    async def scenario() -> None:
        result = await exercise(Channel(echo=echo), Persona(), Storage())
        customer = [
            event.data
            for event in result.committed_events
            if event.type == "message_observed" and event.data.message.actor == "customer"
        ]
        assert len(customer) == 1
        assert customer[0].message.id == "sent-1"
        assert customer[0].message.content == "My account number is 123"
        assert customer[0].identity_evidence == ({"echo": echo} if echo is not None else {})

    asyncio.run(scenario())


def test_initial_greeting_and_no_greeting_have_distinct_observations() -> None:
    class GreetingChannel(Channel):
        async def receive(self) -> CompletedResponse:
            return CompletedResponse(
                messages=[
                    Message(
                        id="greeting-1",
                        actor="assistant",
                        sequence=1,
                        content="Hello",
                        observed_at=NOW,
                    ),
                    Message(
                        id="greeting-2",
                        actor="assistant",
                        sequence=2,
                        content="How can I help?",
                        observed_at=NOW,
                    ),
                ],
                started_at=NOW,
                completed_at=NOW,
                completion_strategy="completion_locator",
            )

    async def scenario() -> None:
        for grace, expected in [(0, False), (2, True)]:
            result = await exercise(
                GreetingChannel(),
                Persona(goal_initial=True),
                Storage(),
                greeting_grace=grace,
            )
            events = result.committed_events
            started = next(event.data for event in events if event.type == "conversation_started")
            assert started.greeting_observed is expected
            assert started.greeting_message_ids == (
                ["greeting-1", "greeting-2"] if expected else []
            )
            assert result.turn_count == 0

    asyncio.run(scenario())


def test_result_is_frozen_and_detached_from_storage() -> None:
    async def scenario() -> None:
        storage = Storage()
        result = await exercise(Channel(), Persona(), storage)
        with pytest.raises(ValidationError):
            result.turn_count = 999
        stored = next(event for event in storage.events if event.type == "message_observed")
        stored.data.message.metadata["external"] = "changed"
        frozen = next(event for event in result.committed_events if event.id == stored.id)
        assert "external" not in frozen.data.message.metadata
        assert isinstance(result.committed_events, tuple)

    asyncio.run(scenario())


def test_creation_failure_skips_channel_and_closes_storage_once() -> None:
    async def scenario() -> None:
        channel = Channel()
        storage = Storage(fail_create=True, fail_append=True)
        result = await exercise(channel, Persona(), storage)
        assert result.termination_reason == "persistence_error"
        assert result.error is not None and result.error.category == "persistence"
        assert channel.started == channel.closed == 0
        assert storage.created == storage.closed == 1
        assert result.committed_events == ()

    asyncio.run(scenario())


def test_multipart_repetition_is_committed_once_before_goal_decision() -> None:
    class MultipartChannel(Channel):
        async def receive(self) -> CompletedResponse:
            return CompletedResponse(
                messages=[
                    Message(
                        id="assistant-1",
                        actor="assistant",
                        sequence=1,
                        content="Please provide the account ID",
                        observed_at=NOW,
                    ),
                    Message(
                        id="assistant-2",
                        actor="assistant",
                        sequence=2,
                        content="What is the account number?",
                        observed_at=NOW,
                    ),
                ],
                started_at=NOW,
                completed_at=NOW,
                completion_strategy="completion_locator",
            )

    class ObservingPersona(Persona):
        def __init__(self) -> None:
            super().__init__()
            self.observations: list[ConversationObservation] = []
            self.states: list[PersonaState] = []

        async def respond(
            self, observation: ConversationObservation, state: PersonaState
        ) -> PersonaAction:
            self.observations.append(observation)
            self.states.append(state)
            return await super().respond(observation, state)

    async def scenario() -> None:
        persona = ObservingPersona()
        result = await exercise(MultipartChannel(), persona, Storage())
        assert result.termination_reason == "goal_believed_complete"
        assert len(persona.observations) == 2
        assert [message.id for message in persona.observations[-1].messages] == [
            "assistant-1",
            "assistant-2",
        ]
        assert persona.states[-1].frustration == pytest.approx(0.3)
        events = result.committed_events
        repetitions = [event for event in events if event.type == "repeated_information"]
        assert len(repetitions) == 1
        repetition = repetitions[0]
        assert repetition.data.original_message_id == "sent-1"
        assert repetition.data.request_message_id == "assistant-1"
        signal = next(
            event
            for event in events
            if event.type == "behavioral_signal_recorded"
            and event.data.signal.signal == "repeated_information"
        )
        assert repetition.id in signal.data.signal.source_event_ids
        transition = next(
            event
            for event in events
            if event.type == "behavioral_state_changed" and event.data.signal_event_id == signal.id
        )
        assert transition.sequence > repetition.sequence
        assert transition.data.after.frustration == pytest.approx(0.3)
        assert [event.sequence for event in events] == list(range(len(events)))
        assert not any(event.type == "evaluation_completed" for event in events)

    asyncio.run(scenario())


def test_max_turns_never_calls_persona_after_final_assistant_response() -> None:
    class FailingAfterFirstCall(Persona):
        async def respond(
            self, observation: ConversationObservation, state: PersonaState
        ) -> PersonaAction:
            if self.calls:
                raise OSError("unexpected extra model invocation")
            return await super().respond(observation, state)

    async def scenario() -> None:
        persona = FailingAfterFirstCall(complete=False)
        result = await exercise(Channel(), persona, Storage(), max_turns=1)
        assert result.termination_reason == "max_turns"
        assert result.error is None
        assert result.turn_count == 1
        assert persona.calls == 1

    asyncio.run(scenario())


def test_journal_append_uses_persistence_deadline_not_journey_deadline() -> None:
    class BlockingStorage(Storage):
        def __init__(self) -> None:
            super().__init__()
            self.appending = asyncio.Event()

        async def append_event(self, run_id: str, event: TypedTraceEvent) -> None:
            if event.sequence == 1:
                self.appending.set()
                await asyncio.Event().wait()
            await super().append_event(run_id, event)

    async def scenario() -> None:
        clock = ManualClock(NOW)
        storage = BlockingStorage()
        channel = Channel()
        task = asyncio.create_task(exercise(channel, Persona(), storage, clock=clock, timeout=20))
        await storage.appending.wait()
        await clock.advance(5)
        result = await task
        assert result.termination_reason == "persistence_error"
        assert result.error is not None and result.error.category == "persistence"
        assert clock.now() == NOW + timedelta(seconds=5)
        assert result.turn_count == 0
        assert [event.type for event in result.committed_events] == ["run_lifecycle_changed"]
        assert channel.closed == storage.closed == 1

    asyncio.run(scenario())


def test_cancellation_during_channel_cleanup_returns_result_and_closes_storage() -> None:
    async def scenario() -> None:
        clock = ManualClock(NOW)
        channel = Channel(close_hangs=True)
        storage = Storage()
        task = asyncio.create_task(exercise(channel, Persona(), storage, clock=clock))
        await channel.closing.wait()
        task.cancel()
        await clock.advance(5)
        result = await task
        assert result.termination_reason == "cancelled"
        assert not result.cleanup_completed
        assert result.error is not None and result.error.category == "cancelled"
        assert result.committed_events[-1].data.termination_reason == "cancelled"
        assert channel.closed == storage.closed == 1

    asyncio.run(scenario())


def test_cancellation_during_storage_cleanup_preserves_committed_termination() -> None:
    class BlockingCloseStorage(Storage):
        def __init__(self) -> None:
            super().__init__()
            self.closing = asyncio.Event()

        async def close(self) -> None:
            self.closed += 1
            self.closing.set()
            await asyncio.Event().wait()

    async def scenario() -> None:
        clock = ManualClock(NOW)
        storage = BlockingCloseStorage()
        channel = Channel()
        # The helper asserts close once and committed events on return.
        task = asyncio.create_task(exercise(channel, Persona(), storage, clock=clock))
        await storage.closing.wait()
        task.cancel()
        await clock.advance(5)
        result = await task
        assert result.termination_reason == "goal_believed_complete"
        assert result.committed_events[-1].data.termination_reason == result.termination_reason
        assert result.error is not None and result.error.category == "cancelled"
        assert not result.cleanup_completed
        assert channel.closed == storage.closed == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("failed_boundary", ["greeting", "customer_turn_completed"])
def test_behavior_failure_propagates_through_owner_as_infrastructure_error(failed_boundary) -> None:
    async def exercise():
        config = load_config(Path("examples/telecom/scenario.yaml"))
        definition = PersonaDefinition.model_validate(config.persona.model_dump()).model_copy(
            update={"behavioral_rules": []}
        )
        selected = BehaviorConfig.model_validate(
            {"detectors": [{"id": "consumer", "type": "custom"}]}
        )

        class Detector:
            async def detect(self, boundary):
                if boundary.kind == failed_boundary:
                    raise RuntimeError("private detector text must not escape")
                return ()

        registered = DetectorRegistration(
            detector_type="custom",
            version="1",
            parameter_model=EmptyDetectorParameters,
            signal_schemas={"notice": EmptyDetectorParameters},
            mode="deterministic",
            factory=lambda metadata, parameters: Detector(),
        )
        detectors = prepare_behavior(selected, [], {"custom": registered}).instantiate()
        storage, channel, simulator = Storage(), Channel(), Persona(complete=False)
        clock = ManualClock(NOW)
        wait = ManualWaitController(clock)
        redactor = Redactor(PrivacyConfig())
        journal = RunJournal("run-1", storage, redactor, clock)
        service = EvaluationService(
            journal=journal,
            wait=wait,
            evaluator_timeout_seconds=5,
            redactor=redactor,
            evaluators={"semantic_goal_completion": GoalEvaluator()},
        )
        result = await run_evaluated(
            run=run(),
            config_artifact=ConfigArtifact(source_content_hash="hash", effective_config={}),
            scenario=Scenario.model_validate(config.scenario.model_dump()),
            persona=definition,
            channel=channel,
            simulator=simulator,
            storage=storage,
            journal=journal,
            clock=clock,
            wait=wait,
            limits=config.execution.model_copy(update={"greeting_grace_seconds": 0}),
            evaluation=service,
            redactor=redactor,
            behavior_detectors=detectors,
        )
        assert result.finalized and result.outcome == "error" and result.exit_code == 3
        assert result.conversation.termination_reason == "behavior_error"
        assert result.conversation.error.category == "behavior"
        assert result.conversation.error.operation == "behavior.detect"
        assert len(channel.sent) == (0 if failed_boundary == "greeting" else 1)
        assert simulator.calls == len(channel.sent)
        assert channel.closed == storage.closed == 1
        assert "private detector text" not in "\n".join(
            event.model_dump_json() for event in result.committed_events
        )
        assert not any(
            event.type == "behavioral_signal_recorded" for event in result.committed_events
        )

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("route_length", "journey_timeout", "expected_reason"),
    [(0, 200, "channel_error"), (4, 200, "goal_believed_complete"), (4, 5, "journey_timeout")],
)
def test_startup_allows_replaying_multiple_actions_but_retains_journey_deadline(
    route_length, journey_timeout, expected_reason
):
    class StartingChannel(Channel):
        def __init__(self):
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def start(self):
            self.started += 1
            self.entered.set()
            await self.release.wait()

    async def scenario():
        from syntheticcx.config.models import WebsiteNavigationStep

        config = load_config(Path("examples/telecom/scenario.yaml"))
        config = config.model_copy(
            update={
                "channel": config.channel.model_copy(
                    update={
                        "navigation_steps": [WebsiteNavigationStep(action="wait")] * route_length,
                    }
                ),
            }
        )
        clock = ManualClock(NOW)
        channel = StartingChannel()
        operation = asyncio.create_task(
            exercise(
                channel,
                Persona(),
                Storage(),
                clock=clock,
                project_config=config,
                timeout=journey_timeout,
            )
        )
        await channel.entered.wait()
        await clock.advance(config.execution.browser_action_timeout_seconds + 1)
        for _ in range(5):
            await asyncio.sleep(0)
        channel.release.set()
        result = await operation
        assert result.termination_reason == expected_reason
        assert channel.closed == 1
        assert bool(channel.sent) == (expected_reason == "goal_believed_complete")

    asyncio.run(scenario())

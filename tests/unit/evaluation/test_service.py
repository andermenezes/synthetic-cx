"""Bounded, ordered criterion execution over a single durable journal."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from syntheticcx.config.loader import load_config
from syntheticcx.config.models import Criterion, PrivacyConfig
from syntheticcx.domain.artifacts import ConfigArtifact, EvaluationContext, Run
from syntheticcx.domain.events import (
    EvaluationCompletedEvent,
    MessageObserved,
    ModelInteractionCompletedEvent,
    RunLifecycleChange,
    TypedTraceEvent,
)
from syntheticcx.domain.models import (
    EvaluationResult,
    LLMRequest,
    LLMResponse,
    Message,
    ModelIdentity,
    ModelInteraction,
    Scenario,
    TokenUsage,
)
from syntheticcx.errors import ConfigurationError, ModelError, PersistenceError
from syntheticcx.evaluation.service import EvaluationService
from syntheticcx.privacy import Redactor
from syntheticcx.runtime.clock import RealClock, RealWaitController
from syntheticcx.runtime.journal import RunJournal

NOW = datetime(2026, 9, 27, tzinfo=UTC)
SCENARIO = Scenario.model_validate(
    load_config(Path("examples/telecom/scenario.yaml")).scenario.model_dump()
)


class MemoryStorage:
    def __init__(self, *, fail_evaluation_append: bool = False) -> None:
        self.events: list[TypedTraceEvent] = []
        self.fail_evaluation_append = fail_evaluation_append

    async def create_run(self, run: Run, config_artifact: ConfigArtifact) -> None:
        pass

    async def append_event(self, run_id: str, event: TypedTraceEvent) -> None:
        if self.fail_evaluation_append and isinstance(event, EvaluationCompletedEvent):
            raise PersistenceError("append event", "storage unavailable")
        self.events.append(event)

    async def finalize_run(self, run: Run) -> None:
        pass


class ScriptedEvaluator:
    def __init__(self, *actions: object) -> None:
        self.actions = actions
        self.calls = 0

    async def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        action = self.actions[self.calls]
        self.calls += 1
        if isinstance(action, Exception):
            raise action
        if action is None:
            return None  # type: ignore[return-value]  # Exercise a missing provider result.
        if isinstance(action, EvaluationResult):
            return action
        return EvaluationResult(
            evaluator_id=context.criterion.evaluator_type,
            criterion_id=context.criterion.id,
            required=context.criterion.required,
            status=action,
            explanation="Observed committed evidence.",
            evidence_event_ids=[context.sanitized_trace[0].id],
        )


class InteractionThenWaitEvaluator:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self._interactions: list[ModelInteraction] = []

    def take_interactions(self) -> tuple[ModelInteraction, ...]:
        interactions = tuple(self._interactions)
        self._interactions.clear()
        return interactions

    async def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        self._interactions.append(
            ModelInteraction(
                id="model-cancelled",
                identity=ModelIdentity(provider="openai", model="evaluator"),
                started_at=NOW,
                ended_at=NOW,
                usage=TokenUsage(unavailable_reason="not reported"),
            )
        )
        self.started.set()
        await self.release.wait()
        raise AssertionError("test must cancel evaluation before it returns")


def criterion(identifier: str, *, required: bool = True) -> Criterion:
    return Criterion(
        id=identifier,
        evaluator_type="journey_timeout",
        result_kind="boolean",
        required=required,
        evidence_policy="required",
    )


def scenario(*criteria: Criterion) -> Scenario:
    return SCENARIO.model_copy(update={"success_criteria": list(criteria)})


def setup(
    *,
    evaluators=None,
    provider=None,
    storage: MemoryStorage | None = None,
    timeout: float = 1,
) -> tuple[EvaluationService, RunJournal, MemoryStorage]:
    backend = storage or MemoryStorage()
    wait = RealWaitController()
    journal = RunJournal("run-1", backend, Redactor(PrivacyConfig()), RealClock())
    journal.configure_persistence(wait, 1)
    service = EvaluationService(
        journal=journal,
        wait=wait,
        evaluator_timeout_seconds=timeout,
        redactor=Redactor(PrivacyConfig()),
        provider=provider,
        evaluators=evaluators,
    )
    return service, journal, backend


async def seed(journal: RunJournal) -> None:
    await journal.append(RunLifecycleChange(current="incomplete", termination_reason="max_turns"))


def run(service: EvaluationService, journal: RunJournal, configured: Scenario):
    async def exercise():
        await seed(journal)
        return await service.evaluate(configured)

    return asyncio.run(exercise())


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (TimeoutError(), "passed", ["passed"]),
        (ModelError("model.generate", "429", retryable=True), "passed", ["passed"]),
        (ModelError("model.generate", "temporary", retryable=True), TimeoutError(), ["error"]),
    ],
)
def test_optional_transient_failure_gets_exactly_one_retry(
    first: Exception, second: object, expected: list[str]
) -> None:
    scripted = ScriptedEvaluator(first, second)
    service, journal, backend = setup(evaluators={"journey_timeout": scripted})
    results = run(service, journal, scenario(criterion("optional", required=False)))
    assert scripted.calls == 2
    assert [result.status for result in results] == expected
    assert (
        len([event for event in backend.events if isinstance(event, EvaluationCompletedEvent)]) == 1
    )
    if results[0].status == "error":
        assert results[0].metadata["attempts"] == 2


@pytest.mark.parametrize(
    "failure",
    [
        ModelError("model.generate", "bad request"),
        ValueError("malformed"),
        ConfigurationError("config", "invalid"),
    ],
)
def test_optional_permanent_failure_is_not_retried(failure: Exception) -> None:
    scripted = ScriptedEvaluator(failure)
    service, journal, _ = setup(evaluators={"journey_timeout": scripted})
    result = run(service, journal, scenario(criterion("optional", required=False)))[0]
    assert result.status == "error"
    assert scripted.calls == 1
    assert str(failure) not in result.explanation


def test_required_transient_failure_is_not_retried() -> None:
    scripted = ScriptedEvaluator(ModelError("model.generate", "rate limit", retryable=True))
    service, journal, _ = setup(evaluators={"journey_timeout": scripted})
    result = run(service, journal, scenario(criterion("required")))[0]
    assert result.status == "error"
    assert scripted.calls == 1


def test_actual_evaluator_deadline_bounds_an_optional_retry() -> None:
    class SlowEvaluator:
        def __init__(self) -> None:
            self.calls = 0

        async def evaluate(self, context: EvaluationContext) -> EvaluationResult:
            self.calls += 1
            await asyncio.sleep(0.1)
            raise AssertionError("the evaluator should have timed out")

    slow = SlowEvaluator()
    service, journal, _ = setup(evaluators={"journey_timeout": slow}, timeout=0.01)
    result = run(service, journal, scenario(criterion("optional", required=False)))[0]
    assert result.status == "error"
    assert slow.calls == 2


def test_missing_required_result_becomes_unavailable_and_later_result_still_commits() -> None:
    scripted = ScriptedEvaluator(None, "passed")
    service, journal, backend = setup(evaluators={"journey_timeout": scripted})
    results = run(service, journal, scenario(criterion("first"), criterion("second")))
    assert [result.status for result in results] == ["unavailable", "passed"]
    assert [
        event.data.result.criterion_id
        for event in backend.events
        if isinstance(event, EvaluationCompletedEvent)
    ] == ["first", "second"]


def test_invalid_evaluator_citation_is_visible_error_without_retry() -> None:
    invalid = EvaluationResult(
        evaluator_id="journey_timeout",
        criterion_id="required",
        required=True,
        status="passed",
        explanation="Invented event.",
        evidence_event_ids=["future-event"],
    )
    scripted = ScriptedEvaluator(invalid)
    service, journal, _ = setup(evaluators={"journey_timeout": scripted})
    result = run(service, journal, scenario(criterion("required")))[0]
    assert result.status == "error"
    assert scripted.calls == 1


def test_semantic_criterion_snapshot_redacts_known_sensitive_fact_values() -> None:
    expected = SCENARIO.success_criteria[0]
    hidden = expected.model_copy(
        update={"parameters": {"expected_facts": ["Private marker secret-42 explains the bill."]}}
    )
    configured = SCENARIO.model_copy(
        update={
            "starting_context": {"account_number": "secret-42"},
            "success_criteria": [hidden],
        }
    )
    scripted = ScriptedEvaluator(None)
    backend = MemoryStorage()
    redactor = Redactor(PrivacyConfig(redaction_keys=["account_number"]))
    wait = RealWaitController()
    journal = RunJournal("run-1", backend, redactor, RealClock())
    journal.configure_persistence(wait, 1)
    service = EvaluationService(
        journal=journal,
        wait=wait,
        evaluator_timeout_seconds=1,
        redactor=redactor,
        evaluators={"semantic_goal_completion": scripted},
    )
    result = run(service, journal, configured)[0]
    assert result.status == "unavailable"
    assert "secret-42" not in backend.events[-1].model_dump_json()


def test_persistence_failure_stops_remaining_evaluation() -> None:
    backend = MemoryStorage(fail_evaluation_append=True)
    scripted = ScriptedEvaluator("passed", "passed")
    service, journal, _ = setup(evaluators={"journey_timeout": scripted}, storage=backend)
    with pytest.raises(PersistenceError):
        run(service, journal, scenario(criterion("first"), criterion("second")))
    assert scripted.calls == 1
    assert len(journal.committed_events) == 1


def test_cancellation_commits_drained_interaction_before_propagating() -> None:
    async def exercise() -> None:
        await seed(journal)
        task = asyncio.create_task(service.evaluate(scenario(criterion("required"))))
        await evaluator.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    evaluator = InteractionThenWaitEvaluator()
    service, journal, backend = setup(evaluators={"journey_timeout": evaluator})
    asyncio.run(exercise())
    assert [event.type for event in backend.events] == [
        "run_lifecycle_changed",
        "model_interaction_completed",
    ]


class Provider:
    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        transcript = json.loads(request.messages[1].content)["transcript"]
        assistant = next(item for item in transcript if item["actor"] == "assistant")
        return LLMResponse(
            content=json.dumps(
                {
                    "passed": True,
                    "explanation": "The plan increase is explained.",
                    "evidence": [
                        {"event_id": assistant["event_id"], "message_id": assistant["message_id"]}
                    ],
                }
            ),
            interaction=ModelInteraction(
                id="model-1",
                identity=ModelIdentity(provider="openai", model="evaluator"),
                started_at=NOW,
                ended_at=NOW,
                usage=TokenUsage(unavailable_reason="not reported"),
                capture_omissions=["prompts_not_captured"],
            ),
        )


def test_semantic_model_interaction_is_committed_before_result() -> None:
    async def exercise() -> tuple[EvaluationResult, ...]:
        await seed(journal)
        await journal.append(
            MessageObserved(
                message=Message(
                    id="message-1",
                    sequence=1,
                    actor="assistant",
                    content="The new plan costs 50 more.",
                    observed_at=NOW,
                )
            )
        )
        return await service.evaluate(SCENARIO)

    provider = Provider()
    service, journal, backend = setup(provider=provider)
    result = asyncio.run(exercise())[0]
    assert result.status == "passed"
    assert [event.type for event in backend.events] == [
        "run_lifecycle_changed",
        "message_observed",
        "model_interaction_completed",
        "evaluation_completed",
    ]
    interaction = backend.events[2]
    assert isinstance(interaction, ModelInteractionCompletedEvent)
    assert interaction.data.input_event_ids == [backend.events[1].id]
    assert interaction.data.output_event_ids == []
    assert "evaluation_result_not_yet_committed" in interaction.data.capture_omissions
    assert provider.requests[0].response_schema is not None

"""Projection derivation, referential integrity, and finalization ordering."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from syntheticcx.adapters.storage.filesystem import FilesystemStorage
from syntheticcx.config.models import Criterion, PrivacyConfig
from syntheticcx.domain.artifacts import ConfigArtifact, MetadataArtifact, Run
from syntheticcx.domain.events import (
    EvaluationCompleted,
    EvaluationCompletedEvent,
    MessageObserved,
    MessageObservedEvent,
    PersonaStateChangedEvent,
    RepeatedInformationEvent,
    RepeatedInformationObserved,
)
from syntheticcx.domain.models import (
    ArtifactRecord,
    EvaluationResult,
    Message,
    PersonaState,
    PersonaStateTransition,
)
from syntheticcx.errors import PersistenceError
from syntheticcx.privacy import Redactor
from syntheticcx.reporting.materialize import build_projections

NOW = datetime(2026, 9, 24, tzinfo=UTC)


def make_run(identifier: str = "run-1") -> Run:
    return Run.model_validate(
        {
            "id": identifier,
            "scenario_id": "billing",
            "persona_id": "standard",
            "started_at": NOW,
            "lifecycle_status": "in_progress",
            "metadata": {
                "configuration_version": "1",
                "framework_version": "0.1.0",
                "persona_model": {"provider": "openai", "model": "reference"},
                "evaluation_model": {"provider": "openai", "model": "reference"},
                "usage": {"unavailable_reason": "no calls"},
                "runtime_platform": "test",
                "owner": {
                    "owner_uuid": "owner",
                    "hostname": "test",
                    "pid": 1,
                    "process_start_evidence": "start-marker",
                },
            },
        }
    )


def make_event(sequence: int) -> MessageObservedEvent:
    return MessageObservedEvent(
        id=f"event-{sequence}",
        sequence=sequence,
        timestamp=NOW + timedelta(seconds=sequence),
        type="message_observed",
        data=MessageObserved(
            message=Message(
                id=f"message-{sequence}",
                sequence=sequence,
                actor="customer",
                content=f"turn {sequence}",
                observed_at=NOW + timedelta(seconds=sequence),
            )
        ),
    )


def make_evaluation(
    sequence: int, criterion_id: str, status: str = "passed"
) -> EvaluationCompletedEvent:
    return EvaluationCompletedEvent(
        id=f"event-{sequence}",
        sequence=sequence,
        timestamp=NOW + timedelta(seconds=sequence),
        type="evaluation_completed",
        data=EvaluationCompleted(
            result=EvaluationResult(
                evaluator_id="semantic_goal_completion",
                criterion_id=criterion_id,
                status=status,
                required=True,
                explanation="billing explained",
            ),
            criterion=Criterion(
                id=criterion_id,
                evaluator_type="semantic_goal_completion",
                result_kind="boolean",
                evidence_policy="required",
            ),
        ),
    )


def make_transition(sequence: int, trigger: str) -> PersonaStateChangedEvent:
    return PersonaStateChangedEvent(
        id=f"event-{sequence}",
        sequence=sequence,
        timestamp=NOW + timedelta(seconds=sequence),
        type="persona_state_changed",
        data=PersonaStateTransition(
            trigger_ids=[trigger],
            before=PersonaState(),
            after=PersonaState(frustration=0.2),
            applied_rule_ids=["rule-1"],
        ),
    )


def make_repetition(sequence: int) -> RepeatedInformationEvent:
    return RepeatedInformationEvent(
        id=f"event-{sequence}",
        sequence=sequence,
        timestamp=NOW + timedelta(seconds=sequence),
        type="repeated_information",
        data=RepeatedInformationObserved(
            information_id="account_number",
            original_message_id="message-0",
            request_message_id="message-1",
        ),
    )


def test_projections_derive_report_from_committed_evidence() -> None:
    events = [
        make_event(0),
        make_event(1),
        make_repetition(2),
        make_transition(3, "event-2"),
        make_evaluation(4, "billing-explanation"),
    ]
    projections = build_projections(
        "run-1",
        events,
        [("billing-explanation", "semantic_goal_completion", True)],
        materialize_conversation=True,
        materialize_evaluation=True,
        materialize_report=True,
    )
    assert projections.conversation is not None
    assert len(projections.conversation.root) == 2
    assert projections.evaluation is not None
    assert len(projections.evaluation.results) == 1
    assert projections.report is not None
    assert projections.report.turn_count == 2
    assert projections.report.repetition_count == 1
    assert projections.report.duration_seconds == 4.0
    assert projections.report.goal_result is not None
    assert projections.report.omissions == []


def test_transition_trigger_text_cannot_invent_repetition() -> None:
    projections = build_projections(
        "run-1",
        [make_transition(0, "repeated_information")],
        [],
        materialize_conversation=False,
        materialize_evaluation=False,
        materialize_report=True,
    )
    assert projections.report is not None
    assert projections.report.repetition_count == 0


def test_projections_preserve_criterion_contract_and_goal_type() -> None:
    criterion = Criterion(
        id="billing-explanation",
        evaluator_type="semantic_goal_completion",
        result_kind="scored",
        threshold=0.8,
        evidence_policy="optional",
        parameters={"prompt": "hidden"},
    )
    original = make_evaluation(0, criterion.id)
    result = original.data.result.model_copy(update={"evaluator_id": "custom-goal-evaluator"})
    event = original.model_copy(
        update={"data": original.data.model_copy(update={"result": result})}
    )
    projections = build_projections(
        "run-1",
        [event],
        [criterion],
        artifact_location="custom-runs/run-1",
        artifacts=[
            ArtifactRecord(path="trace.jsonl", media_type="application/json", status="captured")
        ],
        materialize_conversation=True,
        materialize_evaluation=True,
        materialize_report=True,
    )
    assert projections.evaluation is not None
    assert projections.evaluation.criteria == [criterion]
    assert projections.report is not None
    assert projections.report.goal_result == event.data.result
    assert projections.report.artifact_location == "custom-runs/run-1"
    assert projections.report.artifacts[0].path == "trace.jsonl"


def test_missing_required_criterion_fails_finalization() -> None:
    events = [make_event(0)]
    with pytest.raises(PersistenceError, match="required criterion"):
        build_projections(
            "run-1",
            events,
            [("billing-explanation", "semantic_goal_completion", True)],
            materialize_conversation=True,
            materialize_evaluation=True,
            materialize_report=True,
        )


def test_unknown_criterion_reference_fails_finalization() -> None:
    events = [make_evaluation(0, "unknown-criterion")]
    with pytest.raises(PersistenceError, match="unknown criterion"):
        build_projections(
            "run-1",
            events,
            [("billing-explanation", "semantic_goal_completion", True)],
            materialize_conversation=True,
            materialize_evaluation=True,
            materialize_report=True,
        )


def test_uncommitted_evidence_reference_fails_finalization() -> None:
    result = EvaluationResult(
        evaluator_id="semantic_goal_completion",
        criterion_id="billing-explanation",
        status="passed",
        required=True,
        explanation="ok",
        evidence_event_ids=["missing-event"],
    )
    event = EvaluationCompletedEvent(
        id="event-0",
        sequence=0,
        timestamp=NOW,
        type="evaluation_completed",
        data=EvaluationCompleted(
            result=result,
            criterion=Criterion(
                id="billing-explanation",
                evaluator_type="semantic_goal_completion",
                result_kind="boolean",
                evidence_policy="required",
            ),
        ),
    )
    with pytest.raises(PersistenceError, match="uncommitted evidence"):
        build_projections(
            "run-1",
            [event],
            [("billing-explanation", "semantic_goal_completion", True)],
            materialize_conversation=True,
            materialize_evaluation=True,
            materialize_report=True,
        )


def test_duplicate_criterion_results_are_rejected() -> None:
    events = [make_evaluation(0, "billing-explanation"), make_evaluation(1, "billing-explanation")]
    with pytest.raises(PersistenceError, match="multiple results"):
        build_projections(
            "run-1",
            events,
            [("billing-explanation", "semantic_goal_completion", True)],
            materialize_conversation=True,
            materialize_evaluation=True,
            materialize_report=True,
        )


def test_disabled_flags_record_omissions() -> None:
    events = [make_event(0), make_evaluation(1, "billing-explanation")]
    projections = build_projections(
        "run-1",
        events,
        [("billing-explanation", "semantic_goal_completion", True)],
        materialize_conversation=False,
        materialize_evaluation=False,
        materialize_report=False,
    )
    assert projections.conversation is None
    assert projections.evaluation is None
    assert projections.report is None


def test_finalize_writes_projections_then_metadata_last(tmp_path: Path) -> None:
    async def exercise() -> None:
        redactor = Redactor(PrivacyConfig())
        backend = FilesystemStorage(
            tmp_path,
            redactor,
            timeout_seconds=5,
            criteria=[
                {
                    "id": "billing-explanation",
                    "evaluator_type": "semantic_goal_completion",
                    "required": True,
                }
            ],
        )
        run = make_run()
        try:
            await backend.create_run(
                run, ConfigArtifact(source_content_hash="hash", effective_config={})
            )
            for event in [make_event(0), make_event(1)]:
                await backend.append_event(run.id, event)
            evaluation = make_evaluation(2, "billing-explanation")
            await backend.append_event(run.id, evaluation)
            finalized = run.model_copy(
                update={
                    "lifecycle_status": "finalized",
                    "outcome": "passed",
                    "termination_reason": "goal_reached",
                    "ended_at": NOW + timedelta(seconds=3),
                }
            )
            await backend.finalize_run(finalized)
            with pytest.raises(PersistenceError):
                await backend.append_event(run.id, make_event(3))
            directory = tmp_path / run.id
            assert (directory / "conversation.json").exists()
            assert (directory / "evaluation.json").exists()
            assert (directory / "report.json").exists()
            metadata = MetadataArtifact.model_validate_json(
                (directory / "metadata.json").read_bytes()
            )
            assert metadata.lifecycle_status == "finalized"
            assert metadata.outcome == "passed"
            assert any(record.status == "omitted" for record in metadata.artifacts) is False
            report = json.loads((directory / "report.json").read_bytes())
            assert report["turn_count"] == 2
            assert report["duration_seconds"] == 2.0
            # Idempotent retry with identical evidence.
            await backend.finalize_run(finalized)
        finally:
            await backend.close()

    asyncio.run(exercise())


def test_final_metadata_redacts_caller_artifacts(tmp_path: Path) -> None:
    async def exercise() -> None:
        backend = FilesystemStorage(
            tmp_path,
            Redactor(PrivacyConfig(), resolved_secrets=["artifact-secret"]),
            timeout_seconds=5,
            criteria=[
                {
                    "id": "billing-explanation",
                    "evaluator_type": "semantic_goal_completion",
                    "required": True,
                }
            ],
        )
        run = make_run()
        try:
            await backend.create_run(
                run, ConfigArtifact(source_content_hash="hash", effective_config={})
            )
            await backend.append_event(run.id, make_evaluation(0, "billing-explanation"))
            finalized = run.model_copy(
                update={
                    "lifecycle_status": "finalized",
                    "outcome": "passed",
                    "termination_reason": "goal_reached",
                    "ended_at": NOW,
                    "artifacts": [
                        ArtifactRecord(
                            path="artifact-secret/path",
                            media_type="text/plain",
                            status="failed",
                            reason="artifact-secret",
                        )
                    ],
                }
            )
            await backend.finalize_run(finalized)
            assert "artifact-secret" not in (tmp_path / run.id / "metadata.json").read_text()
        finally:
            await backend.close()

    asyncio.run(exercise())


def test_finalize_requires_created_run_and_finalized_lifecycle(tmp_path: Path) -> None:
    async def exercise() -> None:
        backend = FilesystemStorage(tmp_path, Redactor(PrivacyConfig()), timeout_seconds=5)
        try:
            with pytest.raises(PersistenceError, match="invalid finalization"):
                await backend.finalize_run(make_run())
        finally:
            await backend.close()

    asyncio.run(exercise())


def test_finalize_failure_leaves_no_finalized_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    original_replace = os.replace
    calls = 0

    def fail_first(source, destination):
        nonlocal calls
        if str(destination).endswith("conversation.json"):
            calls += 1
            if calls == 1:
                raise OSError("replace failed")
        original_replace(source, destination)

    async def exercise() -> None:
        monkeypatch.setattr("os.replace", fail_first)
        backend = FilesystemStorage(
            tmp_path,
            Redactor(PrivacyConfig()),
            timeout_seconds=5,
            criteria=[
                {
                    "id": "billing-explanation",
                    "evaluator_type": "semantic_goal_completion",
                    "required": True,
                }
            ],
        )
        run = make_run()
        try:
            await backend.create_run(
                run, ConfigArtifact(source_content_hash="hash", effective_config={})
            )
            await backend.append_event(run.id, make_event(0))
            await backend.append_event(run.id, make_evaluation(1, "billing-explanation"))
            finalized = run.model_copy(
                update={
                    "lifecycle_status": "finalized",
                    "outcome": "passed",
                    "termination_reason": "goal_reached",
                    "ended_at": NOW + timedelta(seconds=2),
                }
            )
            with pytest.raises(PersistenceError):
                await backend.finalize_run(finalized)
            metadata = MetadataArtifact.model_validate_json(
                (tmp_path / run.id / "metadata.json").read_bytes()
            )
            assert metadata.lifecycle_status == "in_progress"
        finally:
            await backend.close()

    asyncio.run(exercise())

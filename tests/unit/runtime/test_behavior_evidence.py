"""Committed behavioral batches, privacy integrity, and safe partial projections."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from syntheticcx.config.models import Criterion, PrivacyConfig
from syntheticcx.domain.behavior import BehavioralRepetitionFact, BehavioralSignal
from syntheticcx.domain.events import (
    BehavioralObservationRecorded,
    BehavioralSignalRecorded,
    MessageObserved,
    RepeatedInformationObserved,
    RunLifecycleChange,
    committed_behavior_batches,
)
from syntheticcx.domain.models import BehavioralStateTransition, Message, PersonaState
from syntheticcx.errors import PersistenceError
from syntheticcx.privacy import Redactor
from syntheticcx.reporting.console import render_report
from syntheticcx.reporting.markdown import render_markdown
from syntheticcx.reporting.materialize import build_projections
from syntheticcx.runtime.clock import ManualClock
from syntheticcx.runtime.journal import RunJournal

NOW = datetime(2026, 10, 10, tzinfo=UTC)


class MemoryStorage:
    def __init__(self, fail_at=None):
        self.events = []
        self.fail_at = fail_at
        self.attempts = 0

    async def append_event(self, run_id, event):
        self.attempts += 1
        if self.fail_at == len(self.events):
            raise OSError("PRIVATE storage detail")
        self.events.append(event)


def journal(storage=None, redactor=None):
    return RunJournal(
        "run-evidence",
        storage or MemoryStorage(),
        redactor or Redactor(PrivacyConfig()),
        ManualClock(NOW),
    )


async def seed(active, *, count=1, detector_ids=None):
    customer = await active.append(
        MessageObserved(
            message=Message(
                id="customer-1",
                sequence=0,
                actor="customer",
                content="Hello",
                observed_at=NOW,
            )
        )
    )
    assistant = await active.append(
        MessageObserved(
            message=Message(
                id="assistant-1",
                sequence=1,
                actor="assistant",
                content="Welcome",
                observed_at=NOW,
            )
        )
    )
    boundary = BehavioralObservationRecorded(
        observation_id=uuid4(),
        kind="customer_turn_completed",
        evidence_event_ids=[customer.id, assistant.id],
        message_ids=["assistant-1"],
        detector_ids=["custom"] if detector_ids is None else detector_ids,
        state=PersonaState(turn_number=1),
        turn_count=1,
        signal_count=count,
    )
    await active.append(boundary)
    signal = BehavioralSignal(
        detector_id="custom",
        detector_type="custom",
        detector_version="1",
        signal="custom.signal",
        deduplication_key="completed-1",
        observation_id=boundary.observation_id,
        source_event_ids=(assistant.id,),
        source_message_ids=("assistant-1",),
        payload={"explanation": "PRIVATE payload"},
    )
    return boundary, signal


async def complete(active, signal):
    recorded = await active.append(BehavioralSignalRecorded(signal=signal))
    before = PersonaState(turn_number=1)
    return await active.append(
        BehavioralStateTransition(
            observation_id=signal.observation_id,
            signal_event_id=recorded.id,
            signal=signal.signal,
            occurrence=1,
            before=before,
            after=before.model_copy(update={"frustration": 0.2}),
            applied_rule_ids=["custom-rule"],
        )
    )


def projection(events):
    return build_projections(
        "run-evidence",
        list(events),
        [],
        materialize_conversation=True,
        materialize_evaluation=True,
        materialize_report=True,
    )


def test_complete_batch_is_detached_and_distinct_from_formal_evaluation():
    async def exercise():
        active = journal()
        _, signal = await seed(active)
        await complete(active, signal)
        await active.append(
            RunLifecycleChange(
                current="finalized", outcome="passed", termination_reason="max_turns"
            )
        )
        return active

    active = asyncio.run(exercise())
    batches, reason = committed_behavior_batches(active.committed_events)
    assert reason is None and len(batches) == 1
    assert len(batches[0].signals) == len(batches[0].transitions) == 1
    report = projection(active.committed_events).report
    assert report.behavior_completeness == "complete"
    assert report.behavior_observation_count == report.behavior_signal_count == 1
    assert report.repetition_count == 0 and report.findings == []
    assert report.behavior_transitions[0].after.frustration == 0.2
    summary = report.behavior_transitions[0]
    assert summary.detector_id == "custom" and summary.detector_version == "1"
    assert summary.source_event_ids == list(batches[0].signals[0].data.signal.source_event_ids)
    assert summary.source_message_ids == ["assistant-1"]
    assert summary.applied_rule_ids == ["custom-rule"]
    assert "payload" not in summary.model_dump()
    assert "PRIVATE payload" not in report.model_dump_json()
    report.behavior_transitions.clear()
    assert len(projection(active.committed_events).report.behavior_transitions) == 1


@pytest.mark.parametrize("signal_committed", [False, True])
def test_incomplete_batch_projects_safe_prefix_without_claiming_any_effect(signal_committed):
    async def exercise():
        active = journal()
        _, signal = await seed(active)
        if signal_committed:
            await active.append(BehavioralSignalRecorded(signal=signal))
        return active.committed_events

    events = asyncio.run(exercise())
    batches, reason = committed_behavior_batches(events)
    assert batches == () and reason == "final behavioral batch is incomplete"
    views = projection(events)
    assert len(views.conversation.root) == 2
    assert views.report.behavior_completeness == "prefix"
    assert views.report.behavior_signal_count == views.report.behavior_observation_count == 0
    assert views.report.behavior_transitions == []
    assert "behavior:incomplete_prefix" in views.report.omissions


def test_storage_failure_poisoning_preserves_safe_partial_evidence_without_more_writes():
    async def exercise():
        storage = MemoryStorage(fail_at=4)
        active = journal(storage)
        _, signal = await seed(active)
        with pytest.raises(PersistenceError) as caught:
            await complete(active, signal)
        assert "PRIVATE" not in str(caught.value)
        assert not active.available
        attempts = storage.attempts
        for _ in range(3):
            views = projection(active.committed_events)
            assert views.report.behavior_completeness == "prefix"
            assert views.report.behavior_signal_count == 0
        with pytest.raises(PersistenceError):
            await active.append(RunLifecycleChange(current="incomplete"))
        assert storage.attempts == attempts
        assert len(active.committed_events) == 4

    asyncio.run(exercise())


def test_signal_payload_redaction_preserves_all_identities_and_references():
    async def exercise():
        active = journal(redactor=Redactor(PrivacyConfig(), resolved_secrets=["PRIVATE payload"]))
        _, signal = await seed(active)
        await complete(active, signal)
        return active.committed_events, signal

    events, original = asyncio.run(exercise())
    recorded = events[3].data.signal
    assert recorded.payload == {"explanation": "<redacted:secret>"}
    assert recorded.model_dump(exclude={"payload"}) == original.model_dump(exclude={"payload"})
    assert original.payload["explanation"] == "PRIVATE payload"
    report = projection(events).report
    assert "PRIVATE payload" not in report.model_dump_json()
    assert "explanation" not in report.model_dump_json()


def test_valid_redacted_reference_is_still_rejected_if_identity_changes():
    async def exercise():
        active = journal(redactor=Redactor(PrivacyConfig(), resolved_secrets=["completed-1"]))
        _, signal = await seed(active)
        with pytest.raises(PersistenceError):
            await active.append(BehavioralSignalRecorded(signal=signal))
        assert len(active.committed_events) == 3
        assert not active.available

    asyncio.run(exercise())


@pytest.mark.parametrize("mutation", ["forward_event", "unknown_message", "unknown_detector"])
def test_invalid_signal_references_are_never_acknowledged(mutation):
    async def exercise():
        active = journal()
        _, signal = await seed(active)
        change = {
            "forward_event": {"source_event_ids": ("future-event",)},
            "unknown_message": {"source_message_ids": ("future-message",)},
            "unknown_detector": {"detector_id": "other"},
        }[mutation]
        with pytest.raises(PersistenceError):
            await active.append(BehavioralSignalRecorded(signal=signal.model_copy(update=change)))
        assert len(active.committed_events) == 3

    asyncio.run(exercise())


def test_transition_cannot_precede_complete_signal_batch():
    async def exercise():
        active = journal()
        _, signal = await seed(active, count=2)
        with pytest.raises(PersistenceError):
            await complete(active, signal)
        assert len(active.committed_events) == 4

    asyncio.run(exercise())


def test_duplicate_key_and_interrupted_batch_cannot_resume():
    async def exercise():
        active = journal()
        _, signal = await seed(active, count=2)
        await active.append(BehavioralSignalRecorded(signal=signal))
        with pytest.raises(PersistenceError):
            await active.append(BehavioralSignalRecorded(signal=signal))
        another = journal()
        _, signal = await seed(another)
        await another.append(
            RunLifecycleChange(current="incomplete", termination_reason="behavior_error")
        )
        with pytest.raises(PersistenceError):
            await another.append(BehavioralSignalRecorded(signal=signal))
        assert projection(another.committed_events).report.behavior_completeness == "prefix"

    asyncio.run(exercise())


def test_no_match_disabled_boundary_is_preserved():
    async def exercise():
        active = journal()
        await seed(active, count=0, detector_ids=[])
        await active.append(
            RunLifecycleChange(
                current="finalized", outcome="passed", termination_reason="max_turns"
            )
        )
        return active.committed_events

    report = projection(asyncio.run(exercise())).report
    assert report.behavior_observation_count == 1
    assert report.behavior_signal_count == 0
    assert report.behavior_completeness == "complete"


def test_report_rejects_invalid_behavior_links_instead_of_claiming_success():
    async def exercise():
        active = journal()
        _, signal = await seed(active)
        await complete(active, signal)
        return list(active.committed_events)

    events = asyncio.run(exercise())
    event = events[-1]
    events[-1] = event.model_copy(
        update={
            "data": event.data.model_copy(
                update={"signal_event_id": "not-a-committed-signal"},
            )
        }
    )
    with pytest.raises(PersistenceError, match="behavioral evidence is inconsistent"):
        projection(events)


def test_repetition_facts_count_independently_with_detectors_disabled():
    async def exercise():
        active = journal()
        customer = await active.append(
            MessageObserved(
                message=Message(
                    id="supplied",
                    sequence=0,
                    actor="customer",
                    content="Account 123",
                    observed_at=NOW,
                )
            )
        )
        assistant = await active.append(
            MessageObserved(
                message=Message(
                    id="requested",
                    sequence=1,
                    actor="assistant",
                    content="Repeat account?",
                    observed_at=NOW,
                )
            )
        )
        fact = await active.append(
            RepeatedInformationObserved(
                information_id="account",
                original_message_id="supplied",
                request_message_id="requested",
            )
        )
        detached = BehavioralRepetitionFact(
            event_id=fact.id,
            information_id="account",
            original_message_id="supplied",
            request_message_id="requested",
        )
        assert detached.event_id == fact.id
        await active.append(
            BehavioralObservationRecorded(
                observation_id=uuid4(),
                kind="customer_turn_completed",
                evidence_event_ids=[customer.id, assistant.id, fact.id],
                message_ids=["requested"],
                detector_ids=[],
                state=PersonaState(turn_number=1),
                turn_count=1,
                signal_count=0,
            )
        )
        await active.append(
            RunLifecycleChange(
                current="finalized", outcome="passed", termination_reason="max_turns"
            )
        )
        return active.committed_events

    report = projection(asyncio.run(exercise())).report
    assert report.repetition_count == 1 and report.behavior_signal_count == 0
    console = render_report(report)
    markdown = render_markdown(report, [], "en-US")
    assert "not formal evaluation" in console
    assert "separate from formal evaluation" in markdown
    assert markdown.index("Owned behavioral effects") < markdown.index("Formal test conclusions")


@pytest.mark.parametrize(
    "values", [{"signal_count": True}, {"turn_count": True}, {"signal_count": 257}]
)
def test_batch_counts_are_strict_and_bounded(values):
    with pytest.raises(ValidationError):
        BehavioralObservationRecorded.model_validate(
            {
                "observation_id": str(uuid4()),
                "kind": "greeting",
                "evidence_event_ids": ["e1"],
                "message_ids": [],
                "detector_ids": ["d"],
                "state": {},
                "turn_count": 0,
                "signal_count": 1,
                **values,
            }
        )


def test_partial_projection_retains_missing_evaluation_as_omission_not_verdict():
    async def exercise():
        active = journal()
        await seed(active)
        return active.committed_events

    criterion = Criterion(
        id="required-goal",
        evaluator_type="semantic_goal_completion",
        result_kind="boolean",
        evidence_policy="required",
        parameters={"expected_facts": ["Required explanation"]},
    )
    views = build_projections(
        "run-evidence",
        list(asyncio.run(exercise())),
        [criterion],
        allow_partial=True,
        materialize_conversation=True,
        materialize_evaluation=True,
        materialize_report=True,
    )
    assert views.report.outcome is None
    assert views.report.findings == []
    assert views.evaluation.criteria == [criterion] and views.evaluation.results == []
    assert "evaluation:incomplete" in views.report.omissions
    assert views.report.behavior_completeness == "prefix"


def test_signals_cannot_cite_other_signals_from_same_unapplied_batch():
    async def exercise():
        active = journal()
        _, signal = await seed(active, count=2)
        first = await active.append(BehavioralSignalRecorded(signal=signal))
        second = signal.model_copy(
            update={
                "deduplication_key": "another-key",
                "source_event_ids": (first.id,),
            }
        )
        with pytest.raises(PersistenceError):
            await active.append(BehavioralSignalRecorded(signal=second))
        assert len(active.committed_events) == 4

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "change",
    [
        {"occurrence": 7},
        {"before": PersonaState(frustration=0.8, turn_number=1)},
    ],
)
def test_report_rejects_inconsistent_recorded_state_or_counter(change):
    async def exercise():
        active = journal()
        _, signal = await seed(active)
        await complete(active, signal)
        return list(active.committed_events)

    events = asyncio.run(exercise())
    event = events[-1]
    events[-1] = event.model_copy(update={"data": event.data.model_copy(update=change)})
    with pytest.raises(PersistenceError, match="behavioral evidence is inconsistent"):
        projection(events)

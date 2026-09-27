"""Local durability, failure, and async acknowledgement boundaries."""

import asyncio
import json
import os
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from threading import Event

import pytest
import yaml
from pydantic import TypeAdapter

from syntheticcx.adapters.storage.filesystem import FilesystemStorage, canonical_bytes
from syntheticcx.adapters.storage.recovery import Recovery
from syntheticcx.config.loader import load_config, load_config_with_source_hash
from syntheticcx.config.models import PrivacyConfig, ProjectConfig
from syntheticcx.domain.artifacts import ConfigArtifact, MetadataArtifact, Run
from syntheticcx.domain.events import (
    ArtifactRecordedEvent,
    ConversationStarted,
    ConversationStartedEvent,
    EvaluationCompleted,
    MessageObserved,
    RunLifecycleChange,
    RunLifecycleEvent,
    TypedTraceEvent,
)
from syntheticcx.domain.models import ArtifactRecord, EvaluationResult, Message, Report
from syntheticcx.errors import PersistenceError
from syntheticcx.privacy import Redactor
from syntheticcx.runtime.journal import RunJournal

NOW = datetime(2026, 9, 24, tzinfo=UTC)


class Clock:
    def now(self) -> datetime:
        return NOW


def run(identifier: str = "run-1") -> Run:
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


def config() -> ConfigArtifact:
    return ConfigArtifact(source_content_hash="hash", effective_config={"secret_field": "api-key"})


def event(identifier: str = "event-1", sequence: int = 0) -> ConversationStartedEvent:
    return ConversationStartedEvent(
        id=identifier,
        sequence=sequence,
        timestamp=NOW,
        type="conversation_started",
        data=ConversationStarted(greeting_observed=False, grace_seconds=2),
    )


def storage(root: Path, timeout: float = 5) -> FilesystemStorage:
    return FilesystemStorage(
        root, Redactor(PrivacyConfig(), resolved_secrets=["api-key"]), timeout_seconds=timeout
    )


def screenshot_storage(root: Path) -> FilesystemStorage:
    source = load_config_with_source_hash(Path("examples/telecom/scenario.yaml")).config
    config_data = source.model_dump(mode="json")
    config_data["privacy"].update(allow_screenshots=True, confirm_screenshot_target_safe=True)
    config_data["storage"]["store_screenshots"] = True
    return FilesystemStorage(
        root,
        Redactor(PrivacyConfig(), resolved_secrets=["api-key"]),
        timeout_seconds=5,
        project_config=ProjectConfig.model_validate(config_data),
    )


def test_screenshot_writer_is_disabled_without_explicit_policy(tmp_path: Path) -> None:
    async def exercise() -> None:
        backend = storage(tmp_path)
        await backend.create_run(run(), config())
        with pytest.raises(PersistenceError, match="disabled"):
            await backend.write_screenshot(
                "run-1", "artifacts/screenshots/failure.png", b"\x89PNG\r\n\x1a\nfixture"
            )
        assert not (tmp_path / "run-1/artifacts").exists()
        await backend.close()

    asyncio.run(exercise())


def test_screenshot_writer_owns_bounded_atomic_png(tmp_path: Path) -> None:
    async def exercise() -> None:
        backend = screenshot_storage(tmp_path)
        await backend.create_run(run(), config())
        png = b"\x89PNG\r\n\x1a\nfixture-image"
        for path, content in [
            ("../outside.png", png),
            ("artifacts/screenshots/../outside.png", png),
            ("artifacts/screenshots/unsafe.jpg", png),
            ("artifacts/screenshots/failure.png", b"not an image"),
            ("artifacts/screenshots/failure.png", png[:8] + b"0" * (5 * 1024 * 1024)),
        ]:
            with pytest.raises(PersistenceError):
                await backend.write_screenshot("run-1", path, content)
        with pytest.raises(PersistenceError):
            await backend.write_screenshot("another-run", "artifacts/screenshots/failure.png", png)
        record = await backend.write_screenshot("run-1", "artifacts/screenshots/failure.png", png)
        assert record.path == "artifacts/screenshots/failure.png"
        assert record.content_hash == sha256(png).hexdigest()
        assert (tmp_path / "run-1" / record.path).read_bytes() == png
        await backend.close()

    asyncio.run(exercise())


def test_screenshot_writer_rejects_symlink_directory(tmp_path: Path) -> None:
    async def exercise() -> None:
        backend = screenshot_storage(tmp_path)
        await backend.create_run(run(), config())
        outside = tmp_path / "outside"
        outside.mkdir()
        (tmp_path / "run-1/artifacts").symlink_to(outside, target_is_directory=True)
        with pytest.raises(PersistenceError):
            await backend.write_screenshot(
                "run-1", "artifacts/screenshots/failure.png", b"\x89PNG\r\n\x1a\nfixture"
            )
        assert list(outside.iterdir()) == []
        await backend.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("status", ["captured", "omitted", "failed"])
def test_committed_screenshot_artifact_projects_to_metadata_and_report(
    tmp_path: Path, status: str
) -> None:
    async def exercise() -> None:
        backend = storage(tmp_path)
        initial = run()
        await backend.create_run(initial, config())
        artifact = ArtifactRecord(
            path="artifacts/screenshots/failure.png" if status == "captured" else None,
            media_type="image/png",
            content_hash="acknowledged-hash" if status == "captured" else None,
            status=status,
            reason={"omitted": "screenshots_disabled", "failed": "capture_failed"}.get(status),
        )
        await backend.append_event(
            initial.id,
            ArtifactRecordedEvent(
                id="image-record",
                sequence=0,
                timestamp=NOW,
                type="artifact_recorded",
                data=artifact,
            ),
        )
        await backend.append_event(
            initial.id,
            RunLifecycleEvent(
                id="final-lifecycle",
                sequence=1,
                timestamp=NOW,
                type="run_lifecycle_changed",
                data=RunLifecycleChange(
                    previous="in_progress",
                    current="finalized",
                    termination_reason="channel_error",
                    outcome="error",
                ),
            ),
        )
        final = initial.model_copy(
            update={
                "lifecycle_status": "finalized",
                "outcome": "error",
                "termination_reason": "channel_error",
                "ended_at": NOW,
            }
        )
        await backend.finalize_run(final)
        metadata = MetadataArtifact.model_validate_json(
            (tmp_path / "run-1/metadata.json").read_bytes()
        )
        report = Report.model_validate_json((tmp_path / "run-1/report.json").read_bytes())
        assert metadata.artifacts.count(artifact) == 1
        assert report.artifacts.count(artifact) == 1
        if status != "captured":
            omission = f"screenshot:{artifact.reason}"
            assert omission in metadata.metadata.capture_omissions
            assert omission in report.omissions
        await backend.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "stage", ["created", "started", "customer", "assistant", "evaluation", "projection"]
)
def test_recovery_at_each_durable_stage_preserves_evidence_and_inventories_artifacts(
    tmp_path: Path, stage: str
) -> None:
    class DeadOwner:
        def supported(self) -> bool:
            return True

        def process_start_evidence(self, pid: int) -> None:
            return None

    async def prepare() -> list[str]:
        backend = storage(tmp_path)
        initial = run()
        journal = RunJournal(initial.id, backend, Redactor(PrivacyConfig()), Clock())
        await backend.create_run(initial, config())
        acknowledged: list[str] = []
        steps = [
            ConversationStarted(greeting_observed=False, grace_seconds=0),
            MessageObserved(
                message=Message(
                    id="customer-1",
                    sequence=0,
                    actor="customer",
                    content="My bill",
                    observed_at=NOW,
                )
            ),
            MessageObserved(
                message=Message(
                    id="assistant-1",
                    sequence=1,
                    actor="assistant",
                    content="Plan change",
                    observed_at=NOW,
                )
            ),
            EvaluationCompleted(
                result=EvaluationResult(
                    evaluator_id="semantic_goal_completion",
                    criterion_id="billing-explanation",
                    required=True,
                    status="passed",
                    explanation="Plan change explained.",
                    evidence_event_ids=[],
                    evidence_message_ids=["assistant-1"],
                ),
                criterion=load_config(
                    Path("examples/telecom/scenario.yaml")
                ).scenario.success_criteria[0],
            ),
        ]
        count = {
            "created": 0,
            "started": 1,
            "customer": 2,
            "assistant": 3,
            "evaluation": 4,
            "projection": 4,
        }[stage]
        for data in steps[:count]:
            committed = await journal.append(data)
            acknowledged.append(committed.id)
        await backend.close()
        return acknowledged

    acknowledged = asyncio.run(prepare())
    directory = tmp_path / "run-1"
    if stage == "projection":
        (directory / "conversation.json").write_text("[]\n", encoding="utf-8")
    trace = directory / "trace.jsonl"
    complete = trace.read_bytes()
    with trace.open("ab") as output:
        output.write(b'{"unfinished":')
    report = Recovery(tmp_path, DeadOwner(), "test").scan()
    assert report.repaired == ["run-1"]
    assert trace.read_bytes() == complete
    recovered = MetadataArtifact.model_validate_json((directory / "metadata.json").read_bytes())
    assert recovered.lifecycle_status == "incomplete" and recovered.outcome is None
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    for item in events:
        TypeAdapter(TypedTraceEvent).validate_python(item)
    assert [event["id"] for event in events] == acknowledged
    assert [event["sequence"] for event in events] == list(range(len(events)))
    assert not (directory / "report.json").exists()
    inventory = {record.path: record for record in recovered.artifacts if record.path}
    assert inventory.keys() == ({"conversation.json"} if stage == "projection" else set())
    assert sum(record.status == "omitted" for record in recovered.artifacts) == (
        2 if stage == "projection" else 3
    )


def test_create_append_retry_and_detached_committed_events(tmp_path: Path) -> None:
    async def exercise() -> None:
        backend = storage(tmp_path)
        source = config()
        try:
            await backend.create_run(run(), source)
            journal = RunJournal("run-1", backend, Redactor(PrivacyConfig()), Clock())
            first = await journal.append(event().data)
            await backend.append_event("run-1", first)
            second = await journal.append(event().data)
            records = (tmp_path / "run-1/trace.jsonl").read_bytes().splitlines(keepends=True)
            assert records == [canonical_bytes(first), canonical_bytes(second)]
            assert [item.sequence for item in journal.committed_events] == [0, 1]
            first.data.greeting_message_ids.append("external-mutation")
            assert journal.committed_events[0].data.greeting_message_ids == []
            metadata = MetadataArtifact.model_validate_json(
                (tmp_path / "run-1/metadata.json").read_bytes()
            )
            assert metadata.lifecycle_status == "in_progress"
            persisted_config = yaml.safe_load((tmp_path / "run-1/config.yaml").read_text())
            assert persisted_config["effective_config"]["secret_field"] == "<redacted:secret>"
            assert source.effective_config["secret_field"] == "api-key"
        finally:
            await backend.close()
            await backend.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["collision", "gap", "conflict", "wrong-run"])
def test_rejects_invalid_writes_preserving_acknowledged_bytes(tmp_path: Path, failure: str) -> None:
    async def exercise() -> None:
        backend = storage(tmp_path)
        await backend.create_run(run(), config())
        await backend.append_event("run-1", event())
        trace = tmp_path / "run-1/trace.jsonl"
        before = trace.read_bytes()
        with pytest.raises(PersistenceError):
            if failure == "collision":
                other = storage(tmp_path)
                try:
                    await other.create_run(run(), config())
                finally:
                    await other.close()
            elif failure == "gap":
                await backend.append_event("run-1", event("event-2", 2))
            elif failure == "wrong-run":
                await backend.append_event("another", event("event-2", 1))
            else:
                changed = event()
                changed.data.greeting_message_ids.append("different")
                await backend.append_event("run-1", changed)
        assert trace.read_bytes() == before
        await backend.close()

    asyncio.run(exercise())


def test_fsync_failure_does_not_advance_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def exercise() -> None:
        backend = storage(tmp_path)
        await backend.create_run(run(), config())
        journal = RunJournal("run-1", backend, Redactor(PrivacyConfig()), Clock())
        first = await journal.append(event().data)

        def fail(descriptor: int) -> None:
            raise OSError("api-key must not escape")

        monkeypatch.setattr(os, "fsync", fail)
        with pytest.raises(PersistenceError) as caught:
            await journal.append(event().data)
        assert "api-key" not in str(caught.value)
        assert len(journal.committed_events) == 1
        assert (tmp_path / "run-1/trace.jsonl").read_bytes().startswith(canonical_bytes(first))
        with pytest.raises(PersistenceError):
            await journal.append(event().data)
        await backend.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("cancel", [False, True])
def test_unresolved_fsync_never_acknowledges_or_allows_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    release = Event()
    completed = Event()
    original = os.fsync

    async def exercise() -> None:
        backend = storage(tmp_path, timeout=0.2 if not cancel else 5)
        await backend.create_run(run(), config())
        loop = asyncio.get_running_loop()
        started = asyncio.Event()

        def stall(descriptor: int) -> None:
            loop.call_soon_threadsafe(started.set)
            try:
                assert release.wait(5)
                original(descriptor)
            finally:
                completed.set()

        monkeypatch.setattr(os, "fsync", stall)
        journal = RunJournal("run-1", backend, Redactor(PrivacyConfig()), Clock())
        pending = asyncio.create_task(journal.append(event().data))
        try:
            await asyncio.wait_for(started.wait(), 3)
            if cancel:
                pending.cancel()
            with pytest.raises(asyncio.CancelledError if cancel else PersistenceError):
                await asyncio.wait_for(pending, 3)
            assert journal.committed_events == ()
            with pytest.raises(PersistenceError):
                await backend.append_event("run-1", event("later", 1))
            release.set()
            await asyncio.to_thread(completed.wait, 3)
            assert journal.committed_events == ()
        finally:
            release.set()
            await backend.close()

    try:
        asyncio.run(exercise())
    finally:
        release.set()


def test_source_hash_uses_original_bytes(tmp_path: Path) -> None:
    original = Path("examples/telecom/scenario.yaml").read_bytes()
    source = tmp_path / "scenario.yaml"
    source.write_bytes(original.replace(b"\n", b"\r\n"))
    loaded = load_config_with_source_hash(source)
    assert loaded.source_content_hash == sha256(source.read_bytes()).hexdigest()
    assert loaded.source_content_hash != sha256(original).hexdigest()
    assert loaded.config.version == "1"


@pytest.mark.parametrize("identifier", ["../escape", "/absolute", ".", "with/slash"])
def test_invalid_run_id_creates_no_files(tmp_path: Path, identifier: str) -> None:
    async def exercise() -> None:
        backend = storage(tmp_path)
        with pytest.raises(PersistenceError):
            await backend.create_run(run(identifier), config())
        assert list(tmp_path.iterdir()) == []

    asyncio.run(exercise())


def test_concurrent_journal_calls_commit_contiguously(tmp_path: Path) -> None:
    async def exercise() -> None:
        backend = storage(tmp_path)
        await backend.create_run(run(), config())
        journal = RunJournal("run-1", backend, Redactor(PrivacyConfig()), Clock())
        try:
            await asyncio.gather(*(journal.append(event().data) for _ in range(10)))
            records = [
                json.loads(line)
                for line in (tmp_path / "run-1/trace.jsonl").read_text().splitlines()
            ]
            assert [record["sequence"] for record in records] == list(range(10))
            assert len({record["id"] for record in records}) == 10
        finally:
            await backend.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["short-write", "write-exception", "flush"])
def test_partial_append_never_advances_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    original_open = Path.open

    class FaultyStream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def write(self, content):
            if failure == "flush":
                return self.stream.write(content)
            count = self.stream.write(content[: len(content) // 2])
            self.stream.flush()
            if failure == "write-exception":
                raise OSError("api-key injected failure")
            return count

        def flush(self):
            raise OSError("api-key injected flush failure")

        def fileno(self):
            return self.stream.fileno()

    def faulty_open(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        return FaultyStream(stream) if path.name == "trace.jsonl" and mode == "ab" else stream

    async def exercise() -> None:
        backend = storage(tmp_path)
        await backend.create_run(run(), config())
        journal = RunJournal("run-1", backend, Redactor(PrivacyConfig()), Clock())
        await journal.append(event().data)
        trace = tmp_path / "run-1/trace.jsonl"
        before = trace.read_bytes()
        monkeypatch.setattr(Path, "open", faulty_open)
        with pytest.raises(PersistenceError) as caught:
            await journal.append(event().data)
        assert "api-key" not in str(caught.value)
        after = trace.read_bytes()
        assert after.startswith(before)
        assert len(after) > len(before)
        if failure != "flush":
            assert not after.endswith(b"\n")
        assert len(journal.committed_events) == 1
        with pytest.raises(PersistenceError):
            await backend.append_event("run-1", event("later", 2))
        assert trace.read_bytes() == after
        await backend.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("boundary", range(1, 8))
def test_initial_fsync_failures_leave_only_in_progress_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: int
) -> None:
    original = os.fsync
    calls = 0

    def fail_at_boundary(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == boundary:
            raise OSError("api-key injected failure")
        original(descriptor)

    monkeypatch.setattr(os, "fsync", fail_at_boundary)

    async def exercise() -> None:
        backend = storage(tmp_path)
        with pytest.raises(PersistenceError) as caught:
            await backend.create_run(run(), config())
        assert "api-key" not in str(caught.value)
        metadata = tmp_path / "run-1/metadata.json"
        if metadata.exists():
            assert json.loads(metadata.read_text())["lifecycle_status"] == "in_progress"
        with pytest.raises(PersistenceError):
            await backend.append_event("run-1", event())
        for path in (tmp_path / "run-1").iterdir():
            assert b"api-key" not in path.read_bytes()
        await backend.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("boundary", [1, 2])
def test_initial_replace_failure_does_not_publish_partial_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: int
) -> None:
    original = os.replace
    calls = 0

    def fail_at_boundary(source, destination):
        nonlocal calls
        calls += 1
        if calls == boundary:
            raise OSError("replace failed")
        original(source, destination)

    monkeypatch.setattr(os, "replace", fail_at_boundary)

    async def exercise() -> None:
        backend = storage(tmp_path)
        with pytest.raises(PersistenceError):
            await backend.create_run(run(), config())
        files = {path.name for path in (tmp_path / "run-1").iterdir()}
        assert files == (set() if boundary == 1 else {"metadata.json"})
        await backend.close()

    asyncio.run(exercise())


def test_acknowledgement_waits_for_successful_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = Event()
    original = os.fsync

    async def exercise() -> None:
        backend = storage(tmp_path)
        await backend.create_run(run(), config())
        loop = asyncio.get_running_loop()
        started = asyncio.Event()

        def stall(descriptor: int) -> None:
            loop.call_soon_threadsafe(started.set)
            assert release.wait(5)
            original(descriptor)

        monkeypatch.setattr(os, "fsync", stall)
        journal = RunJournal("run-1", backend, Redactor(PrivacyConfig()), Clock())
        pending = asyncio.create_task(journal.append(event().data))
        try:
            await asyncio.wait_for(started.wait(), 3)
            assert not pending.done()
            assert journal.committed_events == ()
            release.set()
            acknowledged = await asyncio.wait_for(pending, 3)
            assert journal.committed_events == (acknowledged,)
        finally:
            release.set()
            await backend.close()

    asyncio.run(exercise())

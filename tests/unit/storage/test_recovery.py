"""Conservative recovery: dead-owner repair and untouched ambiguous runs."""

import json
import os
from pathlib import Path

import pytest

from syntheticcx.adapters.storage.recovery import Recovery
from syntheticcx.domain.artifacts import MetadataArtifact

NOW = "2026-09-24T00:00:00+00:00"


def metadata_document(
    *,
    lifecycle_status: str = "in_progress",
    hostname: str = "test-host",
    pid: int = 1,
    process_start_evidence: str = "start-marker",
) -> dict:
    return {
        "id": "run-1",
        "scenario_id": "billing",
        "persona_id": "standard",
        "started_at": NOW,
        "lifecycle_status": lifecycle_status,
        "metadata": {
            "configuration_version": "1",
            "framework_version": "0.1.0",
            "persona_model": {"provider": "openai", "model": "reference"},
            "evaluation_model": {"provider": "openai", "model": "reference"},
            "usage": {"unavailable_reason": "no calls"},
            "runtime_platform": "test",
            "owner": {
                "owner_uuid": "owner",
                "hostname": hostname,
                "pid": pid,
                "process_start_evidence": process_start_evidence,
            },
        },
    }


class Probe:
    def __init__(self, *, supported: bool = True, evidence: str | None = None) -> None:
        self._supported = supported
        self._evidence = evidence

    def supported(self) -> bool:
        return self._supported

    def process_start_evidence(self, pid: int) -> str | None:
        return self._evidence


def write_run(
    tmp_path: Path,
    document: dict,
    trace: bytes = b"",
) -> Path:
    directory = tmp_path / "runs" / document["id"]
    directory.mkdir(parents=True)
    (directory / "metadata.json").write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    )
    (directory / "trace.jsonl").write_bytes(trace)
    return tmp_path / "runs"


def test_confirmed_dead_owner_is_marked_incomplete_and_partial_tail_truncated(
    tmp_path: Path,
) -> None:
    document = metadata_document()
    full = b'{"id":"event-0","sequence":0}\n'
    partial = b'{"id":"event-1","sequ'
    root = write_run(tmp_path, document, full + partial)
    report = Recovery(root, Probe(supported=True, evidence=None), "test-host").scan()
    directory = root / "run-1"
    assert report.repaired == ["run-1"]
    assert report.repaired_count == 1
    assert (directory / "trace.jsonl").read_bytes() == full
    recovered = MetadataArtifact.model_validate_json((directory / "metadata.json").read_bytes())
    assert recovered.lifecycle_status == "incomplete"
    assert recovered.outcome is None


def test_live_owner_reused_pid_and_unsupported_platform_are_untouched(
    tmp_path: Path,
) -> None:
    cases = [
        Probe(supported=True, evidence="different-marker"),  # reused PID
        Probe(supported=True, evidence="start-marker"),  # same live owner
        Probe(supported=False),  # e.g. macOS without /proc
    ]
    for index, probe in enumerate(cases):
        document = metadata_document()
        document["id"] = f"run-{index}"
        case_root = tmp_path / f"case-{index}" / "runs"
        write_run(case_root.parent, document)
        before = (case_root / document["id"] / "metadata.json").read_bytes()
        report = Recovery(case_root, probe, "test-host").scan()
        assert report.repaired == []
        assert report.warned == [document["id"]]
        assert (case_root / document["id"] / "metadata.json").read_bytes() == before


def test_foreign_host_finalized_malformed_and_missing_metadata_are_untouched(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    (root / "foreign").mkdir()
    (root / "foreign/metadata.json").write_text(
        json.dumps(metadata_document(hostname="other-host"), sort_keys=True) + "\n"
    )
    (root / "finalized").mkdir()
    finalized = metadata_document(lifecycle_status="finalized")
    finalized.update(
        {
            "ended_at": NOW,
            "outcome": "passed",
            "termination_reason": "goal_reached",
        }
    )
    (root / "finalized/metadata.json").write_text(json.dumps(finalized, sort_keys=True) + "\n")
    (root / "malformed").mkdir()
    (root / "malformed/metadata.json").write_text("{not-json")
    (root / "empty").mkdir()
    directory_states = {
        name: (root / name / "metadata.json").read_bytes()
        for name in ("foreign", "finalized", "malformed")
        if (root / name / "metadata.json").exists()
    }
    report = Recovery(root, Probe(supported=True, evidence=None), "test-host").scan()
    assert report.repaired == []
    assert "empty" in report.warned
    assert "malformed" in report.warned
    for name, content in directory_states.items():
        assert (root / name / "metadata.json").read_bytes() == content
    # A finalized run with a result is not warned for recovery reasons; it is
    # simply not a candidate. Only warn-worthy states appear in warned.
    assert "finalized" not in report.warned
    assert "foreign" in report.warned


def test_complete_invalid_line_is_not_repaired(tmp_path: Path) -> None:
    document = metadata_document()
    good = b'{"id":"event-0","sequence":0}\n'
    bad = b'{"id": invalid json}\n'
    root = write_run(tmp_path, document, good + bad)
    report = Recovery(root, Probe(supported=True, evidence=None), "test-host").scan()
    directory = root / "run-1"
    assert report.repaired == []
    assert (directory / "trace.jsonl").read_bytes() == good + bad
    assert (
        MetadataArtifact.model_validate_json(
            (directory / "metadata.json").read_bytes()
        ).lifecycle_status
        == "in_progress"
    )


def test_valid_but_unterminated_jsonl_record_is_discarded(tmp_path: Path) -> None:
    root = write_run(tmp_path, metadata_document(), b'{"id":"event-0"}')
    report = Recovery(root, Probe(supported=True, evidence=None), "test-host").scan()
    assert report.repaired == ["run-1"]
    assert (root / "run-1/trace.jsonl").read_bytes() == b""


def test_missing_trace_leaves_in_progress_untouched(tmp_path: Path) -> None:
    root = write_run(tmp_path, metadata_document())
    (root / "run-1/trace.jsonl").unlink()
    before = (root / "run-1/metadata.json").read_bytes()
    report = Recovery(root, Probe(supported=True, evidence=None), "test-host").scan()
    assert report.repaired == []
    assert (root / "run-1/metadata.json").read_bytes() == before


def test_interruption_during_metadata_rewrite_leaves_partial_state_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = metadata_document()
    root = write_run(tmp_path, document)
    original_replace = os.replace
    calls = 0

    def fail_first(source, destination):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("replace failed")
        original_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_first)
    report = Recovery(root, Probe(supported=True, evidence=None), "test-host").scan()
    directory = root / "run-1"
    assert report.repaired == []
    assert (
        MetadataArtifact.model_validate_json(
            (directory / "metadata.json").read_bytes()
        ).lifecycle_status
        == "in_progress"
    )
    # A later scan with working writes completes the repair.
    monkeypatch.setattr(os, "replace", original_replace)
    report = Recovery(root, Probe(supported=True, evidence=None), "test-host").scan()
    assert report.repaired == ["run-1"]
    assert not (directory / "metadata.json.recovery.tmp").exists()


def test_warnings_and_reports_are_value_independent(tmp_path: Path) -> None:
    root = write_run(tmp_path, metadata_document(hostname="other-host"))
    report = Recovery(root, Probe(supported=True, evidence=None), "test-host").scan()
    rendered = json.dumps(report.__dict__)
    assert "start-marker" not in rendered
    assert "other-host" not in rendered

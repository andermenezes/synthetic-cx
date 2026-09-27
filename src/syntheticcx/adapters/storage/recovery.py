"""Conservative local startup recovery for in-progress runs.

Only a locally owned run whose recorded process is confirmed dead is marked
incomplete. Active, ambiguous, malformed, or unsupported ownership evidence is
left byte-for-byte untouched with a sanitized warning. Linux-local process
identity is the supported guarantee; elsewhere uncertain status is ambiguous.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from syntheticcx.domain.artifacts import MetadataArtifact
from syntheticcx.domain.models import ArtifactRecord


class LivenessProbe(Protocol):
    """Injected process inspection so recovery stays deterministic in tests."""

    def supported(self) -> bool: ...

    def process_start_evidence(self, pid: int) -> str | None:
        """Return identity evidence for a live PID, or None when it is not running."""


@dataclass(frozen=True)
class RecoveryReport:
    """Value-independent outcomes for console or log reporting."""

    repaired: list[str]
    warned: list[str]
    repaired_count: int = 0
    warned_count: int = 0


class Recovery:
    """Scan one local run root and repair only confirmed-dead owners."""

    def __init__(self, root: Path, probe: LivenessProbe, hostname: str) -> None:
        self._root = root
        self._probe = probe
        self._hostname = hostname

    def scan(self) -> RecoveryReport:
        """Return per-run outcomes without exposing values from any artifact."""
        repaired: list[str] = []
        warned: list[str] = []
        if not self._root.is_dir():
            return RecoveryReport(repaired, warned)
        for directory in sorted(self._root.iterdir()):
            if not directory.is_dir():
                continue
            outcome = self._recover(directory)
            if outcome == "repaired":
                repaired.append(directory.name)
            elif outcome == "warned":
                warned.append(directory.name)
        return RecoveryReport(
            repaired=sorted(repaired),
            warned=sorted(warned),
            repaired_count=len(repaired),
            warned_count=len(warned),
        )

    def _recover(self, directory: Path) -> str | None:
        metadata_path = directory / "metadata.json"
        try:
            metadata = MetadataArtifact.model_validate_json(metadata_path.read_bytes())
        except (OSError, ValueError):
            return "warned"
        if metadata.lifecycle_status != "in_progress":
            return None
        owner = metadata.metadata.owner
        if owner.hostname != self._hostname:
            return "warned"
        if not self._probe.supported():
            return "warned"
        evidence = self._probe.process_start_evidence(owner.pid)
        if evidence is not None:
            # A live PID (including a reused one) owns this run: never touch it.
            return "warned"
        trace_path = directory / "trace.jsonl"
        try:
            content = trace_path.read_bytes()
        except OSError:
            return "warned"
        if content and not content.endswith(b"\n"):
            # Only a partial trailing record may be discarded; earlier complete
            # records are retained. A complete invalid line is not repaired here.
            trailing = content.rfind(b"\n")
            complete = b"" if trailing == -1 else content[: trailing + 1]
            if not self._validate_complete_lines(complete):
                return "warned"
            try:
                self._atomic_truncate(trace_path, len(complete))
            except OSError:
                return "warned"
        elif content and not self._validate_complete_lines(content):
            return "warned"
        artifacts = list(metadata.artifacts)
        for name in ("conversation.json", "evaluation.json", "report.json"):
            if any(record.path == name for record in artifacts):
                continue
            path = directory / name
            if path.is_file():
                try:
                    content = path.read_bytes()
                    json.loads(content)
                except (OSError, ValueError):
                    return "warned"
                artifacts.append(
                    ArtifactRecord(
                        path=name,
                        media_type="application/json",
                        content_hash=sha256(content).hexdigest(),
                        status="captured",
                    )
                )
            else:
                artifacts.append(
                    ArtifactRecord(
                        media_type="application/json",
                        status="omitted",
                        reason=f"{name}:not_materialized",
                    )
                )
        updated = metadata.model_copy(
            update={"lifecycle_status": "incomplete", "artifacts": artifacts},
        )
        try:
            self._atomic_write(
                metadata_path,
                (
                    json.dumps(
                        updated.model_dump(mode="json"),
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8"),
            )
        except OSError:
            return "warned"
        return "repaired"

    @staticmethod
    def _validate_complete_lines(content: bytes) -> bool:
        """Every newline-terminated record must parse as JSON; no event schema
        enforcement here so recovery cannot reject future valid envelopes."""
        for line in content.splitlines():
            try:
                json.loads(line)
            except ValueError:
                return False
        return True

    @staticmethod
    def _atomic_truncate(path: Path, size: int) -> None:
        with path.open("r+b") as stream:
            stream.truncate(size)
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        temporary = path.with_name(path.name + ".recovery.tmp")
        try:
            with temporary.open("wb") as stream:
                if stream.write(content) != len(content):
                    raise OSError("partial artifact write")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)

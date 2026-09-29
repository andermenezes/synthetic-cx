"""Publish a verified guide and complete its scenario without overwriting user work."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from syntheticcx.config.loader import load_config, load_discovery_draft
from syntheticcx.config.models import ChannelGuide, ReferencedProjectConfig
from syntheticcx.errors import ConfigurationError
from syntheticcx.language import Language


def _write_new(path: Path, data: bytes) -> None:
    """Write a sibling file and publish without replacing any existing destination."""
    temporary: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Fails if the destination already exists.
    except OSError:
        raise ConfigurationError("publish discovery", "could not create output file") from None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _replace(path: Path, data: bytes) -> None:
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError:
        raise ConfigurationError("publish discovery", "could not update scenario") from None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def publish_guide(
    source: Path,
    output: Path,
    guide: ChannelGuide,
    *,
    in_place: bool,
    language: Language | None = None,
) -> Path:
    """Save verified controls, then a validated scenario copy or backed-up edit."""
    load_discovery_draft(source)
    destination = source if in_place else source.with_name(f"{source.stem}.completed.yaml")
    backup = source.with_name(f"{source.name}.bak")
    identities = {source.resolve(), destination.resolve(), backup.resolve()}
    if output.resolve() in identities:
        raise ConfigurationError("publish discovery", "guide output conflicts with scenario files")
    if output.exists() or (not in_place and destination.exists()) or (in_place and backup.exists()):
        raise ConfigurationError("publish discovery", "output or backup already exists")
    reference = os.path.relpath(output.resolve(), start=destination.parent.resolve())
    try:
        original = source.read_bytes()
        document: Any = yaml.safe_load(original)
        if language is not None:
            document["persona"]["communication"]["language"] = language
        document["channel_guide"] = reference
        ReferencedProjectConfig.model_validate(document)
        completed = yaml.safe_dump(document, sort_keys=False, allow_unicode=True).encode("utf-8")
    except (OSError, UnicodeError, yaml.YAMLError, TypeError, ValidationError):
        raise ConfigurationError("publish discovery", "scenario could not be completed") from None
    serialized_guide = yaml.safe_dump(
        guide.model_dump(mode="json", exclude_none=True), sort_keys=False, allow_unicode=True
    ).encode("utf-8")
    guide_written = False
    backup_written = False
    scenario_written = False
    try:
        _write_new(output, serialized_guide)
        guide_written = True
        if in_place:
            _write_new(backup, original)
            backup_written = True
            _replace(source, completed)
        else:
            _write_new(destination, completed)
        scenario_written = True
        # A file path is not a valid guide unless the ordinary loader accepts it.
        load_config(destination)
    except Exception:
        if in_place and scenario_written:
            _replace(source, original)
        elif not in_place and scenario_written:
            destination.unlink(missing_ok=True)
        if backup_written:
            backup.unlink(missing_ok=True)
        if guide_written:
            output.unlink(missing_ok=True)
        raise
    return destination

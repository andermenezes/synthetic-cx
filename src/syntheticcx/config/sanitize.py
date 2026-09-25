"""Serialization of the effective configuration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from syntheticcx.config.models import ProjectConfig
from syntheticcx.privacy.redaction import Redactor


def sanitized_effective_config(
    config: ProjectConfig, *, resolved_secrets: Sequence[str] = ()
) -> dict[str, Any]:
    """Return a detached redacted snapshot, never a runtime settings dump.

    Callers preparing configuration supply all resolved provider secrets. This
    snapshot is evidence, not a configuration intended for replay after redaction.
    """
    redactor = Redactor(config.privacy, resolved_secrets=resolved_secrets)
    return cast(dict[str, Any], redactor.redact_value(config.model_dump(mode="json")))

"""Serialization of the effective configuration."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, cast

from syntheticcx.config.models import ProjectConfig


def sanitized_effective_config(
    config: ProjectConfig, *, resolved_secrets: Sequence[str] = ()
) -> dict[str, Any]:
    """Return a detached redacted snapshot, never a runtime settings dump.

    Callers preparing configuration supply all resolved provider secrets. This
    snapshot is evidence, not a configuration intended for replay after redaction.
    """
    keys = set(config.privacy.redaction_keys)
    patterns = [re.compile(pattern) for pattern in config.privacy.synthetic_id_patterns]
    secrets = sorted({value for value in resolved_secrets if value}, key=len, reverse=True)

    def text(value: str) -> str:
        for secret in secrets:
            value = value.replace(secret, "<redacted:secret>")
        for pattern in patterns:
            value = pattern.sub("<redacted:synthetic_id>", value)
        return value

    def redact(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                text(key): f"<redacted:{text(key)}>" if key in keys else redact(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [redact(item) for item in value]
        if isinstance(value, str):
            return text(value)
        return value

    return cast(dict[str, Any], redact(config.model_dump(mode="json")))

"""Explicit launcher hints, distinct from model-proposed browser actions."""

import re

from syntheticcx.errors import ConfigurationError


def launcher_id(chat_hint: str | None) -> str | None:
    """Reserve id= syntax for an exact, bounded DOM ID; other hints remain text."""
    if chat_hint is None or not chat_hint.startswith("id="):
        return None
    value = chat_hint[3:]
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_:-]{0,127}", value):
        raise ConfigurationError(
            "discover hint", "id hint must contain a valid 1 to 128 character DOM ID"
        )
    return value

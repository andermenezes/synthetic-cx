"""Serialization of the effective configuration."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, cast

from pydantic import JsonValue, TypeAdapter, ValidationError

from syntheticcx.config.models import AnyChannelGuide, PrivacyConfig, ProjectConfig
from syntheticcx.errors import ConfigurationError
from syntheticcx.privacy.redaction import Redactor


def _decoded_css_strings(value: Any) -> Any:
    """Screen reversible locator escaping without changing executable selectors."""
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            if match[1] is None:
                return match[2]
            number = int(match[1], 16)
            return chr(number) if 0 < number <= 0x10FFFF else "\ufffd"

        return re.sub(r"\\(?:([0-9a-fA-F]{1,6})[ \t\r\n\f]?|([^\r\n\f]))", replace, value)
    if isinstance(value, dict):
        return {key: _decoded_css_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decoded_css_strings(item) for item in value]
    return value


def sanitized_channel_guide(
    guide: AnyChannelGuide,
    privacy: PrivacyConfig,
    *,
    resolved_secrets: Sequence[str] = (),
    sensitive_context: JsonValue = None,
) -> dict[str, Any]:
    """Allow only validated contract fields; reject rather than rewrite selectors.

    A redacted guide would no longer describe the replayed controls. Sensitive
    contracts therefore fail publication with an owned, content-free diagnostic.
    """
    try:
        validated = TypeAdapter(AnyChannelGuide).validate_python(guide.model_dump(mode="json"))
    except ValidationError:
        raise ConfigurationError("publish discovery", "invalid channel guide contract") from None
    document = validated.model_dump(mode="json", exclude_none=True)
    redactor = Redactor(
        privacy, resolved_secrets=resolved_secrets, sensitive_context=sensitive_context
    )
    decoded = _decoded_css_strings(document)
    if redactor.redact_value(document) != document or redactor.redact_value(decoded) != decoded:
        raise ConfigurationError("publish discovery", "channel guide contains sensitive data")
    return cast(dict[str, Any], document)


def sanitized_effective_config(
    config: ProjectConfig, *, resolved_secrets: Sequence[str] = ()
) -> dict[str, Any]:
    """Return a detached redacted snapshot, never a runtime settings dump.

    Callers preparing configuration supply all resolved provider secrets. This
    snapshot is evidence, not a configuration intended for replay after redaction.
    """
    redactor = Redactor(
        config.privacy,
        resolved_secrets=resolved_secrets,
        sensitive_context={
            "persona": config.persona.model_dump(mode="json"),
            "starting_context": config.scenario.starting_context,
            "knowledge": config.scenario.knowledge,
        },
    )
    return cast(dict[str, Any], redactor.redact_value(config.model_dump(mode="json")))

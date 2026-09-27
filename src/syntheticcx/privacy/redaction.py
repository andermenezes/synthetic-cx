"""Per-run, detached redaction of text, JSON values, and owned models.

Field-name rules apply to structured fields. Free-text values require an exact
secret or configured pattern. Capture permissions are enforced by callers.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import TypeVar, cast

from pydantic import BaseModel, JsonValue

from syntheticcx.config.models import Criterion, PrivacyConfig, ScenarioConfig
from syntheticcx.errors import ConfigurationError

ModelT = TypeVar("ModelT", bound=BaseModel)
_SECRET = "<redacted:secret>"
_PATTERN = "<redacted:synthetic_id>"
_FIELD = "<redacted:field>"


class Redactor:
    """Snapshot privacy rules and explicitly supplied credentials for one run.

    Reserved placeholders are atomic output tokens, including when supplied in
    input. Matches spanning a token still redact the complete sensitive span.
    Secrets colliding with reserved output tokens are rejected as incompatible
    with stable placeholder output. Instances must never be logged or persisted.
    """

    def __init__(self, privacy: PrivacyConfig, *, resolved_secrets: Sequence[str] = ()) -> None:
        self._keys = frozenset(privacy.redaction_keys)
        secrets = sorted({secret for secret in resolved_secrets if secret}, key=len, reverse=True)
        self._secrets = tuple(secrets)
        self._patterns = tuple(re.compile(pattern) for pattern in privacy.synthetic_id_patterns)
        self._labels = {
            key: (
                f"<redacted:{key}>"
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
                and not any(secret in key for secret in secrets)
                and not any(pattern.search(key) for pattern in self._patterns)
                else _FIELD
            )
            for key in self._keys
        }
        tokens = {_SECRET, _PATTERN, _FIELD, *self._labels.values()}
        if any(secret in token for secret in secrets for token in tokens):
            raise ConfigurationError(
                "prepare redaction", "a resolved secret conflicts with reserved redaction tokens"
            )
        self._tokens = re.compile(
            "(" + "|".join(re.escape(token) for token in sorted(tokens)) + ")"
        )

    @property
    def has_field_redaction(self) -> bool:
        """Whether context-aware redaction needs the originating scenario."""
        return bool(self._keys)

    def redact_criterion(self, criterion: Criterion, scenario: ScenarioConfig) -> Criterion:
        """Retain a valid criterion snapshot without known sensitive context values."""
        if criterion.evaluator_type != "semantic_goal_completion":
            return self.redact_model(criterion)
        context = {"starting_context": scenario.starting_context, "knowledge": scenario.knowledge}
        facts = [
            self.redact_text_with_field_values(fact, context)
            for fact in criterion.parameters["expected_facts"]
        ]
        safe = criterion.model_copy(
            update={"parameters": {**criterion.parameters, "expected_facts": facts}}, deep=True
        )
        return cast(Criterion, self.redact_model(safe))

    def redact_text(self, text: str) -> str:
        """Redact to a stable result using whole-string regex context.

        Replacement can expose new lookaround/boundary matches, so continue to
        a fixed point. Each changing pass removes unprotected content or merges
        tokens; zero-width matches are ignored because they contain no data.
        """
        while True:
            result = self._redact_segment(text)
            if result == text:
                return result
            text = result

    def redact_text_with_field_values(self, text: str, values: JsonValue) -> str:
        """Redact known sensitive structured values when they recur in free text.

        A field name alone cannot identify arbitrary free-text content. Callers
        crossing an external boundary can supply related structured context so
        values under configured redaction keys are replaced before normal exact
        secret and pattern redaction runs.
        """
        replacements = self._field_value_replacements(values)
        for value, replacement in sorted(
            replacements.items(), key=lambda item: len(item[0]), reverse=True
        ):
            text = text.replace(value, replacement)
        return self.redact_text(text)

    def redact_prompt_text(self, text: str, values: JsonValue) -> str:
        """Redact raw and JSON-escaped values in nested, text-encoded prompts."""
        replacements = {
            **self._field_value_replacements(values),
            **{secret: _SECRET for secret in self._secrets},
        }
        encoded: dict[str, str] = {}
        for value, replacement in replacements.items():
            encoded[value] = replacement
            for ascii_mode in (False, True):
                escaped = value
                # TinyTroupe may embed a JSON document in another JSON message.
                for _ in range(3):
                    escaped = json.dumps(escaped, ensure_ascii=ascii_mode)[1:-1]
                    encoded[escaped] = replacement
        for value, replacement in sorted(
            encoded.items(), key=lambda item: len(item[0]), reverse=True
        ):
            text = text.replace(value, replacement)
        return self.redact_text(text)

    def _field_value_replacements(self, value: JsonValue) -> dict[str, str]:
        replacements: dict[str, str] = {}
        if isinstance(value, dict):
            for key, item in value.items():
                if key in self._keys and isinstance(item, str) and item:
                    replacements[item] = self._labels[key]
                else:
                    replacements.update(self._field_value_replacements(item))
        elif isinstance(value, list):
            for item in value:
                replacements.update(self._field_value_replacements(item))
        return replacements

    def _redact_segment(self, text: str) -> str:
        # Collect all spans before replacing, so regexes never consume newly
        # generated placeholders and overlapping matches cannot expose suffixes.
        spans: list[tuple[int, int, bool]] = []
        tokens = [(match.start(), match.end()) for match in self._tokens.finditer(text)]
        for secret in self._secrets:
            start = 0
            while (position := text.find(secret, start)) != -1:
                spans.append((position, position + len(secret), True))
                start = position + 1
        for pattern in self._patterns:
            position = 0
            while position <= len(text):
                match = pattern.search(text, position)
                if match is None:
                    break
                if match.start() != match.end():
                    spans.append((match.start(), match.end(), False))
                position = match.start() + 1
        expanded: list[tuple[int, int, bool]] = []
        for start, end, is_secret in spans:
            if any(left <= start and end <= right for left, right in tokens):
                continue
            for left, right in tokens:
                if start < right and left < end:
                    start, end = min(start, left), max(end, right)
            expanded.append((start, end, is_secret))
        merged: list[tuple[int, int, bool]] = []
        for start, end, is_secret in sorted(expanded):
            if merged and start < merged[-1][1]:
                previous_start, previous_end, previous_secret = merged[-1]
                merged[-1] = (previous_start, max(end, previous_end), is_secret or previous_secret)
            else:
                merged.append((start, end, is_secret))
        output: list[str] = []
        cursor = 0
        for start, end, is_secret in merged:
            output.extend((text[cursor:start], _SECRET if is_secret else _PATTERN))
            cursor = end
        output.append(text[cursor:])
        return "".join(output)

    def redact_value(self, value: JsonValue) -> JsonValue:
        """Copy recursive JSON data; reject redacted-key collisions safely."""
        if isinstance(value, dict):
            result: dict[str, JsonValue] = {}
            for key, item in value.items():
                safe_key = self.redact_text(key)
                if safe_key in result:
                    raise ConfigurationError(
                        "sanitize evidence", "redaction would produce duplicate mapping keys"
                    )
                result[safe_key] = (
                    self._labels[key] if key in self._keys else self.redact_value(item)
                )
            return result
        if isinstance(value, list):
            return [self.redact_value(item) for item in value]
        if isinstance(value, str):
            return self.redact_text(value)
        return value

    def redact_model(self, model: ModelT) -> ModelT:
        """Rebuild a sanitized owned model, failing closed if its shape is invalid.

        Validation/serialization errors are deliberately replaced with a fixed
        message: Pydantic diagnostics may contain sensitive values or field names.
        """
        try:
            value = self.redact_value(model.model_dump(mode="json"))
            return cast(ModelT, type(model).model_validate(value))
        except (ValueError, TypeError, ConfigurationError):
            raise ConfigurationError(
                "sanitize evidence", "sanitized content does not satisfy its model contract"
            ) from None

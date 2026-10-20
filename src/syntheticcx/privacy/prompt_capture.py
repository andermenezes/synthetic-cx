"""Explicit, context-aware prompt capture for owned model evidence."""

from __future__ import annotations

import json
from typing import Any

from pydantic import JsonValue

from syntheticcx.config.models import JsonObject, PersonaConfig, ScenarioConfig
from syntheticcx.privacy.redaction import Redactor


class PromptCapture:
    """Serialize prompts only after opt-in and redact before returning evidence."""

    def __init__(
        self, redactor: Redactor, scenario: ScenarioConfig, *, persona: PersonaConfig | None = None
    ) -> None:
        self._redactor = redactor
        self._context: JsonObject = {
            "starting_context": scenario.starting_context,
            "knowledge": scenario.knowledge,
        }
        if persona is not None:
            self._context["persona"] = persona.model_dump(mode="json")

    def sanitize(self, messages: Any) -> str:
        def redact(value: JsonValue) -> JsonValue:
            if isinstance(value, str):
                return self._redactor.redact_prompt_text(value, self._context)
            if isinstance(value, list):
                return [redact(item) for item in value]
            if isinstance(value, dict):
                return {key: redact(item) for key, item in value.items()}
            return value

        # Serialize only JSON-compatible prompt shapes; do not stringify arbitrary
        # objects that might disclose unreviewed attributes or credentials.
        values = json.loads(json.dumps(messages, ensure_ascii=False))
        safe = self._redactor.redact_value(redact(values))
        return json.dumps(safe, ensure_ascii=False)

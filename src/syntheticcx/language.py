"""Supported customer-facing languages for verification and persona messages."""

from __future__ import annotations

from typing import Literal

from syntheticcx.errors import ConfigurationError

Language = Literal["en-US", "pt-BR"]

_FIRST_PROBES: dict[Language, str] = {
    "en-US": "Synthetic test: please reply with a greeting.",
    "pt-BR": "Teste sintético: por favor, responda com uma saudação.",
}
_SECOND_PROBES: dict[Language, str] = {
    "en-US": "Synthetic test: please confirm the chat is responsive.",
    "pt-BR": "Teste sintético: por favor, confirme que o chat está respondendo.",
}
_PERSONA_INSTRUCTIONS: dict[Language, str] = {
    "en-US": "Use English (United States) for every customer TALK message.",
    "pt-BR": "Use Brazilian Portuguese for every customer TALK message.",
}


def require_language(value: str) -> Language:
    if value not in _FIRST_PROBES:
        raise ConfigurationError("language", "language must be en-US or pt-BR")
    return value


def first_probe(language: Language) -> str:
    return _FIRST_PROBES[language]


def second_probe(language: Language) -> str:
    return _SECOND_PROBES[language]


def persona_instruction(language: Language) -> str:
    return _PERSONA_INSTRUCTIONS[language]

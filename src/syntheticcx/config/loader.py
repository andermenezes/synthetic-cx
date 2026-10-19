"""YAML loading for the versioned project configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, TypeVar, cast, overload

import yaml
from pydantic import TypeAdapter, ValidationError

from syntheticcx.config.models import (
    AnyChannelGuide,
    DiscoveryDraft,
    ProjectConfig,
    ReferencedProjectConfig,
)
from syntheticcx.errors import ConfigurationError


@dataclass(frozen=True)
class LoadedConfig:
    """Validated configuration and original source identity, without source retention."""

    config: ProjectConfig
    source_content_hash: str
    guide_source_hash: str | None = None


ConfigType = TypeVar("ConfigType")


def _read_text(path: Path, operation: str) -> tuple[str, bytes]:
    try:
        source = path.read_bytes()
        return source.decode("utf-8"), source
    except (OSError, UnicodeError):
        raise ConfigurationError(operation, "could not read configuration file") from None


def _parse_mapping(source: str) -> Mapping[str, Any]:
    try:
        document: Any = yaml.safe_load(source)
    except yaml.YAMLError:
        raise ConfigurationError("parse configuration", "invalid YAML") from None
    if not isinstance(document, Mapping):
        raise ConfigurationError("parse configuration", "root document must be a mapping")
    return cast(Mapping[str, Any], document)


@overload
def _validate(document: Mapping[str, Any], model: type[ConfigType]) -> ConfigType: ...


@overload
def _validate(document: Mapping[str, Any], model: object) -> Any: ...


def _validate(document: Mapping[str, Any], model: Any) -> Any:
    try:
        return TypeAdapter(model).validate_python(document)
    except ValidationError as error:
        details = error.errors(include_url=False, include_input=False, include_context=False)
        messages = []
        for detail in details:
            parts = list(detail["loc"])
            # Keep legacy field paths stable despite the new tagged unions.
            if (
                len(parts) > 1
                and parts[0] == "channel"
                and parts[1]
                in {
                    "webchat_playwright",
                    "webchat_playwright_stateful",
                    "webchat_playwright_dynamic",
                    "whatsapp_web_playwright",
                }
            ):
                parts.pop(1)
            if detail["type"] in {"union_tag_invalid", "union_tag_not_found"}:
                if parts == ["channel"]:
                    parts.append("type")
            location = ".".join(str(part) for part in parts) or "configuration"
            # Union-tag diagnostics include the supplied value in their message.
            message = (
                "Unsupported or missing discriminator"
                if detail["type"] in {"union_tag_invalid", "union_tag_not_found"}
                else detail["msg"]
            )
            messages.append(f"{location}: {message}")
        raise ConfigurationError("validate configuration", "; ".join(messages)) from None


def load_config_with_source_hash(path: Path) -> LoadedConfig:
    """Read once, hashing exact bytes before UTF-8 decoding or YAML normalization."""
    text, source = _read_text(path, "read configuration")
    document = _parse_mapping(text)
    if "channel_guide" not in document:
        return LoadedConfig(_validate(document, ProjectConfig), sha256(source).hexdigest())
    if "channel" in document:
        raise ConfigurationError(
            "validate configuration", "channel and channel_guide are exclusive"
        )
    reference = _validate(document, ReferencedProjectConfig)
    guide_path = Path(reference.channel_guide)
    if guide_path.is_absolute():
        raise ConfigurationError("resolve channel guide", "channel_guide must be a relative path")
    location = path.parent / guide_path
    if not location.is_file():
        raise ConfigurationError("resolve channel guide", "channel_guide must name a local file")
    guide_text, guide_source = _read_text(location, "read channel guide")
    guide = _validate(_parse_mapping(guide_text), AnyChannelGuide)
    resolved = _validate(
        {
            **reference.model_dump(mode="json", exclude={"channel_guide"}),
            "channel": guide.channel.model_dump(mode="json"),
        },
        ProjectConfig,
    )
    return LoadedConfig(resolved, sha256(source).hexdigest(), sha256(guide_source).hexdigest())


def load_config(path: Path) -> ProjectConfig:
    """Load one YAML mapping into a project configuration."""
    return load_config_with_source_hash(path).config


def load_config_text(source: str) -> ProjectConfig:
    """Validate YAML text without triggering provider, browser, or storage work."""
    return _validate(_parse_mapping(source), ProjectConfig)


def load_discovery_draft(path: Path) -> DiscoveryDraft:
    """Validate discovery input without requiring a channel or reading credentials."""
    text, _ = _read_text(path, "read discovery draft")
    return _validate(_parse_mapping(text), DiscoveryDraft)

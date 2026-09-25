"""YAML loading for the versioned project configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import ValidationError

from syntheticcx.config.models import ProjectConfig
from syntheticcx.errors import ConfigurationError


@dataclass(frozen=True)
class LoadedConfig:
    """Validated configuration and original source identity, without source retention."""

    config: ProjectConfig
    source_content_hash: str


def load_config_with_source_hash(path: Path) -> LoadedConfig:
    """Read once, hashing exact bytes before UTF-8 decoding or YAML normalization."""
    try:
        source = path.read_bytes()
        text = source.decode("utf-8")
    except (OSError, UnicodeError):
        raise ConfigurationError(
            "read configuration", "could not read configuration file"
        ) from None
    return LoadedConfig(load_config_text(text), sha256(source).hexdigest())


def load_config(path: Path) -> ProjectConfig:
    """Load one YAML mapping into a project configuration."""
    return load_config_with_source_hash(path).config


def load_config_text(source: str) -> ProjectConfig:
    """Validate YAML text without triggering provider, browser, or storage work."""
    try:
        document: Any = yaml.safe_load(source)
    except yaml.YAMLError:
        raise ConfigurationError("parse configuration", "invalid YAML") from None
    if not isinstance(document, Mapping):
        raise ConfigurationError("parse configuration", "root document must be a mapping")
    try:
        return cast(ProjectConfig, ProjectConfig.model_validate(document))
    except ValidationError as error:
        details = error.errors(include_url=False, include_input=False, include_context=False)
        messages = []
        for detail in details:
            location = ".".join(str(part) for part in detail["loc"]) or "configuration"
            # Union-tag diagnostics include the supplied value in their message.
            message = (
                "Unsupported or missing discriminator"
                if detail["type"] in {"union_tag_invalid", "union_tag_not_found"}
                else detail["msg"]
            )
            messages.append(f"{location}: {message}")
        raise ConfigurationError("validate configuration", "; ".join(messages)) from None

"""JSON Schema generation for configuration."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

from syntheticcx.config.models import (
    DiscoveryDraft,
    InterfaceChannelGuide,
    ProjectConfig,
    ReferencedProjectConfig,
)
from syntheticcx.domain.artifacts import (
    ConfigArtifact,
    ConversationArtifact,
    EvaluationArtifact,
    MetadataArtifact,
)
from syntheticcx.domain.behavior import BehaviorReplayResult
from syntheticcx.domain.events import TypedTraceEvent
from syntheticcx.domain.models import Report


def export_project_schema(destination: Path) -> Path:
    """Write the canonical project schema with stable JSON formatting."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            TypeAdapter(ProjectConfig | ReferencedProjectConfig).json_schema(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def export_schemas(destination: Path) -> list[Path]:
    """Export artifact schemas and the configuration schema deterministically."""
    schemas = {
        "metadata": MetadataArtifact.model_json_schema(),
        "config": ConfigArtifact.model_json_schema(),
        "conversation": ConversationArtifact.model_json_schema(),
        "evaluation": EvaluationArtifact.model_json_schema(),
        "report": Report.model_json_schema(),
        "trace-event": TypeAdapter(TypedTraceEvent).json_schema(),
        "discovery-draft": DiscoveryDraft.model_json_schema(),
        "channel-guide": InterfaceChannelGuide.model_json_schema(),
        "behavior-replay": BehaviorReplayResult.model_json_schema(),
    }
    paths = [export_project_schema(destination / "project.schema.json")]
    for name, schema in schemas.items():
        path = destination / f"{name}.schema.json"
        path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths.append(path)
    return paths

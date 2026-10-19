"""Native version-1 journeys, programmatic detectors, and offline CLI replay."""

from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import Field
from typer.testing import CliRunner

from syntheticcx.application import run as application
from syntheticcx.cli import app
from syntheticcx.config.models import StrictModel
from syntheticcx.domain.behavior import BehavioralObservation, BehavioralSignal
from syntheticcx.domain.models import BehavioralDetectorMetadata, PersonaAction
from syntheticcx.runtime.behavior_signals import DetectorRegistration


class PatienceParameters(StrictModel):
    after_turn: int = Field(gt=0, strict=True)
    private_note: str


class PatiencePayload(StrictModel):
    private_note: str


class PatienceDetector:
    def __init__(
        self, metadata: BehavioralDetectorMetadata, parameters: PatienceParameters
    ) -> None:
        self.metadata = metadata
        self.parameters = parameters
        self.observations: list[BehavioralObservation] = []

    async def detect(self, observation: BehavioralObservation) -> tuple[BehavioralSignal, ...]:
        self.observations.append(observation)
        if (
            observation.kind != "customer_turn_completed"
            or observation.turn_count < self.parameters.after_turn
        ):
            return ()
        signal = BehavioralSignal(
            detector_id=self.metadata.detector_id,
            detector_type=self.metadata.detector_type,
            detector_version=self.metadata.detector_version,
            signal="custom_patience",
            deduplication_key=f"completed-{observation.turn_count}",
            observation_id=observation.observation_id,
            source_event_ids=observation.evidence_event_ids,
            source_message_ids=tuple(message.id for message in observation.messages),
            payload={"private_note": self.parameters.private_note},
        )
        # An identical duplicate must neither increment occurrence nor apply an effect twice.
        return signal, signal


class ContinuingCustomer:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.closed = False
        self.decisions = 0

    async def initialize(self, definition: Any, _context: Any) -> Any:
        return definition.fresh_state()

    async def respond(self, _observation: Any, state: Any) -> PersonaAction:
        self.decisions += 1
        return PersonaAction(
            message="My account number is SYNTHETIC-001. Please explain the invoice increase.",
            state=state,
            signal="continue",
        )

    async def close(self) -> None:
        self.closed = True


def run_native_case(
    root: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> tuple[Path, dict[str, Any], list[dict[str, Any]], list[PatienceDetector], ContinuingCustomer]:
    monkeypatch.syspath_prepend(str(Path("examples/telecom").resolve()))
    demo = importlib.import_module("run_demo")
    from syntheticcx.adapters.llm.preflight import CapabilityPreflight

    async def preflight(*_args: Any) -> CapabilityPreflight:
        return CapabilityPreflight()

    customer = ContinuingCustomer()
    monkeypatch.setenv("OPENAI_API_KEY", "offline-extension-secret")
    monkeypatch.setattr(application, "preflight_models", preflight)
    monkeypatch.setattr(
        application, "TinyTroupePersonaSimulator", lambda *_args, **_kwargs: customer
    )
    instances: list[PatienceDetector] = []

    def factory(metadata: BehavioralDetectorMetadata, parameters: Any) -> PatienceDetector:
        detector = PatienceDetector(
            metadata, PatienceParameters.model_validate(parameters.model_dump())
        )
        instances.append(detector)
        return detector

    registry = {
        "custom_patience": DetectorRegistration(
            detector_type="custom_patience",
            version="1",
            parameter_model=PatienceParameters,
            signal_schemas={"custom_patience": PatiencePayload},
            mode="deterministic",
            factory=factory,
        )
    }
    with demo.FixtureServer() as fixture:
        document = demo.configured_example(
            fixture.base_url,
            variant="repeated_question" if case == "repeated_information" else "success",
            persona="standard",
            run_root=root / "runs",
        )
        document["scenario"]["success_criteria"] = [
            {
                "id": "hidden-evaluator-marker",
                "evaluator_type": "turn_count",
                "required": True,
                "result_kind": "boolean",
                "evidence_policy": "none",
                "parameters": {"max_turns": 8},
            }
        ]
        document["telemetry"] = {"enabled": False}
        document["execution"]["max_turns"] = 8
        document["execution"]["greeting_grace_seconds"] = 1
        document["persona"]["behavioral_rules"] = [
            {
                "id": "leave",
                "priority": 10,
                "when": {
                    "signal": case,
                    "occurrence_at_least": 6 if case == "customer_turn_completed" else 2,
                },
                "effects": {"abandon": True},
                "fire": "once",
            }
        ]
        if case == "custom_patience":
            document["persona"]["behavioral_rules"][0]["when"]["occurrence_at_least"] = 1
            document["persona"]["behavioral_rules"][0]["effects"]["frustration_delta"] = 0.2
            document["behavior"] = {
                "detectors": [
                    {
                        "id": "custom-patience",
                        "type": "custom_patience",
                        "parameters": {"after_turn": 2, "private_note": "PRIVATE-DETECTOR-CANARY"},
                    }
                ]
            }
            document["privacy"]["synthetic_id_patterns"] = ["PRIVATE-DETECTOR-CANARY"]
        source = root / "scenario.yaml"
        source.write_text(yaml.safe_dump(document), encoding="utf-8")
        result = asyncio.run(
            application.execute_run(
                source, behavior_registry=registry if case == "custom_patience" else None
            )
        )
    assert result.exit_code == 0 and result.report is not None, (
        result.exit_code,
        result.report,
        result.partial_reason,
    )
    directory = Path(result.artifact_location)
    report = json.loads((directory / "report.json").read_text())
    events = [json.loads(line) for line in (directory / "trace.jsonl").read_text().splitlines()]
    return directory, report, events, instances, customer


@pytest.mark.parametrize(
    "case,turns",
    [("repeated_information", 2), ("customer_turn_completed", 6), ("custom_patience", 2)],
)
def test_native_version_one_rule_and_actual_offline_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, turns: int
) -> None:
    directory, report, events, instances, customer = run_native_case(tmp_path, monkeypatch, case)
    assert report["termination_reason"] == "abandonment"
    assert report["turn_count"] == customer.decisions == turns
    assert customer.closed
    matching = [
        event
        for event in events
        if event["type"] == "behavioral_state_changed" and event["data"]["applied_rule_ids"]
    ]
    assert [event["data"]["applied_rule_ids"] for event in matching] == [["leave"]]
    assert matching[0]["data"]["after"]["abandoned"] is True
    assert report["repetition_count"] == (2 if case == "repeated_information" else 0)
    config = yaml.safe_load((directory / "config.yaml").read_text())
    assert "version" not in config["effective_config"]
    positions = {event["id"]: event["sequence"] for event in events}
    for event in events:
        if event["type"] == "behavioral_signal_recorded":
            assert all(
                positions[reference] < event["sequence"]
                for reference in event["data"]["signal"]["source_event_ids"]
            )
    if case == "custom_patience":
        assert len(instances) == 1
        assert instances[0].parameters.private_note == "PRIVATE-DETECTOR-CANARY"
        assert all(
            "hidden-evaluator-marker" not in observation.model_dump_json()
            for observation in instances[0].observations
        )
        assert matching[0]["data"]["occurrence"] == 1
        assert matching[0]["data"]["after"]["frustration"] == pytest.approx(0.3)
        assert sum(event["type"] == "behavioral_signal_recorded" for event in events) == 1
        assert "PRIVATE-DETECTOR-CANARY" not in "".join(
            path.read_text()
            for path in directory.iterdir()
            if path.suffix in {".json", ".jsonl", ".yaml", ".md"}
        )
    before = {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    monkeypatch.delenv("OPENAI_API_KEY")

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("offline replay must not compose runtime adapters")

    monkeypatch.setattr(application, "execute_run", forbidden)
    output = tmp_path / "replay.json"
    replay = CliRunner().invoke(app, ["replay-behavior", str(directory), "--output", str(output)])
    assert replay.exit_code == 0, replay.output
    result = json.loads(output.read_text())
    assert result["version"] == result["source_config_version"] == "1"
    assert result["completeness"] == "complete" and result["comparison_findings"] == []
    assert result["calculated_transitions"][-1]["after"]["abandoned"] is True
    assert {
        path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()
    } == before
    if case == "customer_turn_completed":
        rules = tmp_path / "rules.yaml"
        rules.write_text(
            yaml.safe_dump(
                {
                    "rules": [
                        {
                            "id": "earlier-exit",
                            "priority": 0,
                            "when": {"signal": "customer_turn_completed", "occurrence_at_least": 2},
                            "effects": {"abandon": True},
                            "fire": "once",
                        }
                    ],
                }
            )
        )
        counterfactual = tmp_path / "counterfactual.json"
        replay = CliRunner().invoke(
            app,
            [
                "replay-behavior",
                str(directory),
                "--output",
                str(counterfactual),
                "--rules",
                str(rules),
            ],
        )
        assert replay.exit_code == 0, replay.output
        alternate = json.loads(counterfactual.read_text())
        assert alternate["mode"] == "counterfactual"
        assert [transition["occurrence"] for transition in alternate["calculated_transitions"]] == [
            1,
            2,
        ]
        assert len(alternate["outside_continuation_observation_ids"]) == 4

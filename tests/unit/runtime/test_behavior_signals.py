"""Trusted extension preflight, isolated observations, and bounded output validation."""

import asyncio
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, Field

from syntheticcx.config.loader import load_config
from syntheticcx.config.models import BehaviorConfig, BehaviorRule, StrictModel
from syntheticcx.config.settings import prepare_settings
from syntheticcx.domain.behavior import BehavioralObservation, BehavioralSignal
from syntheticcx.domain.models import (
    BehavioralDetectorMetadata,
    Message,
    ModelInteraction,
    PersonaState,
)
from syntheticcx.errors import BehaviorError, ConfigurationError
from syntheticcx.runtime.behavior_signals import (
    BUILTIN_DETECTORS,
    DetectorRegistration,
    prepare_behavior,
)
from syntheticcx.runtime.clock import RealWaitController

NOW = datetime(2026, 10, 10, tzinfo=UTC)


class Parameters(StrictModel):
    threshold: int = Field(ge=1, strict=True)


class Payload(StrictModel):
    reason: str


def observation() -> BehavioralObservation:
    return BehavioralObservation(
        kind="greeting",
        messages=(
            Message(
                id="visible",
                sequence=0,
                actor="assistant",
                content="Original private visible text",
                observed_at=NOW,
            ),
        ),
        evidence_event_ids=("committed",),
        state=PersonaState(frustration=0.2, willingness_to_continue=0.8),
        turn_count=0,
    )


def signal(
    metadata: BehavioralDetectorMetadata, boundary: BehavioralObservation, **changes: object
) -> BehavioralSignal:
    data = {
        "detector_id": metadata.detector_id,
        "detector_type": metadata.detector_type,
        "detector_version": metadata.detector_version,
        "signal": "custom_notice",
        "deduplication_key": "committed",
        "observation_id": boundary.observation_id,
        "source_event_ids": ("committed",),
        "source_message_ids": ("visible",),
        "payload": {"reason": "visible text"},
        **changes,
    }
    return BehavioralSignal.model_validate(data)


def config(parameters: dict[str, object] | None = None) -> BehaviorConfig:
    return BehaviorConfig.model_validate(
        {
            "detectors": [
                {
                    "id": "consumer",
                    "type": "custom_type",
                    "version": "exact-3",
                    "parameters": parameters if parameters is not None else {"threshold": 2},
                }
            ]
        }
    )


def registration(factory, **changes: object) -> DetectorRegistration:
    values = {
        "detector_type": "custom_type",
        "version": "exact-3",
        "parameter_model": Parameters,
        "signal_schemas": {"custom_notice": Payload},
        "mode": "deterministic",
        "factory": factory,
        **changes,
    }
    return DetectorRegistration(**values)


def classify(detectors, boundary: BehavioralObservation | None = None, **changes: object):
    async def run():
        return await detectors.detect(
            boundary or observation(),
            event_ids=frozenset({"committed"}),
            message_ids=frozenset({"visible"}),
            wait=RealWaitController(),
            timeout_seconds=changes.get("timeout", 1),
            remaining_time=lambda: changes.get("remaining", 1),
        )

    return asyncio.run(run())


def test_registry_is_explicit_immutable_and_preflight_does_not_construct() -> None:
    factories = []

    def factory(metadata, parameters):
        factories.append((metadata, parameters))

        class Detector:
            async def detect(self, boundary):
                return (signal(metadata, boundary),)

        return Detector()

    registered = registration(factory)
    with pytest.raises(FrozenInstanceError):
        registered.version = "changed"
    with pytest.raises(TypeError):
        registered.signal_schemas["new"] = Payload
    with pytest.raises(TypeError):
        BUILTIN_DETECTORS["new"] = registered
    prepared = prepare_behavior(config(), [], {"custom_type": registered})
    assert not factories
    first, second = prepared.instantiate(), prepared.instantiate()
    assert len(factories) == 2
    assert factories[0][1].threshold == 2
    assert first.metadata == second.metadata == prepared.metadata
    boundary = observation()
    assert len(classify(first, boundary)) == 1
    assert classify(first, boundary) == ()
    assert len(classify(second, boundary)) == 1


@pytest.mark.parametrize(
    "problem",
    ["unknown", "version", "parameters", "schema", "ownership", "rules", "builtin_override"],
)
def test_invalid_registration_configuration_precedes_credentials_and_factories(
    problem: str,
) -> None:
    def forbidden(*args):
        raise AssertionError("factory must not be invoked")

    registered = registration(forbidden)
    selected = config()
    extensions = {"custom_type": registered}
    rules = []
    if problem == "unknown":
        extensions = {}
    elif problem == "version":
        extensions["custom_type"] = replace(registered, version="different")
    elif problem == "parameters":
        selected = config({"threshold": "2", "private": "never leak"})
    elif problem == "schema":
        extensions["custom_type"] = replace(registered, signal_schemas={"custom_notice": BaseModel})
    elif problem == "ownership":
        selected = BehaviorConfig.model_validate(
            {
                "detectors": [
                    *selected.model_dump()["detectors"],
                    {
                        "id": "another",
                        "type": "custom_type",
                        "version": "exact-3",
                        "parameters": {"threshold": 2},
                    },
                ]
            }
        )
    elif problem == "rules":
        rules = [
            BehaviorRule(
                id="rule", priority=0, when={"signal": "missing"}, effects={"abandon": True}
            )
        ]
    elif problem == "builtin_override":
        extensions["repeated_information"] = registered
    project = load_config(Path("examples/telecom/scenario.yaml"))
    project = project.model_copy(
        update={
            "behavior": selected,
            "persona": project.persona.model_copy(update={"behavioral_rules": rules}),
        }
    )
    with pytest.raises(ConfigurationError) as caught:
        prepare_settings(project, {}, behavior_registry=extensions)
    assert caught.value.operation == "prepare behavior"
    assert "never leak" not in str(caught.value)


@pytest.mark.parametrize(
    "problem",
    [
        "detector_id",
        "detector_type",
        "detector_version",
        "signal",
        "observation_id",
        "source_event_ids",
        "source_message_ids",
        "payload",
        "overflow",
        "conflict",
        "constructed_invalid",
    ],
)
def test_entire_output_is_validated_before_any_signal_can_escape(problem: str) -> None:
    def factory(metadata, parameters):
        class Detector:
            async def detect(self, boundary):
                good = signal(metadata, boundary)
                changes = {
                    "detector_id": "wrong",
                    "detector_type": "wrong",
                    "detector_version": "wrong",
                    "signal": "wrong",
                    "observation_id": observation().observation_id,
                    "source_event_ids": ("future",),
                    "source_message_ids": ("missing",),
                    "payload": {"reason": 42},
                }
                if problem == "overflow":
                    return tuple(
                        signal(metadata, boundary, deduplication_key=f"key-{i}") for i in range(33)
                    )
                if problem == "conflict":
                    return (good, signal(metadata, boundary, payload={"reason": "different"}))
                if problem == "constructed_invalid":
                    return (good.model_copy(update={"payload": {"reason": float("nan")}}),)
                return (good, signal(metadata, boundary, **{problem: changes[problem]}))

        return Detector()

    detectors = prepare_behavior(config(), [], {"custom_type": registration(factory)}).instantiate()
    with pytest.raises(BehaviorError) as caught:
        classify(detectors)
    assert caught.value.operation == "behavior.detect"
    assert "Original private" not in str(caught.value)


def test_identical_duplicates_are_ignored_and_cross_boundary_conflicts_fail() -> None:
    calls = []

    def factory(metadata, parameters):
        class Detector:
            cached = None

            async def detect(self, boundary):
                calls.append(boundary)
                if self.cached is None:
                    self.cached = signal(metadata, boundary, payload={"reason": "same"})
                candidate = self.cached.model_copy(deep=True)
                if len(calls) == 3:
                    candidate = candidate.model_copy(update={"payload": {"reason": "changed"}})
                return (candidate, candidate.model_copy(deep=True))

        return Detector()

    detectors = prepare_behavior(config(), [], {"custom_type": registration(factory)}).instantiate()
    assert len(classify(detectors)) == 1
    assert classify(detectors) == ()
    with pytest.raises(BehaviorError):
        classify(detectors)


def test_reused_key_with_new_observation_identity_is_a_conflict() -> None:
    def factory(metadata, parameters):
        class Detector:
            async def detect(self, boundary):
                return (signal(metadata, boundary),)

        return Detector()

    detectors = prepare_behavior(config(), [], {"custom_type": registration(factory)}).instantiate()
    assert len(classify(detectors)) == 1
    with pytest.raises(BehaviorError):
        classify(detectors)


def test_all_detectors_receive_detached_same_preboundary_state() -> None:
    received = []

    def factory(metadata, parameters):
        class Detector:
            async def detect(self, boundary):
                received.append(boundary.model_copy(deep=True))
                boundary.state.__dict__["frustration"] = 1.0
                boundary.messages[0].metadata["mutated"] = True
                return ()

        return Detector()

    registered = registration(factory)
    other = replace(registered, detector_type="second", signal_schemas={"second_signal": Payload})
    selected = BehaviorConfig.model_validate(
        {
            "detectors": [
                *config().model_dump()["detectors"],
                {
                    "id": "second-id",
                    "type": "second",
                    "version": "exact-3",
                    "parameters": {"threshold": 2},
                },
            ]
        }
    )
    detectors = prepare_behavior(
        selected, [], {"custom_type": registered, "second": other}
    ).instantiate()
    original = observation()
    assert classify(detectors, original) == ()
    assert [boundary.state.frustration for boundary in received] == [0.2, 0.2]
    assert original.state.frustration == 0.2
    assert original.messages[0].metadata == {}
    assert received[0].messages[0].content == "Original private visible text"
    assert set(type(original).model_fields) == {
        "observation_id",
        "kind",
        "messages",
        "evidence_event_ids",
        "state",
        "turn_count",
        "last_customer_message",
        "repetition_facts",
    }


def test_async_timeout_is_clipped_cancels_and_does_not_retry() -> None:
    attempts = []
    cancelled = []

    def factory(metadata, parameters):
        class Detector:
            async def detect(self, boundary):
                attempts.append(1)
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.append(1)

        return Detector()

    detectors = prepare_behavior(config(), [], {"custom_type": registration(factory)}).instantiate()
    with pytest.raises(BehaviorError):
        classify(detectors, timeout=20, remaining=0.01)
    assert attempts == cancelled == [1]


def test_blocking_detector_is_bounded_and_late_result_is_not_used() -> None:
    release, started = Event(), Event()

    def factory(metadata, parameters):
        class Detector:
            def detect(self, boundary):
                started.set()
                release.wait()
                return (signal(metadata, boundary),)

        return Detector()

    detectors = prepare_behavior(config(), [], {"custom_type": registration(factory)}).instantiate()
    try:
        with pytest.raises(BehaviorError):
            classify(detectors, remaining=0.01)
        assert started.is_set()
    finally:
        release.set()


def test_sanitization_preserves_identity_and_revalidates_payload() -> None:
    class Detector:
        async def detect(self, boundary):
            return ()

    detectors = prepare_behavior(
        config(), [], {"custom_type": registration(lambda m, p: Detector())}
    ).instantiate()
    candidate = signal(detectors.metadata[0], observation())
    detectors.validate_sanitized(
        candidate, candidate.model_copy(update={"payload": {"reason": "[REDACTED]"}})
    )
    for changes in (
        {"detector_id": "changed"},
        {"payload": {}},
        {"source_event_ids": ("changed",)},
    ):
        with pytest.raises(BehaviorError):
            detectors.validate_sanitized(candidate, candidate.model_copy(update=changes))


def test_model_backed_registration_requires_owned_interaction_evidence() -> None:
    def factory(metadata, parameters):
        class Detector:
            async def detect(self, boundary):
                return (signal(metadata, boundary),)

        return Detector()

    detectors = prepare_behavior(
        config(), [], {"custom_type": registration(factory, mode="model_backed")}
    ).instantiate()
    with pytest.raises(BehaviorError):
        classify(detectors)


def test_model_backed_interactions_are_detached_and_truthfully_reference_committed_input() -> None:
    def factory(metadata, parameters):
        class Detector:
            async def detect(self, boundary):
                return (signal(metadata, boundary),)

            def take_interactions(self):
                return (
                    ModelInteraction(
                        id="classification-call",
                        identity={"provider": "offline", "model": "scripted"},
                        started_at=NOW,
                        ended_at=NOW,
                        usage={"unavailable_reason": "offline"},
                    ),
                )

        return Detector()

    detectors = prepare_behavior(
        config(), [], {"custom_type": registration(factory, mode="model_backed")}
    ).instantiate()
    assert len(classify(detectors)) == 1
    evidence = detectors.take_interactions()
    assert evidence[0].input_event_ids == ["committed"]
    assert evidence[0].output_event_ids == []
    assert "output_event_not_yet_committed" in evidence[0].capture_omissions
    assert detectors.take_interactions() == ()


@pytest.mark.parametrize(
    "reference", ["https://private.invalid/schema", "file:///private/data", "relative.json"]
)
def test_external_payload_schema_references_fail_local_preflight(reference) -> None:
    class RemotePayload(StrictModel):
        model_config = {"json_schema_extra": {"$ref": reference}}

    def forbidden(metadata, parameters):
        pytest.fail("schema preflight must precede factories")

    registered = registration(forbidden, signal_schemas={"custom_notice": RemotePayload})
    with pytest.raises(ConfigurationError) as caught:
        prepare_behavior(config(), [], {"custom_type": registered})
    assert caught.value.operation == "prepare behavior"
    assert reference not in str(caught.value)


def test_declared_json_schema_is_enforced_after_redaction_not_only_python_fields() -> None:
    class RestrictedPayload(Payload):
        model_config = {
            "json_schema_extra": {"properties": {"reason": {"type": "string", "maxLength": 5}}}
        }

    def factory(metadata, parameters):
        class Detector:
            async def detect(self, boundary):
                return (signal(metadata, boundary, payload={"reason": "short"}),)

        return Detector()

    registered = registration(factory, signal_schemas={"custom_notice": RestrictedPayload})
    detectors = prepare_behavior(config(), [], {"custom_type": registered}).instantiate()
    candidate = classify(detectors)[0]
    # The Python field accepts this, but the persisted advertised JSON Schema does not.
    RestrictedPayload.model_validate({"reason": "[REDACTED]"})
    with pytest.raises(BehaviorError):
        detectors.validate_sanitized(
            candidate, candidate.model_copy(update={"payload": {"reason": "[REDACTED]"}})
        )


def test_typed_json_payload_accepts_canonical_uuid_without_python_object_coercion() -> None:
    class IdentityPayload(StrictModel):
        reference: UUID

    reference = str(uuid4())

    def factory(metadata, parameters):
        class Detector:
            async def detect(self, boundary):
                return (signal(metadata, boundary, payload={"reference": reference}),)

        return Detector()

    registered = registration(factory, signal_schemas={"custom_notice": IdentityPayload})
    detectors = prepare_behavior(config(), [], {"custom_type": registered}).instantiate()
    candidate = classify(detectors)[0]
    assert candidate.payload == {"reference": reference}
    assert (
        detectors.metadata[0].signal_schemas["custom_notice"]["properties"]["reference"]["format"]
        == "uuid"
    )

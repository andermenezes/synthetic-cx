"""Explicit trusted detector registration and bounded, per-run signal classification."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, NoReturn, Protocol, cast

from jsonschema import Draft202012Validator
from pydantic import BaseModel, TypeAdapter
from referencing import Registry
from referencing.exceptions import NoSuchResource

from syntheticcx._compat.blocking import run_blocking_bounded
from syntheticcx.config.models import BehaviorConfig, BehaviorRule, StrictModel
from syntheticcx.domain.behavior import (
    BehavioralDetectorOutput,
    BehavioralObservation,
    BehavioralSignal,
)
from syntheticcx.domain.models import BehavioralDetectorMetadata, ModelInteraction
from syntheticcx.domain.protocols import BehavioralSignalDetector, WaitController
from syntheticcx.errors import BehaviorError, ConfigurationError


def _deny_schema_retrieval(uri: str) -> NoReturn:
    raise NoSuchResource(ref=uri)


_SCHEMA_REGISTRY: Registry[object] = Registry(retrieve=_deny_schema_retrieval)


class BlockingBehavioralSignalDetector(Protocol):
    """Trusted synchronous classifiers use a bounded daemon worker."""

    def detect(self, observation: BehavioralObservation) -> tuple[BehavioralSignal, ...]: ...


Detector = BehavioralSignalDetector | BlockingBehavioralSignalDetector


class EmptyDetectorParameters(StrictModel):
    """The shipped detectors accept no configuration parameters or payload fields."""


def _validate_document(model: type[BaseModel], document: Mapping[str, object]) -> BaseModel:
    validated = model.model_validate_json(json.dumps(document, allow_nan=False), strict=True)
    if validated.model_dump(mode="json", exclude_unset=True) != document:
        raise ValueError("registered schema changes or drops supplied data")
    return validated


def _validate_schema(schema: Mapping[str, object]) -> None:
    """Persist only self-contained JSON Schema, without remote reference retrieval."""
    Draft202012Validator.check_schema(schema)
    pending: list[object] = [schema]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            for key in ("$ref", "$dynamicRef"):
                if key in value and (
                    not isinstance(value[key], str) or not value[key].startswith("#")
                ):
                    raise ValueError("registered payload schema has external references")
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)


def _payload_schema(model: type[BaseModel]) -> dict[str, object]:
    """Persist validation keywords, not prose annotations that privacy may redact."""
    schema: dict[str, object] = model.model_json_schema(mode="validation")
    pending = [schema]
    while pending:
        current = pending.pop()
        for annotation in ("title", "description", "examples", "$comment"):
            current.pop(annotation, None)
        for keyword in ("properties", "patternProperties", "$defs", "dependentSchemas"):
            children = current.get(keyword)
            if isinstance(children, dict):
                pending.extend(child for child in children.values() if isinstance(child, dict))
        for keyword in (
            "items",
            "contains",
            "additionalProperties",
            "unevaluatedProperties",
            "propertyNames",
            "not",
            "if",
            "then",
            "else",
            "unevaluatedItems",
            "contentSchema",
        ):
            child = current.get(keyword)
            if isinstance(child, dict):
                pending.append(child)
        for keyword in ("allOf", "anyOf", "oneOf", "prefixItems"):
            children = current.get(keyword)
            if isinstance(children, list):
                pending.extend(child for child in children if isinstance(child, dict))
    return schema


@dataclass(frozen=True)
class DetectorRegistration:
    """Trusted implementation metadata; configuration never imports executable code.

    Factories receive only the configured identity and validated parameters.
    Async detect methods must cooperate with cancellation; synchronous detect
    methods run in the existing bounded daemon-worker boundary.
    """

    detector_type: str
    version: str
    parameter_model: type[BaseModel]
    signal_schemas: Mapping[str, type[BaseModel]]
    mode: Literal["deterministic", "model_backed"]
    factory: Callable[[BehavioralDetectorMetadata, BaseModel], Detector]

    def __post_init__(self) -> None:
        object.__setattr__(self, "signal_schemas", MappingProxyType(dict(self.signal_schemas)))


@dataclass(frozen=True)
class PreparedDetector:
    metadata: BehavioralDetectorMetadata
    registration: DetectorRegistration
    parameters: BaseModel


@dataclass(frozen=True)
class PreparedBehavior:
    detectors: tuple[PreparedDetector, ...]

    @property
    def metadata(self) -> tuple[BehavioralDetectorMetadata, ...]:
        return tuple(detector.metadata.model_copy(deep=True) for detector in self.detectors)

    def instantiate(self) -> BehaviorDetectors:
        """Create fresh instances; no detector state is shared between journeys."""
        try:
            instances = tuple(
                (
                    detector,
                    detector.registration.factory(
                        detector.metadata.model_copy(deep=True),
                        detector.parameters.model_copy(deep=True),
                    ),
                )
                for detector in self.detectors
            )
            if any(not callable(getattr(instance, "detect", None)) for _, instance in instances):
                raise ValueError("factory did not produce a detector")
            return BehaviorDetectors(instances)
        except Exception:
            raise ConfigurationError(
                "prepare behavior", "behavior detector construction failed"
            ) from None


class _CompletedTurnDetector:
    def __init__(self, metadata: BehavioralDetectorMetadata) -> None:
        self.metadata = metadata

    async def detect(self, observation: BehavioralObservation) -> tuple[BehavioralSignal, ...]:
        if observation.kind != "customer_turn_completed":
            return ()
        return (
            BehavioralSignal(
                detector_id=self.metadata.detector_id,
                detector_type=self.metadata.detector_type,
                detector_version=self.metadata.detector_version,
                signal="customer_turn_completed",
                deduplication_key=str(observation.observation_id),
                observation_id=observation.observation_id,
                source_event_ids=observation.evidence_event_ids,
                source_message_ids=tuple(message.id for message in observation.messages),
            ),
        )


class _RepeatedInformationDetector:
    def __init__(self, metadata: BehavioralDetectorMetadata) -> None:
        self.metadata = metadata

    async def detect(self, observation: BehavioralObservation) -> tuple[BehavioralSignal, ...]:
        return tuple(
            BehavioralSignal(
                detector_id=self.metadata.detector_id,
                detector_type=self.metadata.detector_type,
                detector_version=self.metadata.detector_version,
                signal="repeated_information",
                deduplication_key=fact.event_id,
                observation_id=observation.observation_id,
                source_event_ids=(fact.event_id,),
                source_message_ids=(fact.original_message_id, fact.request_message_id),
            )
            for fact in observation.repetition_facts
        )


BUILTIN_DETECTORS: Mapping[str, DetectorRegistration] = MappingProxyType(
    {
        "repeated_information": DetectorRegistration(
            detector_type="repeated_information",
            version="1",
            parameter_model=EmptyDetectorParameters,
            signal_schemas={"repeated_information": EmptyDetectorParameters},
            mode="deterministic",
            factory=lambda metadata, parameters: _RepeatedInformationDetector(metadata),
        ),
        "customer_turn_completed": DetectorRegistration(
            detector_type="customer_turn_completed",
            version="1",
            parameter_model=EmptyDetectorParameters,
            signal_schemas={"customer_turn_completed": EmptyDetectorParameters},
            mode="deterministic",
            factory=lambda metadata, parameters: _CompletedTurnDetector(metadata),
        ),
    }
)


def prepare_behavior(
    behavior: BehaviorConfig,
    rules: list[BehaviorRule],
    registry: Mapping[str, DetectorRegistration] | None = None,
) -> PreparedBehavior:
    """Resolve all configuration before credentials, factories, or external work."""
    registrations = dict(BUILTIN_DETECTORS)
    if registry is not None:
        if set(registrations).intersection(registry):
            raise ConfigurationError(
                "prepare behavior", "built-in detector registration cannot be replaced"
            )
        registrations.update(registry)
    detectors: list[PreparedDetector] = []
    signals: set[str] = set()
    try:
        for configured in behavior.detectors:
            registration = registrations.get(configured.type)
            if registration is None:
                raise ValueError("unregistered detector")
            if registration.detector_type != configured.type:
                raise ValueError("unavailable implementation")
            metadata = BehavioralDetectorMetadata(
                detector_id=configured.id,
                detector_type=registration.detector_type,
                detector_version=registration.version,
                mode=registration.mode,
                output_signals=tuple(registration.signal_schemas),
                signal_schemas={
                    name: _payload_schema(model)
                    for name, model in registration.signal_schemas.items()
                },
            )
            if signals.intersection(metadata.output_signals):
                raise ValueError("ambiguous signal ownership")
            for model in (registration.parameter_model, *registration.signal_schemas.values()):
                if not issubclass(model, BaseModel) or model.model_config.get("extra") != "forbid":
                    raise ValueError("registered schemas must forbid unknown fields")
            for schema in metadata.signal_schemas.values():
                _validate_schema(schema)
            if not callable(registration.factory):
                raise ValueError("invalid factory")
            parameters = _validate_document(registration.parameter_model, configured.parameters)
            signals.update(metadata.output_signals)
            detectors.append(PreparedDetector(metadata, registration, parameters))
        if any(rule.when.signal not in signals for rule in rules):
            raise ValueError("rule signal is unavailable")
    except Exception:
        raise ConfigurationError(
            "prepare behavior", "behavior detector configuration is invalid or unavailable"
        ) from None
    return PreparedBehavior(tuple(detectors))


class BehaviorDetectors:
    """One journey's detector instances and deduplication memory, without effects."""

    def __init__(self, instances: tuple[tuple[PreparedDetector, Detector], ...]) -> None:
        self._instances = instances
        self._interactions: tuple[ModelInteraction, ...] = ()
        self._processed: dict[tuple[str, str], dict[str, object]] = {}
        self._output: TypeAdapter[tuple[BehavioralSignal, ...]] = TypeAdapter(
            BehavioralDetectorOutput
        )

    @property
    def metadata(self) -> tuple[BehavioralDetectorMetadata, ...]:
        return tuple(prepared.metadata.model_copy(deep=True) for prepared, _ in self._instances)

    def take_interactions(self) -> tuple[ModelInteraction, ...]:
        interactions = self._interactions
        self._interactions = ()
        return interactions

    def validate_sanitized(self, original: BehavioralSignal, sanitized: BehavioralSignal) -> None:
        """Redaction may change payload content, never behavioral identity/citations."""
        try:
            if original.model_dump(exclude={"payload"}) != sanitized.model_dump(
                exclude={"payload"}
            ):
                raise ValueError("redaction changed signal identity")
            prepared = next(
                p for p, _ in self._instances if p.metadata.detector_id == original.detector_id
            )
            _validate_document(
                prepared.registration.signal_schemas[original.signal], sanitized.payload
            )
            Draft202012Validator(
                prepared.metadata.signal_schemas[original.signal],
                registry=_SCHEMA_REGISTRY,
                format_checker=Draft202012Validator.FORMAT_CHECKER,
            ).validate(sanitized.payload)
        except Exception:
            raise BehaviorError(
                "behavior.sanitize", "behavior evidence cannot be safely persisted"
            ) from None

    async def detect(
        self,
        observation: BehavioralObservation,
        *,
        event_ids: frozenset[str],
        message_ids: frozenset[str],
        wait: WaitController,
        timeout_seconds: float,
        remaining_time: Callable[[], float],
    ) -> tuple[BehavioralSignal, ...]:
        pending = dict(self._processed)
        unique: list[BehavioralSignal] = []
        interactions: list[ModelInteraction] = []
        try:
            for prepared, detector in self._instances:
                timeout = min(timeout_seconds, remaining_time())
                if timeout <= 0:
                    raise TimeoutError
                detached = observation.model_copy(deep=True)
                output: object
                if inspect.iscoroutinefunction(detector.detect):
                    output = await wait.run(
                        cast(Awaitable[object], detector.detect(detached)), timeout
                    )
                else:
                    bounded = await run_blocking_bounded(
                        lambda: detector.detect(detached), timeout_seconds=timeout
                    )
                    if bounded.cancelled:
                        raise asyncio.CancelledError
                    if bounded.timed_out:
                        raise TimeoutError
                    output = bounded.value
                signals = tuple(
                    BehavioralSignal.model_validate(signal.model_dump(), strict=True)
                    for signal in self._output.validate_python(output, strict=True)
                )
                if prepared.metadata.mode == "model_backed":
                    take_interactions = getattr(detector, "take_interactions", None)
                    if not callable(take_interactions):
                        raise ValueError("model-backed detector lacks interaction evidence")
                    bounded_evidence = await run_blocking_bounded(
                        take_interactions,
                        timeout_seconds=min(timeout_seconds, remaining_time()),
                    )
                    if bounded_evidence.cancelled:
                        raise asyncio.CancelledError
                    if bounded_evidence.timed_out:
                        raise TimeoutError
                    evidence = bounded_evidence.value
                    if not isinstance(evidence, tuple) or not 1 <= len(evidence) <= 32:
                        raise ValueError(
                            "model-backed interaction evidence is unavailable or unbounded"
                        )
                    for interaction in evidence:
                        if not isinstance(interaction, ModelInteraction):
                            raise ValueError("invalid model interaction evidence")
                        interaction = ModelInteraction.model_validate(
                            interaction.model_dump(), strict=True
                        )
                        if (
                            not set(interaction.input_event_ids).issubset(event_ids)
                            or interaction.output_event_ids
                        ):
                            raise ValueError(
                                "model interaction evidence references uncommitted events"
                            )
                        interactions.append(
                            interaction.model_copy(
                                update={
                                    "input_event_ids": interaction.input_event_ids
                                    or list(observation.evidence_event_ids),
                                    "capture_omissions": list(
                                        dict.fromkeys(
                                            [
                                                *interaction.capture_omissions,
                                                "output_event_not_yet_committed",
                                            ]
                                        )
                                    ),
                                },
                                deep=True,
                            )
                        )
                for signal in signals:
                    metadata = prepared.metadata
                    if (
                        signal.detector_id != metadata.detector_id
                        or signal.detector_type != metadata.detector_type
                        or signal.detector_version != metadata.detector_version
                        or signal.signal not in metadata.output_signals
                        or not set(signal.source_event_ids).issubset(event_ids)
                        or not set(signal.source_message_ids).issubset(message_ids)
                    ):
                        raise ValueError("invalid signal identity or references")
                    _validate_document(
                        prepared.registration.signal_schemas[signal.signal], signal.payload
                    )
                    Draft202012Validator(
                        metadata.signal_schemas[signal.signal],
                        registry=_SCHEMA_REGISTRY,
                        format_checker=Draft202012Validator.FORMAT_CHECKER,
                    ).validate(signal.payload)
                    key = (signal.detector_id, signal.deduplication_key)
                    identity = signal.model_dump()
                    if key in pending:
                        if pending[key] != identity:
                            raise ValueError("conflicting signal deduplication key")
                        continue
                    if signal.observation_id != observation.observation_id:
                        raise ValueError("new signal belongs to another observation")
                    pending[key] = identity
                    unique.append(signal.model_copy(deep=True))
        except asyncio.CancelledError:
            raise
        except Exception:
            raise BehaviorError(
                "behavior.detect", "behavior detector failed or returned invalid evidence"
            ) from None
        self._processed = pending
        self._interactions = tuple(interactions)
        return tuple(unique)

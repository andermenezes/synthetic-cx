"""Explicit trusted evaluator composition; YAML never imports implementations."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel

from syntheticcx.config.models import (
    Criterion,
    JourneyTimeoutParameters,
    RepeatedInformationParameters,
    SemanticGoalParameters,
    TurnCountParameters,
)
from syntheticcx.domain.protocols import Evaluator, LLMProvider
from syntheticcx.errors import ConfigurationError
from syntheticcx.evaluation.deterministic import (
    JourneyTimeoutEvaluator,
    RepeatedInformationEvaluator,
    TurnCountEvaluator,
)
from syntheticcx.evaluation.semantic import SemanticGoalEvaluator
from syntheticcx.privacy import Redactor


@dataclass(frozen=True)
class EvaluationDependencies:
    provider: LLMProvider | None
    redactor: Redactor


@dataclass(frozen=True)
class EvaluatorRegistration:
    evaluator_type: str
    parameter_model: type[BaseModel]
    factory: Callable[[BaseModel, EvaluationDependencies], Evaluator]
    result_kinds: tuple[Literal["boolean", "scored"], ...] = ("boolean",)


def _semantic(parameters: BaseModel, dependencies: EvaluationDependencies) -> Evaluator:
    if dependencies.provider is None:
        raise ConfigurationError("compose evaluation", "evaluation model is required")
    return SemanticGoalEvaluator(dependencies.provider, dependencies.redactor)


BUILTIN_EVALUATORS: Mapping[str, EvaluatorRegistration] = MappingProxyType(
    {
        "turn_count": EvaluatorRegistration(
            "turn_count", TurnCountParameters, lambda parameters, dependencies: TurnCountEvaluator()
        ),
        "journey_timeout": EvaluatorRegistration(
            "journey_timeout",
            JourneyTimeoutParameters,
            lambda parameters, dependencies: JourneyTimeoutEvaluator(),
        ),
        "repeated_information": EvaluatorRegistration(
            "repeated_information",
            RepeatedInformationParameters,
            lambda parameters, dependencies: RepeatedInformationEvaluator(),
        ),
        "semantic_goal_completion": EvaluatorRegistration(
            "semantic_goal_completion", SemanticGoalParameters, _semantic, ("boolean", "scored")
        ),
    }
)


@dataclass(frozen=True)
class PreparedEvaluator:
    criterion: Criterion
    registration: EvaluatorRegistration
    parameters: BaseModel

    def instantiate(self, dependencies: EvaluationDependencies) -> Evaluator:
        try:
            instance = self.registration.factory(
                self.parameters.model_copy(deep=True), dependencies
            )
            if not callable(getattr(instance, "evaluate", None)):
                raise ValueError("invalid evaluator factory")
            return instance
        except ConfigurationError:
            raise
        except Exception:
            raise ConfigurationError(
                "compose evaluation", "evaluator construction failed"
            ) from None


def prepare_evaluators(
    criteria: Sequence[Criterion],
    registry: Mapping[str, EvaluatorRegistration] | None = None,
) -> tuple[PreparedEvaluator, ...]:
    """Validate identities, result kinds and parameters before external activity."""
    registrations = dict(BUILTIN_EVALUATORS)
    if registry:
        if set(registrations).intersection(registry):
            raise ConfigurationError(
                "prepare evaluation", "built-in registrations cannot be replaced"
            )
        registrations.update(registry)
    prepared = []
    try:
        for criterion in criteria:
            registration = registrations[criterion.evaluator_type]
            if (
                registration.evaluator_type != criterion.evaluator_type
                or criterion.result_kind not in registration.result_kinds
            ):
                raise ValueError("invalid evaluator identity or result kind")
            parameters = registration.parameter_model.model_validate(criterion.parameters)
            # Persist the exact validated configuration, not factory-only defaults.
            normalized = Criterion.model_validate(
                {
                    **criterion.model_dump(mode="json"),
                    "parameters": parameters.model_dump(mode="json"),
                }
            )
            prepared.append(PreparedEvaluator(normalized, registration, parameters))
    except Exception:
        raise ConfigurationError(
            "prepare evaluation",
            "unregistered evaluator or invalid evaluator parameters/result kind",
        ) from None
    return tuple(prepared)

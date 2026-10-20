"""Extensions are trusted explicit registrations, not YAML imports or core edits."""

from pathlib import Path

import pytest
from pydantic import Field

from syntheticcx.config.loader import load_config
from syntheticcx.config.models import Criterion, StrictModel
from syntheticcx.config.settings import prepare_settings
from syntheticcx.errors import ConfigurationError
from syntheticcx.evaluation.registry import EvaluatorRegistration, prepare_evaluators


class Parameters(StrictModel):
    minimum: int = Field(default=1, ge=1, strict=True)


def criterion(**changes):
    return Criterion.model_validate(
        {
            "id": "extension",
            "evaluator_type": "custom_policy",
            "result_kind": "boolean",
            "evidence_policy": "required",
            **changes,
        }
    )


def registration(**changes):
    return EvaluatorRegistration(
        **{
            "evaluator_type": "custom_policy",
            "parameter_model": Parameters,
            "factory": lambda parameters, dependencies: pytest.fail(
                "factory called during preflight"
            ),
            **changes,
        }
    )


def test_parameter_defaults_are_prepared_without_mutating_the_source():
    original = criterion()
    result = prepare_evaluators([original], {"custom_policy": registration()})
    assert result[0].parameters.minimum == 1
    assert result[0].criterion.parameters == {"minimum": 1}
    assert original.parameters == {}


@pytest.mark.parametrize("parameters", [{"minimum": 0}, {"minimum": True}, {"extra": 1}])
def test_invalid_parameters_fail_without_constructing_evaluators(parameters):
    with pytest.raises(ConfigurationError, match="invalid evaluator"):
        prepare_evaluators([criterion(parameters=parameters)], {"custom_policy": registration()})


def test_unknown_evaluators_fail_before_credentials():
    config = load_config(Path("examples/telecom/scenario.yaml"))
    config = config.model_copy(
        update={"scenario": config.scenario.model_copy(update={"success_criteria": [criterion()]})}
    )
    with pytest.raises(ConfigurationError, match="unregistered evaluator"):
        prepare_settings(config, {})


def test_result_capabilities_and_registration_identity_are_enforced():
    for entry in [registration(evaluator_type="different"), registration(result_kinds=("scored",))]:
        with pytest.raises(ConfigurationError):
            prepare_evaluators([criterion()], {"custom_policy": entry})
    with pytest.raises(ConfigurationError, match="cannot be replaced"):
        prepare_evaluators([], {"turn_count": registration(evaluator_type="turn_count")})

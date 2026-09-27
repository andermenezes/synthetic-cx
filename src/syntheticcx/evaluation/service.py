"""Sequential, bounded evaluation with durable findings and classified retries."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from syntheticcx.domain.artifacts import EvaluationContext
from syntheticcx.domain.events import (
    EvaluationCompleted,
    EvaluationCompletedEvent,
    MessageObservedEvent,
)
from syntheticcx.domain.models import EvaluationResult, ModelInteraction, Scenario
from syntheticcx.domain.protocols import Evaluator, LLMProvider, WaitController
from syntheticcx.errors import ConfigurationError, EvaluationError, ModelError, PersistenceError
from syntheticcx.evaluation.deterministic import (
    JourneyTimeoutEvaluator,
    RepeatedInformationEvaluator,
    TurnCountEvaluator,
)
from syntheticcx.evaluation.semantic import SemanticGoalEvaluator
from syntheticcx.evaluation.validation import validate_result
from syntheticcx.privacy import Redactor
from syntheticcx.runtime.journal import RunJournal

_KINDS = frozenset(
    {"turn_count", "journey_timeout", "repeated_information", "semantic_goal_completion"}
)


@runtime_checkable
class _InteractionSource(Protocol):
    def take_interactions(self) -> tuple[ModelInteraction, ...]: ...


class EvaluationService:
    """Write each finding before starting work dependent on the final outcome.

    The runner still owns the journal and its lifetime; this service assumes its
    events are already sanitized and its writer stays open through evaluation.
    """

    def __init__(
        self,
        *,
        journal: RunJournal,
        wait: WaitController,
        evaluator_timeout_seconds: float,
        redactor: Redactor,
        provider: LLMProvider | None = None,
        evaluators: Mapping[str, Evaluator] | None = None,
    ) -> None:
        if not math.isfinite(evaluator_timeout_seconds) or evaluator_timeout_seconds <= 0:
            raise ValueError("evaluator timeout must be finite and positive")
        registry: dict[str, Evaluator] = {
            "turn_count": TurnCountEvaluator(),
            "journey_timeout": JourneyTimeoutEvaluator(),
            "repeated_information": RepeatedInformationEvaluator(),
        }
        if provider is not None:
            registry["semantic_goal_completion"] = SemanticGoalEvaluator(provider, redactor)
        if evaluators is not None:
            if set(evaluators) - _KINDS:
                raise ConfigurationError("compose evaluation", "unsupported evaluator type")
            registry.update(evaluators)
        self._evaluators = registry
        self._journal = journal
        self._wait = wait
        self._timeout = evaluator_timeout_seconds
        self._redactor = redactor

    async def evaluate(self, scenario: Scenario) -> tuple[EvaluationResult, ...]:
        """Process each configured criterion once, using only acknowledged trace events."""
        criteria = scenario.success_criteria
        configured = {item.id for item in criteria}
        if any(
            isinstance(event, EvaluationCompletedEvent)
            and event.data.result.criterion_id in configured
            for event in self._journal.committed_events
        ):
            raise EvaluationError("evaluate run", "evaluation already started")
        if any(item.evaluator_type not in self._evaluators for item in criteria):
            raise ConfigurationError("evaluate run", "evaluation model is required")

        results: list[EvaluationResult] = []
        for criterion in criteria:
            evaluator = self._evaluators[criterion.evaluator_type]
            attempts = 0
            while True:
                attempts += 1
                context = EvaluationContext(
                    criterion=criterion,
                    scenario=scenario,
                    sanitized_trace=list(self._journal.committed_events),
                )
                failure: Exception | None = None
                candidate: EvaluationResult | None = None
                try:
                    candidate = await self._wait.run(evaluator.evaluate(context), self._timeout)
                except asyncio.CancelledError:
                    raise
                except PersistenceError:
                    raise
                except Exception as exc:
                    failure = exc
                finally:
                    # Provider calls can have succeeded even when verdict validation
                    # fails. Their owned timing/usage evidence must precede the result.
                    await self._commit_interactions_despite_cancellation(evaluator, context)

                if failure is not None and self._transient(failure) and not criterion.required:
                    if attempts == 1:
                        continue
                if failure is not None:
                    candidate = EvaluationResult(
                        evaluator_id=criterion.evaluator_type,
                        criterion_id=criterion.id,
                        required=criterion.required,
                        status="error",
                        explanation="Evaluator operation failed or returned invalid evidence.",
                        metadata={"attempts": attempts, "transient": self._transient(failure)},
                    )
                elif candidate is None:
                    candidate = EvaluationResult(
                        evaluator_id=criterion.evaluator_type,
                        criterion_id=criterion.id,
                        required=criterion.required,
                        status="unavailable",
                        explanation="Evaluator produced no result for this criterion.",
                        metadata={"attempts": attempts},
                    )
                else:
                    try:
                        validate_result(candidate, criterion, list(self._journal.committed_events))
                    except EvaluationError:
                        candidate = EvaluationResult(
                            evaluator_id=criterion.evaluator_type,
                            criterion_id=criterion.id,
                            required=criterion.required,
                            status="error",
                            explanation="Evaluator result or evidence is invalid.",
                            metadata={"attempts": attempts},
                        )
                committed = await self._journal.append(
                    EvaluationCompleted(
                        result=candidate,
                        criterion=self._redactor.redact_criterion(criterion, scenario),
                    )
                )
                if not isinstance(committed, EvaluationCompletedEvent):
                    raise PersistenceError(
                        "append journal", "evaluation evidence was not committed"
                    )
                results.append(committed.data.result)
                break

        return tuple(results)

    async def _commit_interactions_despite_cancellation(
        self, evaluator: Evaluator, context: EvaluationContext
    ) -> None:
        """Finish an acknowledged interaction append before cancellation propagates."""
        task = asyncio.ensure_future(self._commit_interactions(evaluator, context))
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        await task
        if cancelled:
            raise asyncio.CancelledError

    async def _commit_interactions(self, evaluator: Evaluator, context: EvaluationContext) -> None:
        if not isinstance(evaluator, _InteractionSource):
            return
        try:
            interactions = evaluator.take_interactions()
        except Exception:
            raise EvaluationError("evaluate run", "model interaction evidence failed") from None
        input_ids = [
            event.id for event in context.sanitized_trace if isinstance(event, MessageObservedEvent)
        ]
        for interaction in interactions:
            # Journal-side linkage is authoritative, regardless of provisional
            # references supplied by the evaluator. No output event exists yet.
            safe_interaction = interaction.model_copy(
                update={
                    "input_event_ids": input_ids,
                    "output_event_ids": [],
                    "capture_omissions": list(
                        dict.fromkeys(
                            [
                                *interaction.capture_omissions,
                                "evaluation_result_not_yet_committed",
                            ]
                        )
                    ),
                },
                deep=True,
            )
            await self._journal.append(safe_interaction)

    @staticmethod
    def _transient(exc: Exception) -> bool:
        """Only a timeout or a provider-classified temporary failure is retryable."""
        return isinstance(exc, TimeoutError) or isinstance(exc, ModelError) and exc.retryable

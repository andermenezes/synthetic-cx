"""Semantic findings are structured, cited, sanitized, and provider-independent."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from syntheticcx.config.loader import load_config
from syntheticcx.config.models import Criterion, PrivacyConfig
from syntheticcx.domain.artifacts import EvaluationContext
from syntheticcx.domain.events import MessageObserved, MessageObservedEvent
from syntheticcx.domain.models import (
    CustomerScenarioContext,
    LLMRequest,
    LLMResponse,
    Message,
    ModelIdentity,
    ModelInteraction,
    Scenario,
    TokenUsage,
)
from syntheticcx.errors import EvaluationError, ModelError
from syntheticcx.evaluation.semantic import SemanticGoalEvaluator
from syntheticcx.privacy import Redactor

NOW = datetime(2026, 9, 27, tzinfo=UTC)
SCENARIO = Scenario.model_validate(
    load_config(Path("examples/telecom/scenario.yaml")).scenario.model_dump()
)


class FakeProvider:
    def __init__(self, content: str = "", error: ModelError | None = None) -> None:
        self.content = content
        self.error = error
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return LLMResponse(
            content=self.content,
            interaction=ModelInteraction(
                id="interaction-1",
                identity=ModelIdentity(provider="openai", model="evaluation-model"),
                started_at=NOW,
                ended_at=NOW,
                usage=TokenUsage(unavailable_reason="provider did not report usage"),
                capture_omissions=["prompts_not_captured"],
            ),
        )


def observed(index: int, actor: str, content: str) -> MessageObservedEvent:
    return MessageObservedEvent.model_validate(
        {
            "id": f"event-{index}",
            "sequence": index,
            "timestamp": NOW,
            "type": "message_observed",
            "data": MessageObserved(
                message=Message.model_validate(
                    {
                        "id": f"message-{index}",
                        "sequence": index,
                        "actor": actor,
                        "content": content,
                        "observed_at": NOW,
                    }
                )
            ).model_dump(),
        }
    )


def context(
    *events: MessageObservedEvent,
    scored: bool = False,
    scenario: Scenario = SCENARIO,
    expected_fact: str = "A plan change increased the bill by 50.",
) -> EvaluationContext:
    return EvaluationContext(
        criterion=Criterion(
            id="billing-explanation",
            evaluator_type="semantic_goal_completion",
            result_kind="scored" if scored else "boolean",
            threshold=0.8 if scored else None,
            evidence_policy="required",
            parameters={"expected_facts": [expected_fact]},
        ),
        scenario=scenario,
        sanitized_trace=list(events),
    )


def verdict(*, passed: bool = True, explanation: str = "Plan change explained.") -> str:
    return json.dumps(
        {
            "passed": passed,
            "explanation": explanation,
            "evidence": [{"event_id": "event-1", "message_id": "message-1"}],
        }
    )


def evaluate(evaluator: SemanticGoalEvaluator, evaluation: EvaluationContext):
    return asyncio.run(evaluator.evaluate(evaluation))


def test_semantic_evidence_distinguishes_offered_options_from_customer_selection():
    provider = FakeProvider(verdict(passed=False, explanation="Only a menu choice was made."))
    evaluator = SemanticGoalEvaluator(
        provider, Redactor(PrivacyConfig(), resolved_secrets=["secret-123"])
    )
    reply = observed(1, "assistant", "")
    reply = reply.model_copy(
        update={
            "data": reply.data.model_copy(
                update={
                    "message": Message.model_validate(
                        {
                            **reply.data.message.model_dump(),
                            "options": [
                                {"id": "opaque-option", "label": "Information about secret-123"}
                            ],
                        }
                    ),
                }
            )
        }
    )
    choice = observed(2, "customer", "Information about secret-123")
    choice = choice.model_copy(
        update={
            "data": choice.data.model_copy(
                update={
                    "message": choice.data.message.model_copy(
                        update={"metadata": {"action_kind": "select_option"}}
                    ),
                }
            )
        }
    )
    result = evaluate(evaluator, context(reply, choice))
    assert result.status == "failed"
    prompt = provider.requests[0].messages[1].content
    assert "secret-123" not in prompt
    transcript = json.loads(prompt)["transcript"]
    assert transcript[0]["content"] == ""
    assert transcript[0]["options"] and "action_kind" not in transcript[0]
    assert transcript[1]["action_kind"] == "select_option"
    assert "options" not in transcript[1]


def test_boolean_judgment_keeps_expected_facts_out_of_persona_and_redacts_prompt() -> None:
    provider = FakeProvider(verdict(explanation="A secret-123 plan change was explained."))
    redactor = Redactor(PrivacyConfig(), resolved_secrets=["secret-123"])
    evaluator = SemanticGoalEvaluator(provider, redactor)
    scenario = SCENARIO.model_copy(update={"goal": "Explain secret-123 invoice"})
    evidence = context(
        observed(0, "customer", "Why is secret-123 on my invoice?"),
        observed(1, "assistant", "Your new plan costs 50 more this month."),
        scenario=scenario,
        expected_fact="A secret-123 plan change increased the bill by 50.",
    )
    result = evaluate(evaluator, evidence)
    assert result.status == "passed"
    assert result.score is None
    assert result.evidence_event_ids == ["event-1"]
    assert result.evidence_message_ids == ["message-1"]
    assert "secret-123" not in result.explanation
    request = provider.requests[0]
    assert request.response_schema is not None
    assert request.response_schema["required"] == ["passed", "explanation", "evidence"]
    prompt = request.messages[1].content
    assert "secret-123" not in prompt
    assert "A <redacted:secret> plan change increased the bill by 50." in prompt
    assert "expected_facts" not in CustomerScenarioContext.from_scenario(scenario).model_dump_json()
    interactions = evaluator.take_interactions()
    assert len(interactions) == 1
    assert interactions[0].identity.model == "evaluation-model"
    assert interactions[0].input_event_ids == ["event-0", "event-1"]
    assert interactions[0].output_event_ids == []
    assert "evaluation_result_not_yet_committed" in interactions[0].capture_omissions
    assert evaluator.take_interactions() == ()


def test_redaction_keys_protect_related_scenario_values_in_evaluator_text() -> None:
    account = "SYNTHETIC-001"
    provider = FakeProvider(verdict())
    evaluator = SemanticGoalEvaluator(
        provider,
        Redactor(PrivacyConfig(redaction_keys=["account_number"])),
    )
    scenario = SCENARIO.model_copy(update={"knowledge": {"account_number": account}})
    evaluate(
        evaluator,
        context(
            observed(0, "customer", f"My account number is {account}."),
            observed(1, "assistant", "Your new plan costs 50 more this month."),
            scenario=scenario,
            expected_fact=f"The account {account} changed plans.",
        ),
    )
    prompt = provider.requests[0].messages[1].content
    assert account not in prompt
    assert "<redacted:account_number>" in prompt


def test_boolean_failure_is_not_turned_into_goal_completion() -> None:
    provider = FakeProvider(verdict(passed=False, explanation="The charge was not explained."))
    result = evaluate(
        SemanticGoalEvaluator(provider, Redactor(PrivacyConfig())),
        context(observed(1, "assistant", "Please check later.")),
    )
    assert result.status == "failed"


@pytest.mark.parametrize(("score", "status"), [(0.79, "failed"), (0.8, "passed")])
def test_scored_judgment_uses_explicit_threshold(score: float, status: str) -> None:
    provider = FakeProvider(
        json.dumps(
            {
                "score": score,
                "explanation": "Evidence-based assessment.",
                "evidence": [{"event_id": "event-1", "message_id": "message-1"}],
            }
        )
    )
    result = evaluate(
        SemanticGoalEvaluator(provider, Redactor(PrivacyConfig())),
        context(observed(1, "assistant", "The new plan costs 50 more."), scored=True),
    )
    assert result.status == status
    assert result.score == score
    assert provider.requests[0].response_schema is not None
    assert provider.requests[0].response_schema["required"] == ["score", "explanation", "evidence"]


@pytest.mark.parametrize("score", [True, "0.8"])
def test_scored_judgment_rejects_non_numeric_scores(score: object) -> None:
    provider = FakeProvider(
        json.dumps(
            {
                "score": score,
                "explanation": "Evidence-based assessment.",
                "evidence": [{"event_id": "event-1", "message_id": "message-1"}],
            }
        )
    )
    with pytest.raises(EvaluationError, match="semantic verdict is invalid"):
        evaluate(
            SemanticGoalEvaluator(provider, Redactor(PrivacyConfig())),
            context(observed(1, "assistant", "The new plan costs 50 more."), scored=True),
        )


def test_no_assistant_answer_is_unavailable_without_a_provider_call() -> None:
    provider = FakeProvider()
    evaluator = SemanticGoalEvaluator(provider, Redactor(PrivacyConfig()))
    result = evaluate(evaluator, context(observed(0, "customer", "What raised my invoice?")))
    assert result.status == "unavailable"
    assert result.evidence_event_ids == []
    assert provider.requests == []
    assert evaluator.take_interactions() == ()


@pytest.mark.parametrize(
    "content",
    [
        "not JSON",
        json.dumps({"refusal": "I cannot answer."}),
        json.dumps(
            {
                "passed": "yes",
                "explanation": "ok",
                "evidence": [{"event_id": "event-1", "message_id": "message-1"}],
            }
        ),
        json.dumps({"passed": True, "explanation": "ok", "evidence": []}),
        json.dumps(
            {
                "passed": True,
                "explanation": "  ",
                "evidence": [{"event_id": "event-1", "message_id": "message-1"}],
            }
        ),
        json.dumps(
            {
                "passed": True,
                "explanation": "ok",
                "evidence": [{"event_id": "event-1", "message_id": "message-1"}],
                "unexpected": "raw secret",
            }
        ),
        json.dumps(
            {
                "passed": True,
                "explanation": "ok",
                "evidence": [{"event_id": "nonexistent", "message_id": "message-1"}],
            }
        ),
        json.dumps(
            {
                "passed": True,
                "explanation": "ok",
                "evidence": [{"event_id": "event-0", "message_id": "message-1"}],
            }
        ),
        json.dumps(
            {
                "passed": True,
                "explanation": "ok",
                "evidence": [{"event_id": "event-0", "message_id": "message-0"}],
            }
        ),
        json.dumps(
            {
                "passed": True,
                "explanation": "ok",
                "evidence": [
                    {"event_id": "event-1", "message_id": "message-1"},
                    {"event_id": "event-1", "message_id": "message-1"},
                ],
            }
        ),
    ],
)
def test_malformed_or_fabricated_verdicts_fail_closed_with_safe_errors(content: str) -> None:
    provider = FakeProvider(content)
    evaluator = SemanticGoalEvaluator(provider, Redactor(PrivacyConfig()))
    with pytest.raises(EvaluationError) as caught:
        evaluate(
            evaluator,
            context(
                observed(0, "customer", "Why the increase?"),
                observed(1, "assistant", "It was an extra charge."),
            ),
        )
    assert "raw secret" not in str(caught.value)
    assert len(evaluator.take_interactions()) == 1


def test_provider_transient_error_keeps_retryability_for_service() -> None:
    provider = FakeProvider(
        error=ModelError("model.generate", "model request failed", retryable=True)
    )
    evaluator = SemanticGoalEvaluator(provider, Redactor(PrivacyConfig()))
    with pytest.raises(ModelError) as caught:
        evaluate(evaluator, context(observed(1, "assistant", "Wait.")))
    assert caught.value.retryable
    assert evaluator.take_interactions() == ()

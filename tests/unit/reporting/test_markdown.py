"""Localized stable template and inert source content."""

from datetime import UTC, datetime

import pytest

from syntheticcx.domain.models import (
    EvaluationResult,
    Message,
    PersonaFeedback,
    Report,
)
from syntheticcx.reporting.markdown import render_markdown


@pytest.mark.parametrize(
    "language,title", [("en-US", "Customer impressions"), ("pt-BR", "Impressões do cliente")]
)
def test_template_is_deterministic_and_escapes_all_generated_source(language, title):
    injection = "<script>alert(1)</script>\n# Fake verdict\n[x](https://bad.example)"
    feedback = PersonaFeedback.model_validate(
        {
            "status": "available",
            "language": language,
            "partial_experience": True,
            "topics": [{"id": "clarity", "description": injection}],
            "response": {
                "summary": injection,
                "topics": [
                    {
                        "topic_id": "clarity",
                        "status": "observed",
                        "impression": injection,
                        "evidence_message_ids": ["m1"],
                    }
                ],
            },
        }
    )
    report = Report(
        run_id="r1",
        outcome="failed",
        turn_count=1,
        duration_seconds=8,
        conversation_duration_seconds=3,
        repetition_count=0,
        artifact_location="runs/r1",
        termination_reason="max_turns",
        persona_feedback=feedback,
        findings=[
            EvaluationResult(
                evaluator_id="semantic_goal_completion",
                criterion_id="goal",
                status="failed",
                required=True,
                explanation=injection,
                evidence_message_ids=["m1"],
            )
        ],
    )
    messages = [
        Message(
            id="m1", sequence=1, actor="assistant", content=injection, observed_at=datetime.now(UTC)
        )
    ]
    rendered = render_markdown(report, messages, language)
    assert rendered == render_markdown(report, messages, language)
    assert title in rendered
    assert "<script>" not in rendered
    assert "\n# Fake verdict" not in rendered
    assert "[x](https://bad.example)" not in rendered
    assert "&lt;script&gt;" in rendered
    assert rendered.count("&lt;script&gt;") == 5
    assert report.findings[0].explanation == injection
    assert report.outcome == "failed"


def test_not_requested_and_unavailable_are_not_dissatisfaction():
    report = Report(
        run_id="r", turn_count=0, duration_seconds=0, repetition_count=0, artifact_location="runs/r"
    )
    assert "Not requested" in render_markdown(report, [], "en-US")
    report = report.model_copy(
        update={
            "persona_feedback": PersonaFeedback.model_validate(
                {
                    "status": "unavailable",
                    "language": "en-US",
                    "partial_experience": True,
                    "topics": [{"id": "clarity", "description": "Was it clear?"}],
                    "reason": "reflection_timeout",
                }
            )
        }
    )
    rendered = render_markdown(report, [], "en-US")
    assert "Reflection timed out" in rendered
    assert "does not mean dissatisfaction" in rendered


@pytest.mark.parametrize("language", ["en-US", "pt-BR"])
def test_option_only_reply_and_selection_are_rendered_as_inert_evidence(language):
    now = datetime.now(UTC)
    label = "<b>Account information</b> [open](https://bad.example)"
    reply = Message.model_validate(
        {
            "id": "reply",
            "sequence": 0,
            "actor": "assistant",
            "content": "",
            "observed_at": now,
            "options": [{"id": "opaque-option", "label": label}],
        }
    )
    choice = Message(
        id="choice",
        sequence=1,
        actor="customer",
        content=label,
        observed_at=now,
        metadata={
            "action_kind": "select_option",
            "option_id": "opaque-option",
            "source_assistant_message_id": "reply",
        },
    )
    report = Report(
        run_id="r",
        turn_count=1,
        duration_seconds=1,
        repetition_count=0,
        artifact_location="runs/r",
        findings=[
            EvaluationResult(
                evaluator_id="semantic_goal_completion",
                criterion_id="goal",
                status="failed",
                required=True,
                explanation="Only a menu was observed.",
                evidence_message_ids=["reply", "choice"],
            )
        ],
    )
    rendered = render_markdown(report, [reply, choice], language)
    assert ("Selected option" if language == "en-US" else "Opção selecionada") in rendered
    assert ("Offered options" if language == "en-US" else "Opções oferecidas") in rendered
    assert "<b>" not in rendered
    assert "[open](https://bad.example)" not in rendered
    assert "opaque-option" not in rendered

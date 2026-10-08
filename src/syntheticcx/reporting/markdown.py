"""Deterministic localized Markdown; generated content never controls report structure."""

from __future__ import annotations

import html
import re

from syntheticcx.domain.models import Message, Report
from syntheticcx.language import Language


def _text(value: object) -> str:
    text = str(value)
    text = "".join(c for c in text if c in "\n\t" or (ord(c) >= 32 and c != "\x7f"))
    text = html.escape(text, quote=True)
    return re.sub(r"([\\`*_{}\[\]()#+.!|>~-])", r"\\\1", text)


def _quote(value: str) -> str:
    return "\n".join("> " + line for line in _text(value).splitlines()) or ">"


def render_markdown(
    report: Report,
    messages: list[Message],
    language: Language,
) -> str:
    """Render source text as inert quotations, retaining original wording/language."""
    pt = language == "pt-BR"

    def label(en: str, br: str) -> str:
        return br if pt else en

    statuses = {
        "passed": label("Passed", "Aprovado"),
        "failed": label("Failed", "Reprovado"),
        "error": label("Error", "Erro"),
        "cancelled": label("Cancelled", "Cancelado"),
        "unavailable": label("Unavailable", "Indisponível"),
        "captured": label("Captured", "Registrado"),
        "omitted": label("Omitted", "Omitido"),
    }
    stops = {
        "goal_believed_complete": label(
            "Customer believed the goal was complete", "Cliente acreditou ter concluído o objetivo"
        ),
        "abandonment": label("Customer abandoned", "Cliente abandonou"),
        "max_turns": label("Turn limit", "Limite de turnos"),
        "journey_timeout": label("Journey timeout", "Tempo da jornada esgotado"),
        "cancelled": statuses["cancelled"],
        "channel_error": label("Channel error", "Erro de canal"),
        "model_error": label("Model error", "Erro de modelo"),
        "persistence_error": label("Persistence error", "Erro de persistência"),
    }
    unavailable = label("Unavailable", "Indisponível")
    lines = [
        "# " + label("SyntheticCX execution report", "Relatório de execução SyntheticCX"),
        "",
        "## " + label("Context", "Contexto"),
        "",
        "- " + label("Run", "Execução") + ": " + _text(report.run_id),
        "- " + label("Report language", "Idioma do relatório") + ": " + language,
        "- "
        + label(
            "Original source text is preserved below.",
            "O texto original das fontes é preservado abaixo.",
        ),
        "",
        "## " + label("Execution summary", "Resumo da execução"),
        "",
        "- "
        + label("Formal outcome", "Resultado formal")
        + ": "
        + statuses.get(report.outcome or "", unavailable),
        "- "
        + label("Conversation stop", "Término da conversa")
        + ": "
        + stops.get(report.termination_reason or "", unavailable),
        "- " + label("Customer turns", "Turnos do cliente") + f": {report.turn_count}",
        "- "
        + label("Repeated information requests", "Pedidos repetidos de informação")
        + f": {report.repetition_count}",
        "- "
        + label("Conversation duration (seconds)", "Duração da conversa (segundos)")
        + ": "
        + (
            f"{report.conversation_duration_seconds:.2f}"
            if report.conversation_duration_seconds is not None
            else unavailable
        ),
        "- "
        + label("Recorded trace duration (seconds)", "Duração do registro (segundos)")
        + f": {report.duration_seconds:.2f}",
        "",
        "## " + label("Customer impressions", "Impressões do cliente"),
        "",
        label(
            "Subjective first-person feedback from the same simulated customer. "
            "Not a formal usability measurement and not used for pass/fail.",
            "Feedback subjetivo em primeira pessoa do mesmo cliente simulado. "
            "Não é uma medição formal de usabilidade e não determina aprovação ou reprovação.",
        ),
        "",
    ]
    feedback = report.persona_feedback
    if feedback is None:
        lines.append(
            label(
                "Not requested: no observation topics configured.",
                "Não solicitado: nenhum tópico de observação configurado.",
            )
        )
    else:
        if feedback.partial_experience:
            lines += [
                label(
                    "Partial experience: the conversation stopped before "
                    "customer-believed completion.",
                    "Experiência parcial: a conversa terminou antes da "
                    "conclusão percebida pelo cliente.",
                ),
                "",
            ]
        if feedback.experience_truncated:
            lines += [
                label(
                    "The supplied evidence excerpt was bounded/truncated.",
                    "O trecho de evidências fornecido foi limitado/truncado.",
                ),
                "",
            ]
        if feedback.response:
            lines += [_quote(feedback.response.summary), ""]
            by_id = {topic.topic_id: topic for topic in feedback.response.topics}
            for topic in feedback.topics:
                result = by_id[topic.id]
                lines += [
                    "### " + _text(topic.id),
                    "",
                    _quote(topic.description),
                    "",
                    label("Observed", "Observado")
                    if result.status == "observed"
                    else label("Not observed", "Não observado"),
                    "",
                    _quote(result.impression),
                    "",
                    label("Message references", "Referências de mensagens")
                    + ": "
                    + ", ".join(_text(ref) for ref in result.evidence_message_ids),
                    "",
                ]
        else:
            reasons = {
                "cancelled": label("Cancelled", "Cancelado"),
                "no_experience": label("No recorded experience", "Sem experiência registrada"),
                "unsupported": label(
                    "Adapter does not support reflection", "Adaptador não oferece reflexão"
                ),
                "persona_unavailable": label(
                    "Original persona session unavailable",
                    "Sessão da persona original indisponível",
                ),
                "reflection_failed": label("Reflection failed", "Falha na reflexão"),
                "reflection_timeout": label("Reflection timed out", "Tempo da reflexão esgotado"),
                "invalid_feedback": label(
                    "Feedback could not be validated", "Feedback não pôde ser validado"
                ),
            }
            lines += [unavailable + ": " + reasons.get(feedback.reason or "", unavailable), ""]
            for topic in feedback.topics:
                lines += [
                    "### " + _text(topic.id),
                    "",
                    _quote(topic.description),
                    "",
                    unavailable,
                    "",
                ]
    lines += ["", "## " + label("Formal test conclusions", "Conclusões formais do teste"), ""]
    for finding in report.findings:
        lines += [
            "### " + _text(finding.criterion_id),
            "",
            "- " + label("Status", "Estado") + ": " + statuses[finding.status],
            "- "
            + label("Required", "Obrigatório")
            + ": "
            + (label("Yes", "Sim") if finding.required else label("No", "Não")),
            "- "
            + label("Score", "Pontuação")
            + ": "
            + (str(finding.score) if finding.score is not None else unavailable),
            "- "
            + label("Evidence event references", "Referências de eventos de evidência")
            + ": "
            + ", ".join(_text(ref) for ref in finding.evidence_event_ids),
            "- "
            + label("Evidence message references", "Referências de mensagens de evidência")
            + ": "
            + ", ".join(_text(ref) for ref in finding.evidence_message_ids),
            "",
            label(
                "Evaluator explanation — original source:",
                "Explicação do avaliador — fonte original:",
            ),
            "",
            _quote(finding.explanation),
            "",
        ]
    lines += [
        "## "
        + label(
            "Conversation evidence — original source", "Evidências da conversa — fonte original"
        ),
        "",
    ]
    referenced = {ref for finding in report.findings for ref in finding.evidence_message_ids}
    if feedback and feedback.response:
        referenced.update(
            ref for topic in feedback.response.topics for ref in topic.evidence_message_ids
        )
    for message in messages:
        if message.id in referenced:
            lines += [
                "### "
                + _text(message.id)
                + " — "
                + label(
                    "Customer" if message.actor == "customer" else "Assistant",
                    "Cliente" if message.actor == "customer" else "Assistente",
                ),
                "",
                *(
                    [label("Selected option", "Opção selecionada") + ":", ""]
                    if message.metadata.get("action_kind") == "select_option"
                    else []
                ),
                _quote(message.content),
                "",
            ]
            if message.options:
                lines += [label("Offered options", "Opções oferecidas") + ":", ""]
                for option in message.options:
                    lines += [_quote(option.label), ""]
    lines += [
        "## " + label("Limitations and artifacts", "Limitações e artefatos"),
        "",
        label(
            "Unavailable feedback does not mean dissatisfaction. "
            "Customer-believed completion is not formal success. "
            "Source explanations and topic wording may use another language.",
            "Feedback indisponível não significa insatisfação. "
            "Conclusão percebida pelo cliente não é sucesso formal. "
            "Explicações das fontes e tópicos podem estar em outro idioma.",
        ),
        "",
    ]
    for error in report.errors:
        reason = f" [reason: {_text(error.reason_code)}]" if error.reason_code else ""
        lines += ["- " + _text(error.operation) + ": " + _text(error.message) + reason]
    for artifact in report.artifacts:
        lines += [
            "- "
            + _text(artifact.path or artifact.media_type)
            + ": "
            + statuses[artifact.status]
            + (" (" + _text(artifact.reason) + ")" if artifact.reason else "")
        ]
    for omission in report.omissions:
        lines += ["- " + label("Omission", "Omissão") + ": " + _text(omission)]
    return "\n".join(lines).rstrip() + "\n"

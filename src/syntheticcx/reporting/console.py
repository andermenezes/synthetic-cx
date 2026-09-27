"""Console presentation from the same validated report used for JSON artifacts."""

from __future__ import annotations

from syntheticcx.domain.models import Report


def render_report(report: Report, *, report_omitted: bool = False) -> str:
    """Render only fields of an acknowledged report projection."""
    goal = report.goal_result.status if report.goal_result is not None else "unavailable"
    errors = ", ".join(
        f"{error.category} ({error.operation}): {error.message}" for error in report.errors
    )
    lines = [
        f"Run ID: {report.run_id}",
        f"Outcome: {report.outcome}",
        f"Termination: {report.termination_reason}",
        f"Goal: {goal}",
        f"Turns: {report.turn_count}",
        f"Duration (seconds): {report.duration_seconds}",
        f"Repetition findings: {report.repetition_count}",
        f"Errors: {errors or 'none'}",
        f"Artifacts: {report.artifact_location}",
    ]
    if report_omitted:
        lines.append("report.json: omitted by storage configuration")
    return "\n".join(lines)


def render_partial(run_id: str | None, artifact_location: str | None, reason: str) -> str:
    """Do not imply a completed report when finalization is unavailable."""
    lines = [f"Run: {run_id or 'not created'}", f"Status: {reason}"]
    if artifact_location is not None:
        lines.append(f"Partial evidence: {artifact_location}")
    return "\n".join(lines)

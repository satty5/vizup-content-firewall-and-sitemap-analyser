"""
Redline Report Generator
Produces a structured, human-readable issue report.

Format:
  Line 12 — COMPETITOR_PROMOTION — Blocker
  Mentions Datadog as a recommended option in a brand-led article.
"""

from __future__ import annotations

from models.review_result import FirewallVerdict, ReviewIssue
from config.taxonomy import Severity, PublishDecision


_SEVERITY_LABELS = {
    Severity.BLOCKER: "BLOCKER",
    Severity.MAJOR: "MAJOR",
    Severity.MINOR: "MINOR",
    Severity.STYLE: "STYLE",
}

_SEVERITY_ORDER = {
    Severity.BLOCKER: 0,
    Severity.MAJOR: 1,
    Severity.MINOR: 2,
    Severity.STYLE: 3,
}

_DECISION_LABELS = {
    PublishDecision.PASS: "PASS — Content is publish-ready.",
    PublishDecision.PASS_WITH_REVISIONS: "PASS WITH REVISIONS — Fixable issues found. Revise before publish.",
    PublishDecision.FAIL: "FAIL — Content has blocking issues. Do not publish.",
}


def generate_redline_report(verdict: FirewallVerdict) -> str:
    """Generate a detailed redline report from the firewall verdict."""

    lines: list[str] = []

    lines.append("=" * 72)
    lines.append("  CONTENT QUALITY FIREWALL — REDLINE REPORT")
    lines.append("=" * 72)
    lines.append("")

    lines.append(f"  DECISION:  {_DECISION_LABELS.get(verdict.decision, verdict.decision.value)}")
    lines.append(f"  SCORE:     {verdict.overall_score}/100")
    lines.append(f"  ISSUES:    {verdict.total_issues} total")
    lines.append(f"             {verdict.blocker_count} blockers | {verdict.major_count} major | {verdict.minor_count} minor | {verdict.style_count} style")
    lines.append("")
    lines.append("-" * 72)

    sorted_issues = sorted(
        verdict.all_issues,
        key=lambda i: (_SEVERITY_ORDER.get(i.severity, 9), i.line_number),
    )

    if verdict.blockers:
        lines.append("")
        lines.append("  BLOCKERS (must fix before publish)")
        lines.append("  " + "-" * 40)
        for issue in verdict.blockers:
            lines.append(_format_issue(issue))

    majors = [i for i in sorted_issues if i.severity == Severity.MAJOR]
    if majors:
        lines.append("")
        lines.append("  MAJOR ISSUES (high-risk quality problems)")
        lines.append("  " + "-" * 40)
        for issue in majors:
            lines.append(_format_issue(issue))

    minors = [i for i in sorted_issues if i.severity == Severity.MINOR]
    if minors:
        lines.append("")
        lines.append("  MINOR ISSUES (improve quality)")
        lines.append("  " + "-" * 40)
        for issue in minors:
            lines.append(_format_issue(issue))

    style_issues = [i for i in sorted_issues if i.severity == Severity.STYLE]
    if style_issues:
        lines.append("")
        lines.append("  STYLE (optional polish)")
        lines.append("  " + "-" * 40)
        for issue in style_issues:
            lines.append(_format_issue(issue))

    lines.append("")
    lines.append("-" * 72)
    lines.append("  SCORE BREAKDOWN")
    lines.append("-" * 72)
    for key, val in sorted(verdict.score_breakdown.items()):
        bar = _score_bar(val)
        lines.append(f"  {key:.<45} {val:.2f}  {bar}")

    lines.append("")
    lines.append("-" * 72)
    lines.append("  REVIEWER SUMMARIES")
    lines.append("-" * 72)
    for ro in verdict.reviewer_outputs:
        if ro.summary:
            lines.append(f"\n  [{ro.reviewer_name}]")
            for line in ro.summary.split("\n"):
                lines.append(f"    {line}")

    lines.append("")
    lines.append("=" * 72)
    lines.append("")

    return "\n".join(lines)


def _format_issue(issue: ReviewIssue) -> str:
    parts = []
    loc = f"Line {issue.line_number}" if issue.line_number else issue.section or "General"
    severity_label = _SEVERITY_LABELS.get(issue.severity, issue.severity.value)

    parts.append(f"\n  {loc} — {issue.category.value} — {severity_label}")
    if issue.reviewer:
        parts.append(f"  Source: {issue.reviewer}")
    if issue.line_text:
        truncated = issue.line_text[:120]
        parts.append(f'  Text: "{truncated}"')
    parts.append(f"  Issue: {issue.explanation}")
    if issue.suggested_fix:
        parts.append(f"  Fix: {issue.suggested_fix}")
    parts.append(f"  Confidence: {issue.confidence:.0%}")

    return "\n".join(parts)


def _score_bar(value: float, width: int = 20) -> str:
    if value > 1:
        return ""
    filled = int(value * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def generate_redline_json(verdict: FirewallVerdict) -> dict:
    """Generate a JSON-serializable redline report."""
    return {
        "decision": verdict.decision.value,
        "overall_score": verdict.overall_score,
        "issue_counts": {
            "total": verdict.total_issues,
            "blocker": verdict.blocker_count,
            "major": verdict.major_count,
            "minor": verdict.minor_count,
            "style": verdict.style_count,
        },
        "issues": [
            {
                "line_number": i.line_number,
                "line_text": i.line_text[:200],
                "section": i.section,
                "category": i.category.value,
                "severity": i.severity.value,
                "explanation": i.explanation,
                "suggested_fix": i.suggested_fix,
                "confidence": i.confidence,
                "reviewer": i.reviewer,
            }
            for i in sorted(
                verdict.all_issues,
                key=lambda x: (_SEVERITY_ORDER.get(x.severity, 9), x.line_number),
            )
        ],
        "score_breakdown": verdict.score_breakdown,
        "reviewer_summaries": {
            ro.reviewer_name: ro.summary
            for ro in verdict.reviewer_outputs
            if ro.summary
        },
    }

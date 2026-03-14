"""
Scoring engine — aggregates reviewer outputs into a final verdict.
Not just "quality score: 72." — produces actionable publish decisions.
"""

from __future__ import annotations

from models.review_result import ReviewerOutput, ReviewIssue, FirewallVerdict
from config.taxonomy import Severity, PublishDecision, SEVERITY_WEIGHT
from config.settings import BLOCKER_THRESHOLD, MAJOR_THRESHOLD


def compute_verdict(
    reviewer_outputs: list[ReviewerOutput],
) -> FirewallVerdict:
    """Aggregate all reviewer outputs into a single firewall verdict."""

    all_issues: list[ReviewIssue] = []
    for ro in reviewer_outputs:
        all_issues.extend(ro.issues)

    blocker_count = sum(1 for i in all_issues if i.severity == Severity.BLOCKER)
    major_count = sum(1 for i in all_issues if i.severity == Severity.MAJOR)
    minor_count = sum(1 for i in all_issues if i.severity == Severity.MINOR)
    style_count = sum(1 for i in all_issues if i.severity == Severity.STYLE)

    if blocker_count >= BLOCKER_THRESHOLD:
        decision = PublishDecision.FAIL
    elif major_count >= MAJOR_THRESHOLD:
        decision = PublishDecision.FAIL
    elif major_count > 0 or minor_count > 2:
        decision = PublishDecision.PASS_WITH_REVISIONS
    else:
        decision = PublishDecision.PASS

    weighted_penalty = sum(
        SEVERITY_WEIGHT.get(i.severity, 1) * i.confidence
        for i in all_issues
    )
    max_possible = len(all_issues) * SEVERITY_WEIGHT[Severity.BLOCKER] if all_issues else 1
    penalty_ratio = min(weighted_penalty / max(max_possible, 1), 1.0)
    overall_score = round(max(0, (1 - penalty_ratio) * 100), 1)

    score_breakdown: dict[str, float] = {}
    for ro in reviewer_outputs:
        for key, val in ro.scores.items():
            score_breakdown[f"{ro.reviewer_name}.{key}"] = round(val, 3)

    summaries = []
    for ro in reviewer_outputs:
        if ro.summary:
            summaries.append(f"[{ro.reviewer_name}] {ro.summary}")

    summary_text = "\n".join(summaries)

    return FirewallVerdict(
        decision=decision,
        overall_score=overall_score,
        total_issues=len(all_issues),
        blocker_count=blocker_count,
        major_count=major_count,
        minor_count=minor_count,
        style_count=style_count,
        reviewer_outputs=reviewer_outputs,
        all_issues=all_issues,
        score_breakdown=score_breakdown,
        summary=summary_text,
    )

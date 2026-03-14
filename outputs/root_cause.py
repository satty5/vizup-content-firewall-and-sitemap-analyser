"""
Root Cause Dashboard (Module 4: Learning Layer)
Across many drafts, identifies recurring failure modes.

Example output:
  21% have competitor leakage
  38% miss ICP context
  46% contain intro fluff
  33% use generic frameworks
  18% contain logical overclaiming

This is what improves generation upstream.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

from models.review_result import FirewallVerdict
from config.taxonomy import IssueCategory, Severity


class RootCauseDashboard:
    """Tracks patterns across multiple review runs to identify systemic issues."""

    def __init__(self, history_path: str = "review_history.jsonl"):
        self.history_path = Path(history_path)
        self._history: list[dict] = []
        self._load_history()

    def _load_history(self) -> None:
        if self.history_path.exists():
            with open(self.history_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self._history.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

    def record_review(self, verdict: FirewallVerdict, content_id: str = "") -> None:
        """Record a review verdict for pattern analysis."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "content_id": content_id,
            "decision": verdict.decision.value,
            "overall_score": verdict.overall_score,
            "total_issues": verdict.total_issues,
            "blocker_count": verdict.blocker_count,
            "major_count": verdict.major_count,
            "minor_count": verdict.minor_count,
            "categories": [i.category.value for i in verdict.all_issues],
            "severities": [i.severity.value for i in verdict.all_issues],
            "reviewers": [i.reviewer for i in verdict.all_issues],
            "score_breakdown": verdict.score_breakdown,
        }
        self._history.append(entry)

        with open(self.history_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def generate_dashboard(self, last_n: int = 50) -> str:
        """Generate a root cause analysis dashboard from recent reviews."""
        recent = self._history[-last_n:]
        if not recent:
            return "No review history available."

        total_reviews = len(recent)
        lines: list[str] = []

        lines.append("=" * 72)
        lines.append("  ROOT CAUSE DASHBOARD")
        lines.append(f"  Analyzing last {total_reviews} reviews")
        lines.append("=" * 72)

        pass_count = sum(1 for r in recent if r["decision"] == "pass")
        revision_count = sum(1 for r in recent if r["decision"] == "pass_with_revisions")
        fail_count = sum(1 for r in recent if r["decision"] == "fail")

        lines.append("")
        lines.append("  PUBLISH RATES")
        lines.append(f"    Pass:            {pass_count}/{total_reviews} ({_pct(pass_count, total_reviews)})")
        lines.append(f"    Pass w/revisions: {revision_count}/{total_reviews} ({_pct(revision_count, total_reviews)})")
        lines.append(f"    Fail:            {fail_count}/{total_reviews} ({_pct(fail_count, total_reviews)})")

        avg_score = sum(r["overall_score"] for r in recent) / total_reviews
        lines.append(f"    Avg score:       {avg_score:.1f}/100")

        lines.append("")
        lines.append("-" * 72)
        lines.append("  TOP FAILURE MODES (% of reviews containing this issue)")
        lines.append("-" * 72)

        category_counts: Counter[str] = Counter()
        for r in recent:
            seen_categories = set(r.get("categories", []))
            for cat in seen_categories:
                category_counts[cat] += 1

        for cat, count in category_counts.most_common(15):
            pct = _pct(count, total_reviews)
            bar = _bar(count / total_reviews)
            lines.append(f"    {cat:.<40} {pct:>5}  {bar}")

        lines.append("")
        lines.append("-" * 72)
        lines.append("  SEVERITY DISTRIBUTION (across all reviews)")
        lines.append("-" * 72)

        severity_counts: Counter[str] = Counter()
        for r in recent:
            for sev in r.get("severities", []):
                severity_counts[sev] += 1

        total_issues = sum(severity_counts.values())
        for sev in ["blocker", "major", "minor", "style"]:
            count = severity_counts.get(sev, 0)
            pct = _pct(count, total_issues) if total_issues else "0%"
            lines.append(f"    {sev.upper():.<20} {count:>5} issues  ({pct})")

        lines.append("")
        lines.append("-" * 72)
        lines.append("  REVIEWER CONTRIBUTION (which reviewers find the most issues)")
        lines.append("-" * 72)

        reviewer_counts: Counter[str] = Counter()
        for r in recent:
            for rev in r.get("reviewers", []):
                reviewer_counts[rev] += 1

        for rev, count in reviewer_counts.most_common():
            lines.append(f"    {rev:.<35} {count:>5} issues")

        avg_blockers = sum(r["blocker_count"] for r in recent) / total_reviews
        avg_majors = sum(r["major_count"] for r in recent) / total_reviews

        lines.append("")
        lines.append("-" * 72)
        lines.append("  KEY METRICS")
        lines.append("-" * 72)
        lines.append(f"    Avg blockers per article:  {avg_blockers:.1f}")
        lines.append(f"    Avg major issues per article: {avg_majors:.1f}")
        lines.append(f"    Avg total issues per article: {sum(r['total_issues'] for r in recent) / total_reviews:.1f}")

        blocker_cats: Counter[str] = Counter()
        for r in recent:
            cats = r.get("categories", [])
            sevs = r.get("severities", [])
            for cat, sev in zip(cats, sevs):
                if sev == "blocker":
                    blocker_cats[cat] += 1

        if blocker_cats:
            lines.append("")
            lines.append("  TOP BLOCKER CATEGORIES:")
            for cat, count in blocker_cats.most_common(5):
                lines.append(f"    {cat}: {count} times")

        lines.append("")
        lines.append("=" * 72)

        return "\n".join(lines)

    def get_trending_issues(self, window: int = 10) -> dict:
        """Compare recent window to overall to find worsening patterns."""
        if len(self._history) < window * 2:
            return {"status": "insufficient_data"}

        recent = self._history[-window:]
        older = self._history[-window * 2 : -window]

        recent_cats = Counter()
        older_cats = Counter()

        for r in recent:
            for cat in set(r.get("categories", [])):
                recent_cats[cat] += 1

        for r in older:
            for cat in set(r.get("categories", [])):
                older_cats[cat] += 1

        trending_up = {}
        trending_down = {}

        all_cats = set(list(recent_cats.keys()) + list(older_cats.keys()))
        for cat in all_cats:
            recent_rate = recent_cats.get(cat, 0) / window
            older_rate = older_cats.get(cat, 0) / window
            diff = recent_rate - older_rate
            if diff > 0.1:
                trending_up[cat] = round(diff, 2)
            elif diff < -0.1:
                trending_down[cat] = round(abs(diff), 2)

        return {
            "worsening": trending_up,
            "improving": trending_down,
        }


def _pct(count: int, total: int) -> str:
    if total == 0:
        return "0%"
    return f"{count / total * 100:.0f}%"


def _bar(ratio: float, width: int = 20) -> str:
    filled = int(ratio * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"

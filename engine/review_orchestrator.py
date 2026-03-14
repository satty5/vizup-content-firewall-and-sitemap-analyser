"""
Review orchestrator — runs all reviewers in the correct order.

Architecture:
1. Rule engine runs first (fast, deterministic, cheap)
2. All LLM reviewers run concurrently (context fidelity, brand safety, slop, logic, editorial, structural)
3. Results aggregated into final verdict
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable

from models.content_unit import ParsedContent
from models.review_result import ReviewerOutput, ReviewIssue, FirewallVerdict
from models.brand_context import BrandContext, ContentBrief
from engine.rule_engine import run_rule_engine
from engine.scoring import compute_verdict
from reviewers.context_fidelity import ContextFidelityReviewer
from reviewers.brand_safety import BrandSafetyReviewer
from reviewers.ai_slop_detector import AISlopDetector
from reviewers.logic_checker import LogicChecker
from reviewers.editorial_quality import EditorialQualityReviewer
from reviewers.structural_quality import StructuralQualityReviewer
from config.settings import MAX_CONCURRENT_REVIEWERS


OnProgress = Callable[[str, str], None]


async def run_full_review(
    parsed: ParsedContent,
    brand_context: BrandContext,
    brief: ContentBrief | None = None,
    on_progress: OnProgress | None = None,
    skip_reviewers: list[str] | None = None,
) -> FirewallVerdict:
    """
    Run the complete firewall review pipeline.

    1. Deterministic rule engine (Module 1: Pre-Publish Linter)
    2. All LLM reviewers concurrently (Module 2: Deep Editorial Auditor)
    3. Aggregate into verdict
    """
    skip = set(skip_reviewers or [])
    start_time = time.time()

    def _progress(stage: str, detail: str = "") -> None:
        if on_progress:
            on_progress(stage, detail)

    _progress("rule_engine", "Running deterministic checks...")
    rule_issues = run_rule_engine(parsed, brand_context)

    rule_output = ReviewerOutput(
        reviewer_name="rule_engine",
        issues=rule_issues,
        scores={"deterministic_issues": len(rule_issues)},
        summary=f"Found {len(rule_issues)} issues via deterministic rules.",
        passed=not any(
            i.severity.value == "blocker" for i in rule_issues
        ),
    )
    _progress("rule_engine", f"Done — {len(rule_issues)} issues found.")

    all_reviewers = [
        ContextFidelityReviewer(),
        BrandSafetyReviewer(),
        AISlopDetector(),
        LogicChecker(),
        EditorialQualityReviewer(),
        StructuralQualityReviewer(),
    ]

    active_reviewers = [r for r in all_reviewers if r.name not in skip]

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REVIEWERS)

    async def _run_reviewer(reviewer) -> ReviewerOutput:
        async with semaphore:
            _progress(reviewer.name, f"Starting {reviewer.name}...")
            try:
                result = await reviewer.review(parsed, brand_context, brief)
                _progress(
                    reviewer.name,
                    f"Done — {len(result.issues)} issues, passed={result.passed}",
                )
                return result
            except Exception as e:
                _progress(reviewer.name, f"ERROR: {e}")
                return ReviewerOutput(
                    reviewer_name=reviewer.name,
                    issues=[],
                    summary=f"Reviewer failed with error: {e}",
                    passed=True,
                )

    _progress("llm_reviewers", f"Running {len(active_reviewers)} LLM reviewers concurrently...")
    reviewer_outputs = await asyncio.gather(
        *[_run_reviewer(r) for r in active_reviewers]
    )

    all_outputs = [rule_output] + list(reviewer_outputs)

    verdict = compute_verdict(all_outputs)

    elapsed = round(time.time() - start_time, 2)
    _progress("complete", f"Review complete in {elapsed}s — Decision: {verdict.decision.value}")

    return verdict


async def run_quick_review(
    parsed: ParsedContent,
    brand_context: BrandContext,
    brief: ContentBrief | None = None,
    on_progress: OnProgress | None = None,
) -> FirewallVerdict:
    """
    Quick review — rules + brand safety + slop detection only.
    Use for fast pre-screening before investing in full review.
    """
    return await run_full_review(
        parsed=parsed,
        brand_context=brand_context,
        brief=brief,
        on_progress=on_progress,
        skip_reviewers=[
            "context_fidelity",
            "logic_checker",
            "editorial_quality",
            "structural_quality",
        ],
    )

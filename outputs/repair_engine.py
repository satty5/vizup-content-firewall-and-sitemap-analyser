"""
Repair Engine (Module 3)
Given flagged issues, rewrites only the bad lines/paragraphs
while preserving the good parts.

NOT a full rewrite. Targeted surgical fixes only.
"""

from __future__ import annotations

from models.review_result import FirewallVerdict, ReviewIssue
from models.brand_context import BrandContext, ContentBrief
from config.taxonomy import Severity
from engine.llm_judge import llm_review


_SYSTEM_PROMPT = """You are a Content Repair Engine. You receive flagged issues from a content quality firewall
and your job is to produce MINIMAL, TARGETED fixes. Not rewrites. Fixes.

Rules:
1. Only fix what is flagged. Do not rewrite good content.
2. Preserve the author's voice and style where possible.
3. For COMPETITOR_PROMOTION: remove the competitor reference entirely or replace with brand-owned language.
4. For AI_SLOP: rewrite with specific, concrete language tied to the actual topic.
5. For UNSUPPORTED_CLAIM: add qualification ("can help", "in many cases") or remove the overclaim.
6. For LOGICAL_GAP: add the missing logical step or soften the causal claim.
7. For EMPTY_ABSTRACTION: replace with a concrete statement or delete.
8. For SEMANTIC_REPETITION: keep the strongest version, suggest deleting others.
9. For REDUNDANT_PARAGRAPH: suggest deletion with brief justification.

Respond in JSON:
{
  "repairs": [
    {
      "original_text": "the exact text to replace",
      "repaired_text": "the fixed version",
      "issue_category": "the category of the issue being fixed",
      "repair_rationale": "why this fix works"
    }
  ],
  "unrepairable": [
    {
      "original_text": "text that cannot be fixed with a targeted repair",
      "reason": "why this needs a full rewrite or human intervention"
    }
  ],
  "summary": "1-2 sentence summary of repairs made"
}

Keep repairs minimal. If a sentence needs to be deleted, set repaired_text to empty string.
If a paragraph needs major restructuring, put it in unrepairable."""


async def generate_repairs(
    verdict: FirewallVerdict,
    brand_context: BrandContext,
    brief: ContentBrief | None = None,
    max_issues: int = 20,
) -> dict:
    """Generate targeted repairs for flagged issues."""

    fixable = [
        i for i in verdict.all_issues
        if i.severity in {Severity.BLOCKER, Severity.MAJOR}
        and i.line_text
    ]

    fixable.sort(
        key=lambda i: (0 if i.severity == Severity.BLOCKER else 1, -i.confidence)
    )
    fixable = fixable[:max_issues]

    if not fixable:
        return {
            "repairs": [],
            "unrepairable": [],
            "summary": "No issues requiring repair.",
        }

    issues_text = []
    for idx, issue in enumerate(fixable, 1):
        issues_text.append(
            f"{idx}. [{issue.category.value}] [{issue.severity.value}]\n"
            f"   Text: \"{issue.line_text[:200]}\"\n"
            f"   Issue: {issue.explanation}\n"
            f"   Suggested direction: {issue.suggested_fix}"
        )

    user_prompt = f"""BRAND CONTEXT:
{brand_context.to_context_string()}

CONTENT BRIEF:
{brief.to_context_string() if brief else 'Not provided'}

ISSUES TO REPAIR:
{chr(10).join(issues_text)}

Generate minimal, targeted repairs for each issue. Respond in the required JSON format."""

    result = await llm_review(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    if result.get("parse_error"):
        return {
            "repairs": [],
            "unrepairable": [],
            "summary": "Failed to generate repairs — LLM response parsing error.",
            "raw_response": result.get("raw_response", ""),
        }

    return result


def format_repairs_report(repair_result: dict) -> str:
    """Format repair results into a readable report."""
    lines: list[str] = []

    lines.append("=" * 72)
    lines.append("  REPAIR ENGINE — TARGETED FIXES")
    lines.append("=" * 72)
    lines.append("")

    repairs = repair_result.get("repairs", [])
    if repairs:
        lines.append(f"  {len(repairs)} repairs generated:")
        lines.append("")
        for idx, repair in enumerate(repairs, 1):
            lines.append(f"  Repair #{idx} [{repair.get('issue_category', 'UNKNOWN')}]")
            orig = repair.get("original_text", "")[:150]
            fixed = repair.get("repaired_text", "")[:150]
            lines.append(f'  BEFORE: "{orig}"')
            if fixed:
                lines.append(f'  AFTER:  "{fixed}"')
            else:
                lines.append("  ACTION: DELETE")
            lines.append(f"  WHY:    {repair.get('repair_rationale', '')}")
            lines.append("")

    unrepairable = repair_result.get("unrepairable", [])
    if unrepairable:
        lines.append("-" * 72)
        lines.append(f"  {len(unrepairable)} issues need human intervention:")
        lines.append("")
        for item in unrepairable:
            lines.append(f'  Text: "{item.get("original_text", "")[:150]}"')
            lines.append(f"  Reason: {item.get('reason', '')}")
            lines.append("")

    summary = repair_result.get("summary", "")
    if summary:
        lines.append("-" * 72)
        lines.append(f"  Summary: {summary}")

    lines.append("")
    lines.append("=" * 72)

    return "\n".join(lines)

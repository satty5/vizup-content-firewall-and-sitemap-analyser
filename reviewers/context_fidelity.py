"""
Layer 1: Context Fidelity Reviewer
Checks whether the content actually reflects the input brief.

Questions it answers:
- Does each section connect to the target keyword, topic, ICP, funnel stage, and brand positioning?
- Is the draft missing required context from the brief?
- Has it invented context not provided?
- Is the content drifting into adjacent but irrelevant territory?
"""

from __future__ import annotations

from models.content_unit import ParsedContent, UnitType
from models.review_result import ReviewerOutput, ReviewIssue
from models.brand_context import BrandContext, ContentBrief
from config.taxonomy import IssueCategory, Severity
from engine.llm_judge import llm_review
from reviewers.base_reviewer import BaseReviewer


_SYSTEM_PROMPT = """You are a Context Fidelity Reviewer for a content quality firewall.
Your job is to check whether the content faithfully reflects the input brief and brand context.
You are strict, precise, and miss nothing.

You must identify:
1. MISSING_CONTEXT — required topics, keywords, or angles from the brief that are absent
2. CONTEXT_DRIFT — sections that wander into adjacent but irrelevant territory
3. INVENTED_CONTEXT — claims, examples, or positioning not grounded in the brief or brand context
4. READER_INTENT_MISMATCH — content that doesn't match the target audience, funnel stage, or intent

For each issue, respond in this JSON format:
{
  "issues": [
    {
      "line_text": "the offending text (first 120 chars)",
      "section": "parent heading if any",
      "category": "MISSING_CONTEXT | CONTEXT_DRIFT | INVENTED_CONTEXT | READER_INTENT_MISMATCH",
      "severity": "blocker | major | minor",
      "explanation": "why this is a problem",
      "suggested_fix": "what should be done",
      "confidence": 0.0-1.0
    }
  ],
  "context_coverage_score": 0.0-1.0,
  "brief_alignment_score": 0.0-1.0,
  "summary": "2-3 sentence assessment"
}

If content is perfect, return empty issues array with high scores.
Be surgical. Do not flag things that are reasonable expansions of the topic.
Flag things that genuinely miss the brief or invent unsupported context."""


class ContextFidelityReviewer(BaseReviewer):
    name = "context_fidelity"

    async def review(
        self,
        parsed: ParsedContent,
        brand_context: BrandContext,
        brief: ContentBrief | None = None,
    ) -> ReviewerOutput:
        sections_text = []
        for section in parsed.sections:
            heading = section.heading.text if section.heading else "Intro"
            paras = "\n".join(p.text for p in section.paragraphs)
            items = "\n".join(f"- {li.text}" for li in section.list_items)
            sections_text.append(f"## {heading}\n{paras}\n{items}")

        content_for_review = "\n\n".join(sections_text)

        user_prompt = f"""BRAND CONTEXT:
{brand_context.to_context_string()}

CONTENT BRIEF:
{brief.to_context_string() if brief else "No specific brief provided — evaluate against brand context only."}

CONTENT TO REVIEW:
{content_for_review}

Review this content for context fidelity. Check every section against the brief and brand context.
Respond in the required JSON format."""

        result = await llm_review(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        issues: list[ReviewIssue] = []
        if not result.get("parse_error"):
            for item in result.get("issues", []):
                cat_str = item.get("category", "CONTEXT_DRIFT")
                try:
                    category = IssueCategory(cat_str)
                except ValueError:
                    category = IssueCategory.CONTEXT_DRIFT

                sev_str = item.get("severity", "major")
                try:
                    severity = Severity(sev_str)
                except ValueError:
                    severity = Severity.MAJOR

                issues.append(ReviewIssue(
                    line_text=item.get("line_text", ""),
                    section=item.get("section", ""),
                    category=category,
                    severity=severity,
                    explanation=item.get("explanation", ""),
                    suggested_fix=item.get("suggested_fix", ""),
                    confidence=item.get("confidence", 0.8),
                    reviewer=self.name,
                ))

        scores = {
            "context_coverage": result.get("context_coverage_score", 0.5),
            "brief_alignment": result.get("brief_alignment_score", 0.5),
        }

        has_blocker = any(i.severity == Severity.BLOCKER for i in issues)
        return ReviewerOutput(
            reviewer_name=self.name,
            issues=issues,
            scores=scores,
            summary=result.get("summary", ""),
            passed=not has_blocker,
        )

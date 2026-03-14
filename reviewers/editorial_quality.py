"""
Layer 5: Human Editorial Quality Reviewer
Different from slop detection — a sentence can be non-robotic and still useless.

This layer asks:
- Is this useful?
- Is this specific?
- Would an actual expert write this?
- Is there a real point of view?
- Is this written for a reader, or just assembled to satisfy structure?

Checks for: specificity, examples, depth, originality, editorial sharpness,
sentence economy, reader intent match.
"""

from __future__ import annotations

from models.content_unit import ParsedContent
from models.review_result import ReviewerOutput, ReviewIssue
from models.brand_context import BrandContext, ContentBrief
from config.taxonomy import IssueCategory, Severity
from engine.llm_judge import llm_review
from reviewers.base_reviewer import BaseReviewer


_SYSTEM_PROMPT = """You are an Editorial Quality Reviewer — you think like a senior human editor at a top publication.
Your job is NOT to check grammar or detect AI patterns. Other reviewers handle that.

Your job is to answer one question for every section and paragraph:
"Would a serious editor keep this? Would a real expert write this?"

You check:

1. SPECIFICITY — Does the content have concrete details, real examples, actual numbers, named methods?
   Or is it all abstract advice that could apply to anything?

2. DEPTH — Does it go beyond surface-level? Does it explain WHY and HOW, not just WHAT?
   Shallow content that lists things without explaining them fails this check.

3. POINT OF VIEW — Does the author have a stance? Is there editorial courage?
   Content that hedges everything and commits to nothing fails.

4. READER VALUE — If the target reader spent 5 minutes reading this section, would they learn something actionable?
   Or is this "assembled to satisfy structure" rather than written to help someone?

5. SENTENCE ECONOMY — Is every sentence earning its place? Are there bloated sentences saying in 20 words what could be said in 8?

6. EXAMPLES — Are there real examples, case references, scenarios? Or is everything hypothetical?

7. ORIGINALITY — Is this saying something the reader couldn't find in 10 other articles?
   Or is it the same recycled advice?

WHAT TO FLAG:

NO_POINT_OF_VIEW — Section has no editorial stance, just lists both sides without commitment.
WEAK_EXAMPLE — Example given is generic, hypothetical, or doesn't actually illustrate the point.
READER_INTENT_MISMATCH — Content doesn't match what the target reader actually needs.
REDUNDANT_PARAGRAPH — Paragraph could be deleted without losing any information.
TONE_MISMATCH — Writing tone doesn't match the brand voice or target audience.

Respond in JSON:
{
  "issues": [
    {
      "line_text": "the offending text (first 120 chars)",
      "section": "parent heading if any",
      "category": "NO_POINT_OF_VIEW | WEAK_EXAMPLE | READER_INTENT_MISMATCH | REDUNDANT_PARAGRAPH | TONE_MISMATCH",
      "severity": "major | minor | style",
      "explanation": "what editorial weakness exists",
      "suggested_fix": "specific direction for improvement",
      "confidence": 0.0-1.0
    }
  ],
  "specificity_score": 0.0-1.0,
  "depth_score": 0.0-1.0,
  "originality_score": 0.0-1.0,
  "editorial_sharpness_score": 0.0-1.0,
  "reader_value_score": 0.0-1.0,
  "summary": "2-3 sentence editorial verdict"
}

A perfectly written but completely obvious article should score LOW.
A rough but insightful article should score HIGHER.
Substance over polish. Always."""


class EditorialQualityReviewer(BaseReviewer):
    name = "editorial_quality"

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
Company: {brand_context.company_name}
Brand voice: {brand_context.brand_voice}
Tone guidelines: {brand_context.tone_guidelines}
ICP: {brand_context.icp_description}
ICP pain points: {', '.join(brand_context.icp_pain_points) if brand_context.icp_pain_points else 'Not specified'}

CONTENT BRIEF:
Topic: {brief.topic if brief else 'Not specified'}
Target audience: {brief.target_audience if brief else 'Not specified'}
Intent: {brief.intent if brief else 'Not specified'}
Funnel stage: {brief.funnel_stage if brief else 'Not specified'}
Angle: {brief.angle if brief else 'Not specified'}

CONTENT TO REVIEW:
{content_for_review}

Judge every section as a senior editor would. Check for substance, not polish.
Respond in the required JSON format."""

        result = await llm_review(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        issues: list[ReviewIssue] = []
        if not result.get("parse_error"):
            for item in result.get("issues", []):
                cat_str = item.get("category", "NO_POINT_OF_VIEW")
                try:
                    category = IssueCategory(cat_str)
                except ValueError:
                    category = IssueCategory.NO_POINT_OF_VIEW

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
                    confidence=item.get("confidence", 0.75),
                    reviewer=self.name,
                ))

        scores = {
            "specificity": result.get("specificity_score", 0.5),
            "depth": result.get("depth_score", 0.5),
            "originality": result.get("originality_score", 0.5),
            "editorial_sharpness": result.get("editorial_sharpness_score", 0.5),
            "reader_value": result.get("reader_value_score", 0.5),
        }

        major_count = sum(1 for i in issues if i.severity == Severity.MAJOR)
        return ReviewerOutput(
            reviewer_name=self.name,
            issues=issues,
            scores=scores,
            summary=result.get("summary", ""),
            passed=major_count < 4,
        )

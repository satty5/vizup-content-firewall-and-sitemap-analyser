"""
Layer 6: Structural Quality Reviewer
Checks if the content is mechanically sound.

Includes: heading hierarchy, redundancy between sections, missing definitions,
weak introduction, dead conclusion, abrupt transitions, overlong paragraphs,
list bloat, keyword stuffing, FAQ irrelevance.
"""

from __future__ import annotations

from models.content_unit import ParsedContent, UnitType
from models.review_result import ReviewerOutput, ReviewIssue
from models.brand_context import BrandContext, ContentBrief
from config.taxonomy import IssueCategory, Severity
from engine.llm_judge import llm_review
from reviewers.base_reviewer import BaseReviewer


_SYSTEM_PROMPT = """You are a Structural Quality Reviewer for a content quality firewall.
You check whether the content is mechanically sound as a publishable document.

Note: heading hierarchy and overlong paragraph checks are handled by the rule engine.
Your job is the SEMANTIC structural assessment that rules cannot do.

WHAT TO CHECK:

1. WEAK_INTRO — Does the introduction hook the reader? Does it set up the article's promise?
   Or is it generic filler before the real content starts?

2. DEAD_CONCLUSION — Does the conclusion add value? Does it synthesize insights?
   Or is it just restating what was already said, or a weak "in conclusion..." wrapper?

3. REDUNDANT_PARAGRAPH — Are any sections redundant with each other?
   Do two sections make the same point under different headings?

4. ABRUPT_TRANSITION — Are there jarring jumps between sections where the reader would be confused?

5. SEO_IRRELEVANT_SECTION — Are there sections that don't serve the target keyword or reader intent?
   Sections added for length or template completeness rather than reader need?

6. FAQ_IRRELEVANCE — If FAQs exist, are they genuinely useful? Or are they SEO-stuffed questions
   that no real person would ask?

7. KEYWORD_STUFFING — Is the target keyword forced into places where it reads unnaturally?

8. GENERIC_FRAMEWORK — Was this structure earned by the topic, or imposed by a template?
   Does every article follow the same "What is X → Benefits → How to → Best Practices → Conclusion" pattern
   regardless of whether the topic needs it?

9. SECTION NECESSITY — For each section, ask: "Does the reader need this section to achieve their goal?"
   If not, it's structural bloat.

Respond in JSON:
{
  "issues": [
    {
      "line_text": "the offending text or heading (first 120 chars)",
      "section": "parent heading if any",
      "category": "WEAK_INTRO | DEAD_CONCLUSION | REDUNDANT_PARAGRAPH | ABRUPT_TRANSITION | SEO_IRRELEVANT_SECTION | FAQ_IRRELEVANCE | KEYWORD_STUFFING | GENERIC_FRAMEWORK",
      "severity": "major | minor | style",
      "explanation": "what structural problem exists",
      "suggested_fix": "specific fix",
      "confidence": 0.0-1.0
    }
  ],
  "structural_soundness_score": 0.0-1.0,
  "section_necessity_score": 0.0-1.0,
  "flow_score": 0.0-1.0,
  "template_imposition_score": 0.0-1.0,
  "summary": "2-3 sentence assessment"
}

template_imposition_score: 0.0 = structure earned by topic, 1.0 = pure template imposition
A high template_imposition_score with GENERIC_FRAMEWORK issues is a serious quality signal."""


class StructuralQualityReviewer(BaseReviewer):
    name = "structural_quality"

    async def review(
        self,
        parsed: ParsedContent,
        brand_context: BrandContext,
        brief: ContentBrief | None = None,
    ) -> ReviewerOutput:
        outline = []
        for level, text in parsed.heading_hierarchy:
            indent = "  " * (level - 1)
            outline.append(f"{indent}H{level}: {text}")

        sections_text = []
        for section in parsed.sections:
            heading = section.heading.text if section.heading else "Intro"
            para_preview = "\n".join(
                p.text[:200] for p in section.paragraphs[:3]
            )
            item_count = len(section.list_items)
            sections_text.append(
                f"## {heading}\n{para_preview}"
                + (f"\n[{item_count} list items]" if item_count else "")
            )

        user_prompt = f"""TARGET KEYWORD: {brief.target_keyword if brief else 'Not specified'}
CONTENT TYPE: {brief.content_type if brief else 'Not specified'}
READER INTENT: {brief.intent if brief else 'Not specified'}
FUNNEL STAGE: {brief.funnel_stage if brief else 'Not specified'}

HEADING OUTLINE:
{chr(10).join(outline) if outline else 'No headings found'}

TOTAL STATS:
- Words: {parsed.total_words}
- Paragraphs: {parsed.total_paragraphs}
- Headings: {parsed.total_headings}
- Sections: {len(parsed.sections)}

CONTENT SECTIONS:
{chr(10).join(sections_text)}

FULL CONTENT (for intro/conclusion/transition checks):
{parsed.raw_text[:8000]}

Check the structural quality. Is this structure earned by the topic or imposed by a template?
Respond in the required JSON format."""

        result = await llm_review(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        issues: list[ReviewIssue] = []
        if not result.get("parse_error"):
            for item in result.get("issues", []):
                cat_str = item.get("category", "GENERIC_FRAMEWORK")
                try:
                    category = IssueCategory(cat_str)
                except ValueError:
                    category = IssueCategory.GENERIC_FRAMEWORK

                sev_str = item.get("severity", "minor")
                try:
                    severity = Severity(sev_str)
                except ValueError:
                    severity = Severity.MINOR

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
            "structural_soundness": result.get("structural_soundness_score", 0.5),
            "section_necessity": result.get("section_necessity_score", 0.5),
            "flow": result.get("flow_score", 0.5),
            "template_imposition": result.get("template_imposition_score", 0.5),
        }

        major_count = sum(1 for i in issues if i.severity == Severity.MAJOR)
        return ReviewerOutput(
            reviewer_name=self.name,
            issues=issues,
            scores=scores,
            summary=result.get("summary", ""),
            passed=major_count < 3,
        )

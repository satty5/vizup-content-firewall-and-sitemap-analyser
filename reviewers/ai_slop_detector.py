"""
Layer 3: AI Slop / AI Dump Detection
The heart of the system.

Detects:
- Fluffy intros, robotic transitions, template phrases
- Empty abstraction, fake authority, generic listicles
- Repetitive sentence rhythm, "framework for the sake of framework"
- Paragraph saying same thing in 3 ways
- Headings that sound useful but contain no unique information
- Overuse of symmetry: "not just X, but Y"
- Fake nuance without evidence
- Generic triads: "efficiency, scalability, and innovation"

Produces: slop score, robotic phrasing density, template residue score, semantic repetition score
"""

from __future__ import annotations

from models.content_unit import ParsedContent, UnitType
from models.review_result import ReviewerOutput, ReviewIssue
from models.brand_context import BrandContext, ContentBrief
from config.taxonomy import IssueCategory, Severity
from engine.llm_judge import llm_review
from reviewers.base_reviewer import BaseReviewer


_SYSTEM_PROMPT = """You are an AI Slop Detector — the most aggressive reviewer in a content quality firewall.
Your purpose is to find and destroy every trace of AI-generated filler, template residue, and empty language.

You review content LINE BY LINE and PARAGRAPH BY PARAGRAPH.

For each sentence, you must ask:
1. Why is this sentence here?
2. Does it add NEW, SPECIFIC information?
3. Is it a template phrase that could appear in any article on any topic?
4. Does it sound like an LLM default or a human expert with a real point?
5. Is it semantically redundant with nearby lines?
6. Could this be deleted without losing meaning?

WHAT TO FLAG:

AI_SLOP — Generic, fluffy, or template language. Examples:
- "In today's fast-paced digital landscape..."
- "Businesses looking to stay ahead..."
- "This not only helps X but also enables Y..."
- Sentences that could be true about literally anything

ROBOTIC_TRANSITION — Mechanical transitions. Examples:
- "Now let's dive into..."
- "Having established that..."
- "With that said..."

EMPTY_ABSTRACTION — Sounds profound, says nothing. Examples:
- "The importance of X cannot be overstated"
- "It plays a crucial role in..."
- "This is increasingly important..."

GENERIC_FRAMEWORK — Template structures imposed without earned relevance. Examples:
- Forced 3-step / 5-step frameworks
- "Benefits, Challenges, Best Practices, Future Trends" sections when topic doesn't need them
- Every article looking structurally identical
- Headings inserted for template reasons, not reader need

SEMANTIC_REPETITION — Same idea restated multiple ways within close proximity.
- Paragraph saying the same thing 2-3 different ways
- Heading and first sentence saying the same thing
- Consecutive sections making the same point

TEMPLATE_RESIDUE — Artifacts of AI generation templates.
- Overly symmetrical structure
- Perfectly balanced "on one hand / on the other hand"
- Stock conclusions that summarize without insight

For each issue, respond in JSON:
{
  "issues": [
    {
      "line_text": "the offending text (first 120 chars)",
      "section": "parent heading if any",
      "category": "AI_SLOP | ROBOTIC_TRANSITION | EMPTY_ABSTRACTION | GENERIC_FRAMEWORK | SEMANTIC_REPETITION | TEMPLATE_RESIDUE",
      "severity": "major | minor | style",
      "explanation": "exactly what makes this slop",
      "suggested_fix": "specific rewrite direction or 'delete'",
      "confidence": 0.0-1.0
    }
  ],
  "slop_score": 0.0-1.0,
  "robotic_density": 0.0-1.0,
  "template_residue_score": 0.0-1.0,
  "semantic_repetition_score": 0.0-1.0,
  "summary": "2-3 sentence verdict"
}

slop_score: 0.0 = perfectly human and specific, 1.0 = pure AI slop
robotic_density: fraction of sentences that sound robotic
template_residue_score: how much the structure feels template-imposed vs earned
semantic_repetition_score: how much content repeats itself

Be ruthless. An article full of correct grammar but empty meaning is WORSE than a rough draft with real ideas.
The question is always: "Was this structure earned by the topic, or imposed by a template?"

CRITICAL: Do not flag specific, concrete, useful statements just because they use common words.
Flag the ABSENCE of specificity, the PRESENCE of filler, the PATTERN of template."""


class AISlopDetector(BaseReviewer):
    name = "ai_slop_detector"

    async def review(
        self,
        parsed: ParsedContent,
        brand_context: BrandContext,
        brief: ContentBrief | None = None,
    ) -> ReviewerOutput:
        sections_text = []
        for section in parsed.sections:
            heading = section.heading.text if section.heading else "Intro"
            paras = "\n".join(
                f"[L{p.line_number}] {p.text}" for p in section.paragraphs
            )
            items = "\n".join(
                f"[L{li.line_number}] - {li.text}" for li in section.list_items
            )
            sections_text.append(f"## {heading}\n{paras}\n{items}")

        content_for_review = "\n\n".join(sections_text)

        user_prompt = f"""BRAND/TOPIC CONTEXT (for judging specificity):
Company: {brand_context.company_name}
Product: {brand_context.product_name}
What it does: {brand_context.what_product_does}
Topic: {brief.topic if brief else 'Not specified'}
Target keyword: {brief.target_keyword if brief else 'Not specified'}

CONTENT TO REVIEW (line numbers in [L#] prefix):
{content_for_review}

Review every line for AI slop, robotic language, template residue, empty abstraction, and semantic repetition.
Be surgical and specific. Respond in the required JSON format."""

        result = await llm_review(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        issues: list[ReviewIssue] = []
        if not result.get("parse_error"):
            for item in result.get("issues", []):
                cat_str = item.get("category", "AI_SLOP")
                try:
                    category = IssueCategory(cat_str)
                except ValueError:
                    category = IssueCategory.AI_SLOP

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
            "slop_score": result.get("slop_score", 0.5),
            "robotic_density": result.get("robotic_density", 0.5),
            "template_residue": result.get("template_residue_score", 0.5),
            "semantic_repetition": result.get("semantic_repetition_score", 0.5),
        }

        slop = scores.get("slop_score", 0.5)
        return ReviewerOutput(
            reviewer_name=self.name,
            issues=issues,
            scores=scores,
            summary=result.get("summary", ""),
            passed=slop < 0.4,
        )

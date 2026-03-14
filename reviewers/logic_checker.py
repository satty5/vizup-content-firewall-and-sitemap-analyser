"""
Layer 4: Logic & Truthfulness Checker
Catches grammatically correct but logically weak content.

Checks at sentence, paragraph, and whole-article level:
- Internal contradiction
- Unsupported claims
- Causal jumps
- False comparisons
- Undefined terms
- Vague pronouns
- Impossible assertions
- Sentence-to-sentence inconsistency
"""

from __future__ import annotations

from models.content_unit import ParsedContent
from models.review_result import ReviewerOutput, ReviewIssue
from models.brand_context import BrandContext, ContentBrief
from config.taxonomy import IssueCategory, Severity
from engine.llm_judge import llm_review
from reviewers.base_reviewer import BaseReviewer


_SYSTEM_PROMPT = """You are a Logic & Truthfulness Checker for a content quality firewall.
Your job is to catch content that is grammatically polished but logically weak, misleading, or unsupported.

You operate at three levels:
1. SENTENCE LEVEL — Is this single claim true, supportable, and logically sound?
2. PARAGRAPH LEVEL — Do the sentences within a paragraph form a coherent argument?
3. ARTICLE LEVEL — Are there contradictions between different sections?

WHAT TO FLAG:

UNSUPPORTED_CLAIM — Hard claims with no evidence, mechanism, or qualification.
Examples:
- "SEO traffic is free" → oversimplified / misleading
- "This guarantees better ranking" → impossible to guarantee
- "The tool reduces CAC" without explaining how
- "X is the best approach" without comparative evidence
- Percentages or numbers with no source

LOGICAL_GAP — Missing logical steps between cause and effect.
Examples:
- "Use this tool → get more revenue" with no middle steps
- "Content marketing drives growth" — how? for whom? what kind of growth?
- Jumping from problem statement to solution without connecting logic

INTERNAL_CONTRADICTION — The content says two incompatible things.
Examples:
- "Organic is the fastest growth channel" after saying it compounds slowly
- Claiming something is simple then listing 10 complex steps
- Saying "no technical knowledge needed" then using technical jargon

FACTUAL_RISK — Statements that could be factually wrong or dangerously misleading.
Examples:
- Legal or compliance claims without qualification
- Medical/health claims
- Financial projections presented as fact
- "Industry standard" claims without source

CAUSAL_JUMP — Asserting causation without evidence.
- "Companies that do X see Y% improvement" without source
- "This leads to..." without logical support

FALSE_COMPARISON — Comparing incomparable things or cherry-picked comparisons.

VAGUE_PRONOUN — "This", "It", "They" without clear referent causing confusion.

UNDEFINED_TERM — Technical or product-specific term used without definition.

Respond in JSON:
{
  "issues": [
    {
      "line_text": "the offending text (first 120 chars)",
      "section": "parent heading if any",
      "category": "UNSUPPORTED_CLAIM | LOGICAL_GAP | INTERNAL_CONTRADICTION | FACTUAL_RISK | CAUSAL_JUMP | FALSE_COMPARISON | VAGUE_PRONOUN | UNDEFINED_TERM",
      "severity": "blocker | major | minor",
      "explanation": "what is logically wrong",
      "suggested_fix": "how to fix the logic",
      "confidence": 0.0-1.0,
      "level": "sentence | paragraph | article"
    }
  ],
  "logic_score": 0.0-1.0,
  "claim_support_score": 0.0-1.0,
  "internal_consistency_score": 0.0-1.0,
  "summary": "2-3 sentence assessment"
}

INTERNAL_CONTRADICTION and FACTUAL_RISK with high confidence should be severity: blocker.
UNSUPPORTED_CLAIM is severity: major unless it's a minor overstatement.

Be precise. Don't flag reasonable generalizations or well-known facts.
Flag claims that a careful editor would question or that a reader could be misled by."""


class LogicChecker(BaseReviewer):
    name = "logic_checker"

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
            sections_text.append(f"## {heading}\n{paras}")

        content_for_review = "\n\n".join(sections_text)

        user_prompt = f"""BRAND CONTEXT (for judging claim validity):
Company: {brand_context.company_name}
Product: {brand_context.product_name}
What it does: {brand_context.what_product_does}
Approved claims: {'; '.join(brand_context.approved_claims) if brand_context.approved_claims else 'None listed'}
No-go claims: {'; '.join(brand_context.no_go_claims) if brand_context.no_go_claims else 'None listed'}
Proof points: {'; '.join(brand_context.proof_points) if brand_context.proof_points else 'None listed'}

CONTENT TO REVIEW:
{content_for_review}

Check every claim, every causal statement, every comparison for logical soundness.
Work at sentence level, paragraph level, and article level.
Respond in the required JSON format."""

        result = await llm_review(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        issues: list[ReviewIssue] = []
        if not result.get("parse_error"):
            for item in result.get("issues", []):
                cat_str = item.get("category", "LOGICAL_GAP")
                try:
                    category = IssueCategory(cat_str)
                except ValueError:
                    category = IssueCategory.LOGICAL_GAP

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
            "logic": result.get("logic_score", 0.5),
            "claim_support": result.get("claim_support_score", 0.5),
            "internal_consistency": result.get("internal_consistency_score", 0.5),
        }

        has_blocker = any(i.severity == Severity.BLOCKER for i in issues)
        return ReviewerOutput(
            reviewer_name=self.name,
            issues=issues,
            scores=scores,
            summary=result.get("summary", ""),
            passed=not has_blocker,
        )

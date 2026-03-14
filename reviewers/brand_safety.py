"""
Layer 2: Brand & Competitor Safety Reviewer
Hard-block layer for competitor leakage and brand dilution.

Catches:
- Mentions of competitors when not explicitly requested
- Subtle competitor promotion
- Brand dilution through "other tools you can use"
- Mentioning alternatives that weaken the client
- Cross-brand contamination from training context / prompt residue
"""

from __future__ import annotations

from models.content_unit import ParsedContent, UnitType
from models.review_result import ReviewerOutput, ReviewIssue
from models.brand_context import BrandContext, ContentBrief
from config.taxonomy import IssueCategory, Severity
from engine.llm_judge import llm_review
from reviewers.base_reviewer import BaseReviewer


_SYSTEM_PROMPT = """You are a Brand Safety & Competitor Leakage Reviewer for a content quality firewall.
Your job is the most critical in the pipeline — you prevent competitor promotion and brand dilution.

You must detect:
1. COMPETITOR_PROMOTION — any mention of competitors, their products, or links to them (unless explicitly allowed)
2. UNAUTHORIZED_BRAND_MENTION — references to brands/tools/platforms not approved in brand context
3. BRAND_DILUTION — "alternatives" lists, "other tools" suggestions, or language weakening the brand

Be EXTREMELY strict. Even subtle references count:
- "Tools like [Competitor]..." — BLOCKER
- "Unlike some platforms..." without grounding — flag
- "You could also try..." suggesting alternatives — flag
- Product names that might be competitors even if not in the banned list — flag with lower confidence
- Phrases like "industry leaders like X and Y" where X or Y compete — BLOCKER

Respond in this JSON format:
{
  "issues": [
    {
      "line_text": "the offending text (first 120 chars)",
      "section": "parent heading if any",
      "category": "COMPETITOR_PROMOTION | UNAUTHORIZED_BRAND_MENTION | BRAND_DILUTION",
      "severity": "blocker | major | minor",
      "explanation": "what is wrong and why it is dangerous",
      "suggested_fix": "specific fix",
      "confidence": 0.0-1.0,
      "detected_entity": "the competitor or brand name detected"
    }
  ],
  "brand_safety_score": 0.0-1.0,
  "competitor_leakage_detected": true/false,
  "summary": "2-3 sentence assessment"
}

COMPETITOR_PROMOTION issues are ALWAYS severity: blocker unless explicitly allowed in brand context.
When in doubt, flag it. False positives are better than letting competitor promotion through."""


class BrandSafetyReviewer(BaseReviewer):
    name = "brand_safety"

    async def review(
        self,
        parsed: ParsedContent,
        brand_context: BrandContext,
        brief: ContentBrief | None = None,
    ) -> ReviewerOutput:
        competitor_policy = []
        for c in brand_context.competitors:
            competitor_policy.append(
                f"- {c.name}: mention_allowed={c.mention_allowed}, "
                f"comparison_allowed={c.comparison_allowed}, notes={c.notes}"
            )
        banned_names = brand_context.get_competitor_names()

        user_prompt = f"""BRAND: {brand_context.company_name}
PRODUCT: {brand_context.product_name}

BANNED COMPETITOR NAMES (hard block if mentioned):
{chr(10).join(f'- {n}' for n in banned_names) if banned_names else 'No specific list — flag any competitor-like mentions'}

COMPETITOR POLICIES:
{chr(10).join(competitor_policy) if competitor_policy else 'Default: no competitor mentions allowed'}

NO-GO CLAIMS:
{chr(10).join(f'- {c}' for c in brand_context.no_go_claims) if brand_context.no_go_claims else 'None specified'}

CONTENT TO REVIEW:
{parsed.raw_text}

Scan every line for competitor mentions, brand dilution, unauthorized tool/platform references.
Respond in the required JSON format."""

        result = await llm_review(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        issues: list[ReviewIssue] = []
        if not result.get("parse_error"):
            for item in result.get("issues", []):
                cat_str = item.get("category", "COMPETITOR_PROMOTION")
                try:
                    category = IssueCategory(cat_str)
                except ValueError:
                    category = IssueCategory.COMPETITOR_PROMOTION

                sev_str = item.get("severity", "blocker")
                try:
                    severity = Severity(sev_str)
                except ValueError:
                    severity = Severity.BLOCKER

                issues.append(ReviewIssue(
                    line_text=item.get("line_text", ""),
                    section=item.get("section", ""),
                    category=category,
                    severity=severity,
                    explanation=item.get("explanation", ""),
                    suggested_fix=item.get("suggested_fix", ""),
                    confidence=item.get("confidence", 0.9),
                    reviewer=self.name,
                ))

        scores = {
            "brand_safety": result.get("brand_safety_score", 0.5),
        }

        has_blocker = any(i.severity == Severity.BLOCKER for i in issues)
        return ReviewerOutput(
            reviewer_name=self.name,
            issues=issues,
            scores=scores,
            summary=result.get("summary", ""),
            passed=not has_blocker,
        )

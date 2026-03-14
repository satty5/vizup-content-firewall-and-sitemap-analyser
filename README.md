# Vizup Soul — Content Quality Firewall

An independent review engine that audits content line by line, claim by claim, paragraph by paragraph before publish. The generator does not decide if content is good. This system decides if content is publishable.

This is not content QA. This is an **Editorial Intelligence Layer** that solves for trust, relevance, brand control, reasoning quality, and publishability.

## Architecture

```
Content Draft
    │
    ▼
┌─────────────────────────────────┐
│  Module 1: Pre-Publish Linter   │  ← Deterministic rules (fast, cheap)
│  Banned phrases, competitors,   │
│  structural checks, n-gram      │
│  repetition, regex patterns     │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Module 2: Deep Editorial Audit │  ← 6 LLM reviewers (concurrent)
│                                 │
│  Layer 1: Context Fidelity      │  Does content match the brief?
│  Layer 2: Brand Safety          │  Competitor leakage? Brand dilution?
│  Layer 3: AI Slop Detection     │  Template residue? Empty language?
│  Layer 4: Logic & Truth         │  Unsupported claims? Contradictions?
│  Layer 5: Editorial Quality     │  Specific? Deep? Has a POV?
│  Layer 6: Structural Quality    │  Template-imposed or earned structure?
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Scoring & Publish Decision     │
│  PASS / PASS WITH REVISIONS /   │
│  FAIL                           │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Module 3: Repair Engine        │  ← Targeted fixes, not full rewrites
│  Module 4: Learning Layer       │  ← Root cause tracking across drafts
└─────────────────────────────────┘
```

## Setup

```bash
cd Vizup_Soul
cp .env.example .env
# Add your OpenAI API key to .env

pip install -r requirements.txt
```

## Usage

### CLI

```bash
# Full review
python run.py review examples/sample_content.md --brand examples/sample_brand_context.json --brief examples/sample_brief.json

# Quick scan (rules + brand safety + slop only)
python run.py quick examples/sample_content.md --brand examples/sample_brand_context.json

# Full audit with repair suggestions
python run.py audit examples/sample_content.md --brand examples/sample_brand_context.json --brief examples/sample_brief.json

# Save report to file
python run.py review content.md --brand brand.json -o report.json

# Root cause dashboard (after multiple reviews)
python run.py dashboard
```

### Python API

```python
import asyncio
from models.brand_context import BrandContext, ContentBrief
from firewall import ContentFirewall

brand = BrandContext(
    company_name="VizUp",
    product_name="VizUp Organic",
    what_product_does="AI-powered organic growth platform",
    banned_competitor_names=["Jasper", "SurferSEO", "Clearscope"],
    # ... full brand context
)

brief = ContentBrief(
    target_keyword="B2B SaaS content marketing",
    topic="How to build a content engine",
    # ... full brief
)

firewall = ContentFirewall(brand_context=brand, brief=brief)

async def main():
    content = open("draft.md").read()

    # Full review
    verdict = await firewall.review(content)
    print(verdict.decision)        # pass / pass_with_revisions / fail
    print(verdict.overall_score)   # 0-100

    # Redline report
    print(firewall.get_redline_report())

    # Repair suggestions
    repairs = await firewall.get_repairs()

    # Root cause dashboard (across multiple reviews)
    print(firewall.get_dashboard())

asyncio.run(main())
```

## The 6 Review Layers

### Layer 1: Context Fidelity
Does the content actually reflect the input brief? Catches missing context, context drift, invented claims, and audience mismatch.

### Layer 2: Brand & Competitor Safety
**Hard-block layer.** Catches competitor mentions, brand dilution, "other tools" suggestions, and cross-brand contamination. Any competitor promotion is an automatic BLOCKER.

### Layer 3: AI Slop Detection
The heart of the system. Detects fluffy intros, robotic transitions, template phrases, empty abstractions, semantic repetition, generic frameworks, and "polished but says nothing" content. Produces slop score, robotic density, template residue score, and semantic repetition score.

### Layer 4: Logic & Truthfulness
Catches grammatically correct but logically weak content. Checks at sentence, paragraph, and article level for unsupported claims, internal contradictions, causal jumps, false comparisons, and factual risk.

### Layer 5: Editorial Quality
Different from slop detection. Asks: Is this useful? Is this specific? Would an expert write this? Is there a real POV? Checks specificity, depth, originality, editorial sharpness, and reader value.

### Layer 6: Structural Quality
Checks if the structure was earned by the topic or imposed by a template. Catches weak intros, dead conclusions, redundant sections, abrupt transitions, FAQ irrelevance, and keyword stuffing.

## Issue Taxonomy

Every issue is tagged with one of these categories:

| Category | Description |
|----------|-------------|
| `MISSING_CONTEXT` | Required context from brief is absent |
| `CONTEXT_DRIFT` | Content drifts into irrelevant territory |
| `INVENTED_CONTEXT` | Claims or examples not grounded in brief |
| `COMPETITOR_PROMOTION` | Competitor mentioned or promoted |
| `UNAUTHORIZED_BRAND_MENTION` | Unapproved brand/tool reference |
| `BRAND_DILUTION` | Language weakening the brand |
| `AI_SLOP` | Generic, fluffy, template language |
| `ROBOTIC_TRANSITION` | Mechanical, formulaic transitions |
| `EMPTY_ABSTRACTION` | Sounds profound, says nothing |
| `GENERIC_FRAMEWORK` | Template structure without earned relevance |
| `SEMANTIC_REPETITION` | Same idea restated multiple ways |
| `TEMPLATE_RESIDUE` | Artifacts of AI generation templates |
| `LOGICAL_GAP` | Missing logical steps |
| `UNSUPPORTED_CLAIM` | Claims without evidence |
| `INTERNAL_CONTRADICTION` | Content contradicts itself |
| `FACTUAL_RISK` | Potentially false or misleading |
| `CAUSAL_JUMP` | Asserting causation without evidence |
| `TONE_MISMATCH` | Tone doesn't match brand |
| `READER_INTENT_MISMATCH` | Doesn't match reader's actual need |
| `NO_POINT_OF_VIEW` | No editorial stance |
| `WEAK_EXAMPLE` | Generic or unhelpful example |
| `REDUNDANT_PARAGRAPH` | Can be deleted without losing meaning |

## Severity Levels

| Level | Meaning | Action |
|-------|---------|--------|
| **Blocker** | Must fix before publish | Competitor promotion, fabricated claims, wrong brand |
| **Major** | High-risk quality issue | AI slop, empty sections, contradictions |
| **Minor** | Improves quality | Repetitive phrasing, weak transitions |
| **Style** | Optional polish | Sentence tightening, minor tone adjustments |

## 4 Outputs

1. **Publish Decision**: Pass / Pass with Revisions / Fail
2. **Redline Report**: Line-by-line issue report with category, severity, explanation, and fix
3. **Repair Suggestions**: Targeted fixes for blocker and major issues (not full rewrites)
4. **Root Cause Dashboard**: Recurring failure patterns across multiple reviews

## Brand Context Pack

The firewall uses a structured brand context for every review:
- Company & product description
- ICP (size, industry, pain points, geography)
- Brand voice & tone guidelines
- Approved and no-go claims
- Competitor policies
- Proof points
- Writing style DOs and DON'Ts

Without this context, the reviewer cannot reliably detect drift. See `examples/sample_brand_context.json`.

## Key Metric

> How much human editorial intervention is still required after QA?

That tells you whether the system is actually helping.

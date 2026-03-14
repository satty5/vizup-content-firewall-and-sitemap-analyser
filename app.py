"""
Content Quality Firewall — Web Interface
FastAPI backend serving the review engine + static HTML frontend.

Run:  python app.py
Then: open http://localhost:8500
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from models.brand_context import BrandContext, ContentBrief
from firewall import ContentFirewall
from parsers.file_parser import parse_file
from outputs.root_cause import RootCauseDashboard
from engine.llm_judge import llm_review, set_model
from config.settings import (
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    DEFAULT_MODEL,
    MODEL_REGISTRY,
)

app = FastAPI(title="Vizup Soul — Content Quality Firewall")
app.mount("/static", StaticFiles(directory="static"), name="static")

EXAMPLES_DIR = Path("examples")
HISTORY_PATH = "review_history.jsonl"

_KEY_CHECK = {
    "OPENAI_API_KEY": bool(OPENAI_API_KEY),
    "ANTHROPIC_API_KEY": bool(ANTHROPIC_API_KEY),
}

ALLOWED_EXTENSIONS = {".md", ".txt", ".html", ".htm", ".docx", ".pdf"}


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse("static/index.html")


@app.get("/api/models")
async def get_models():
    models = []
    for m in MODEL_REGISTRY:
        available = _KEY_CHECK.get(m["api_key_env"], False)
        models.append({
            "id": m["id"],
            "name": m["name"],
            "provider": m["provider"],
            "available": available,
        })
    default = DEFAULT_MODEL
    if not any(m["id"] == default and m["available"] for m in models):
        for m in models:
            if m["available"]:
                default = m["id"]
                break
    return {"models": models, "default": default}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Parse an uploaded file and return extracted text."""
    filename = file.filename or "unknown.txt"
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="File is empty.")

    try:
        content = parse_file(file_bytes, filename)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse file: {e}")

    word_count = len(content.split())
    return {
        "content": content,
        "filename": filename,
        "file_type": ext,
        "word_count": word_count,
        "char_count": len(content),
    }


@app.get("/api/sample-brand")
async def get_sample_brand():
    path = EXAMPLES_DIR / "sample_brand_context.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


@app.get("/api/sample-brief")
async def get_sample_brief():
    path = EXAMPLES_DIR / "sample_brief.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


@app.get("/api/sample-content")
async def get_sample_content():
    path = EXAMPLES_DIR / "sample_content.md"
    if path.exists():
        return {"content": path.read_text()}
    return {"content": ""}


@app.post("/api/review")
async def run_review(
    content: str = Form(...),
    brand_context_json: str = Form(...),
    brief_json: str = Form(""),
    mode: str = Form("full"),
    model: str = Form(""),
):
    try:
        brand_data = json.loads(brand_context_json)
        brand = BrandContext(**brand_data)
    except (json.JSONDecodeError, Exception):
        brand = BrandContext(raw_description=brand_context_json.strip())

    brief = None
    if brief_json and brief_json.strip():
        try:
            brief_data = json.loads(brief_json)
            brief = ContentBrief(**brief_data)
        except (json.JSONDecodeError, Exception):
            brief = ContentBrief(notes=brief_json.strip())

    firewall = ContentFirewall(
        brand_context=brand,
        brief=brief,
        history_path=HISTORY_PATH,
    )

    model_to_use = model if model else DEFAULT_MODEL

    if mode == "quick":
        verdict = await firewall.quick_review(content, model=model_to_use)
    else:
        verdict = await firewall.review(content, model=model_to_use)

    redline_json = firewall.get_redline_json()

    repairs = None
    if mode == "audit" and verdict.decision.value != "pass":
        repairs = await firewall.get_repairs()

    return {
        "decision": verdict.decision.value,
        "overall_score": verdict.overall_score,
        "total_issues": verdict.total_issues,
        "blocker_count": verdict.blocker_count,
        "major_count": verdict.major_count,
        "minor_count": verdict.minor_count,
        "style_count": verdict.style_count,
        "issues": redline_json.get("issues", []),
        "score_breakdown": redline_json.get("score_breakdown", {}),
        "reviewer_summaries": redline_json.get("reviewer_summaries", {}),
        "repairs": repairs,
        "model_used": model_to_use,
    }


@app.get("/api/dashboard")
async def get_dashboard():
    dashboard = RootCauseDashboard(history_path=HISTORY_PATH)
    if not dashboard._history:
        return {"text": "No review history yet. Run some reviews first.", "data": None}

    text = dashboard.generate_dashboard()
    trending = dashboard.get_trending_issues()

    category_counts: dict[str, int] = {}
    total = len(dashboard._history)
    for entry in dashboard._history[-50:]:
        for cat in set(entry.get("categories", [])):
            category_counts[cat] = category_counts.get(cat, 0) + 1

    return {
        "text": text,
        "data": {
            "total_reviews": total,
            "category_rates": {
                k: round(v / min(total, 50) * 100, 1)
                for k, v in sorted(category_counts.items(), key=lambda x: -x[1])
            },
            "trending": trending,
        },
    }


@app.post("/api/competitor-analysis")
async def competitor_analysis(
    url: str = Form(...),
    model: str = Form(""),
):
    """Fetch a competitor URL, extract content, and analyze it with LLM."""
    model_to_use = model if model else DEFAULT_MODEL
    set_model(model_to_use)

    try:
        page_text, word_count = await _fetch_and_extract(url)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to fetch URL: {e}")

    if word_count < 50:
        raise HTTPException(status_code=422, detail="Could not extract meaningful content from URL.")

    max_chars = 12000
    analysis_text = page_text[:max_chars]

    system_prompt = """You are an expert content strategist and SEO analyst. You analyze competitor content that is currently ranking on Google to understand search intent, writing quality, and content patterns.

Return a JSON object with these exact keys:
{
  "search_intent": "What search intent does this content serve? What questions does it answer? What is Google rewarding here? What does the reader expect when they click this? Be specific and actionable.",
  "writing_style": "Analyze the writing style: tone (formal/casual/technical), sentence structure (short/long/mixed), use of data/examples/stories, persuasion techniques, how they handle transitions, paragraph length patterns, use of active vs passive voice. Note specific patterns.",
  "structure_analysis": "How is the content structured? Heading hierarchy, section flow, use of lists/tables/visuals, intro pattern, conclusion pattern, CTA placement. What structural choices make this rank?",
  "ai_analysis": "Is this likely AI-written or human-written? Look for: repetitive sentence openers, generic frameworks, lack of specific examples/data, formulaic transitions, semantic repetition, overly balanced 'on one hand / on the other' patterns, absence of genuine opinion or expertise signals. Give specific evidence.",
  "ai_probability": 0.0 to 1.0,
  "reading_level": "Grade level or complexity descriptor",
  "recommendations": "Based on this analysis, what should the user do differently in their own content to compete? Be specific: what to copy, what to avoid, what gaps to exploit, what angle to take."
}"""

    user_prompt = f"""Analyze this competitor content that is currently ranking on Google.

URL: {url}
Word count: {word_count}

--- CONTENT ---
{analysis_text}
--- END ---

Provide your analysis as JSON."""

    try:
        result = await llm_review(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model_to_use,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM analysis failed: {e}")

    if result.get("parse_error"):
        raise HTTPException(status_code=500, detail="Failed to parse LLM response.")

    result["word_count"] = word_count
    result["url"] = url
    result["model_used"] = model_to_use
    return result


@app.post("/api/generate-content-brief")
async def generate_content_brief(
    competitor_url: str = Form(""),
    competitor_content: str = Form(""),
    competitor_analysis_json: str = Form(""),
    brand_context_text: str = Form(""),
    model: str = Form(""),
):
    """Generate a structured content creation brief from competitor analysis."""
    model_to_use = model if model else DEFAULT_MODEL
    set_model(model_to_use)

    system_prompt = """You are an elite content strategist who creates detailed content creation briefs. You analyze competitor content that is currently ranking and reverse-engineer the exact instructions needed to create superior content.

You MUST return a JSON object with ALL of these keys. Every field must be filled with specific, actionable recommendations — never leave a field empty or say "N/A":

{
  "page_kind": "blog_post" or "landing_page",
  "page_kind_reasoning": "Why this page kind is recommended",

  "content_goal": exactly one of: "educate_inform", "thought_leadership", "drive_organic_traffic", "showcase_results", "compare_evaluate", "generate_leads",
  "content_goal_reasoning": "Why this goal fits best",

  "primary_keyword": "The exact primary keyword/topic to target",
  "primary_keyword_reasoning": "Why this keyword and how the competitor is using it",

  "content_nature": exactly one of: "how_to", "what_is", "listicle", "comparison", "guide", "case_study", "review", "opinion", "alternative",
  "content_nature_reasoning": "Why this content structure fits",

  "headline_recommendations": ["5 specific headline options, ranked by predicted CTR"],

  "secondary_keywords": ["6-8 secondary keywords extracted from competitor + gaps identified"],

  "tonality": exactly one of: "professional", "conversational", "technical", "founder_voice", "authoritative", "friendly_approachable",
  "tonality_reasoning": "Why this tone works for this topic/audience",

  "theme_category": "The theme or category this content belongs to",

  "depth_word_count": exactly one of: "short_500_800", "medium_1000_1500", "long_1500_3000", "deep_dive_3000_plus",
  "depth_reasoning": "Why this word count range based on competitor analysis",

  "reading_format": exactly one of: "narrative", "step_by_step", "qa_format", "hybrid",
  "reading_format_reasoning": "Why this reading format",

  "components": {
    "include_faq": true/false,
    "faq_reasoning": "Why or why not, and suggested FAQ topics if yes",
    "include_table": true/false,
    "table_reasoning": "What tables/comparison matrices to include",
    "include_embeds": true/false,
    "embed_reasoning": "What embeds would strengthen this content",
    "include_proof_block": true/false,
    "proof_reasoning": "What proof elements to include",
    "include_callouts": true/false,
    "callout_reasoning": "What callout types and content",
    "include_ctas": true/false,
    "cta_reasoning": "CTA placement strategy and copy direction",
    "include_images": true/false,
    "image_reasoning": "What images/visuals to include and where"
  },

  "custom_persona": "A specific writing persona description tailored for this content — who the writer should be, their expertise, their voice",

  "custom_instructions": "Detailed paragraph of specific instructions for the content writer. Include: sections to cover, angles to emphasize, competitor gaps to exploit, specific data points to include, things the competitor does well to match, things they do poorly to beat. Be very specific and actionable.",

  "suggested_outline": ["Ordered list of section headings that form the article outline"],

  "competitor_gaps": ["Specific things the competitor missed or did poorly that we should capitalize on"],

  "differentiation_strategy": "How to make this content clearly better and different from the competitor"
}"""

    user_prompt_parts = []

    if competitor_url:
        user_prompt_parts.append(f"Competitor URL: {competitor_url}")

    if competitor_analysis_json:
        try:
            analysis = json.loads(competitor_analysis_json)
            user_prompt_parts.append(f"Competitor Analysis Summary:\n- Search Intent: {analysis.get('search_intent', '')}\n- Writing Style: {analysis.get('writing_style', '')}\n- Structure: {analysis.get('structure_analysis', '')}\n- AI Analysis: {analysis.get('ai_analysis', '')}\n- AI Probability: {analysis.get('ai_probability', '')}\n- Recommendations: {analysis.get('recommendations', '')}")
        except json.JSONDecodeError:
            pass

    if brand_context_text:
        user_prompt_parts.append(f"Brand Context:\n{brand_context_text}")

    if competitor_content:
        max_chars = 10000
        user_prompt_parts.append(f"--- COMPETITOR CONTENT (first {max_chars} chars) ---\n{competitor_content[:max_chars]}\n--- END ---")

    user_prompt = "\n\n".join(user_prompt_parts)
    user_prompt += "\n\nBased on the above, generate a complete content creation brief as JSON. Every field must be filled with specific, actionable recommendations."

    try:
        result = await llm_review(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model_to_use,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM brief generation failed: {e}")

    if result.get("parse_error"):
        raise HTTPException(status_code=500, detail="Failed to parse LLM response for content brief.")

    result["model_used"] = model_to_use
    return result


async def _fetch_and_extract(url: str) -> tuple[str, int]:
    """Fetch a URL and return (extracted_text, word_count)."""
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=30.0,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        },
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "iframe"]):
        tag.decompose()

    paragraphs = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote"]):
        text = el.get_text(strip=True)
        if not text or len(text) < 10:
            continue
        if el.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(el.name[1])
            paragraphs.append(f"{'#' * level} {text}")
        elif el.name == "li":
            paragraphs.append(f"- {text}")
        else:
            paragraphs.append(text)

    page_text = "\n\n".join(paragraphs)
    if not page_text.strip():
        page_text = soup.get_text(separator="\n", strip=True)

    return page_text, len(page_text.split())


@app.post("/api/content-compare")
async def content_compare(
    your_content: str = Form(...),
    competitor_url_1: str = Form(...),
    competitor_url_2: str = Form(""),
    model: str = Form(""),
):
    """Deep comparison of user content vs 1-2 competitor articles."""
    model_to_use = model if model else DEFAULT_MODEL
    set_model(model_to_use)

    import asyncio

    async def safe_fetch(url: str, label: str) -> dict:
        try:
            text, wc = await _fetch_and_extract(url)
            return {"url": url, "text": text, "word_count": wc, "label": label, "error": None}
        except Exception as e:
            return {"url": url, "text": "", "word_count": 0, "label": label, "error": str(e)}

    fetch_tasks = [safe_fetch(competitor_url_1, "Competitor 1")]
    if competitor_url_2:
        fetch_tasks.append(safe_fetch(competitor_url_2, "Competitor 2"))
    fetched = await asyncio.gather(*fetch_tasks)

    for f in fetched:
        if f["error"]:
            raise HTTPException(status_code=422, detail=f"Failed to fetch {f['label']} ({f['url']}): {f['error']}")

    your_wc = len(your_content.split())

    has_two = len(fetched) == 2
    your_max = 30000
    comp_max = 20000 if not has_two else 15000

    competitor_blocks = ""
    for f in fetched:
        competitor_blocks += f"\n\n--- {f['label'].upper()} (URL: {f['url']}, {f['word_count']} words) ---\n{f['text'][:comp_max]}\n--- END {f['label'].upper()} ---"

    system_prompt = f"""You are an elite content analyst performing a deep, section-by-section and line-by-line comparison between a user's content and {'2 competitor articles' if has_two else '1 competitor article'} that are currently ranking on Google.

Your comparison must be brutally honest, specific, and actionable. Do NOT give generic feedback. Reference specific sentences, paragraphs, and sections.

Return a JSON object with this exact structure:
{{
  "articles": [
    {{
      "label": "Your Content",
      "title": "Detected or inferred title",
      "word_count": {your_wc},
      "overall_grade": "A+ to F letter grade"
    }},
    {{
      "label": "Competitor 1",
      "title": "Detected title",
      "word_count": {fetched[0]['word_count']},
      "overall_grade": "A+ to F letter grade"
    }}{(',' + chr(10) + '    { "label": "Competitor 2", "title": "Detected title", "word_count": ' + str(fetched[1]["word_count"]) + ', "overall_grade": "A+ to F letter grade" }') if has_two else ''}
  ],

  "overall_verdict": "A 3-4 sentence executive summary. Where does the user's content stand versus competition? Is it publishable as-is, or does it need work? What is the single biggest gap?",

  "comparative_scores": {{
    "Content Depth & Value": [score_yours_0_to_100, score_comp1, {('score_comp2' if has_two else '')}],
    "Expertise & Authority Signals": [score_yours, score_comp1, {('score_comp2' if has_two else '')}],
    "SEO Optimization": [score_yours, score_comp1, {('score_comp2' if has_two else '')}],
    "Readability & Flow": [score_yours, score_comp1, {('score_comp2' if has_two else '')}],
    "Originality & Unique Value": [score_yours, score_comp1, {('score_comp2' if has_two else '')}],
    "Structure & Formatting": [score_yours, score_comp1, {('score_comp2' if has_two else '')}],
    "Engagement & Hook Quality": [score_yours, score_comp1, {('score_comp2' if has_two else '')}],
    "Actionability": [score_yours, score_comp1, {('score_comp2' if has_two else '')}]
  }},

  "section_comparisons": [
    {{
      "dimension": "Title & Meta",
      "your_analysis": "How your title compares — specific feedback",
      "competitor_1_analysis": "What competitor 1 does",
      {"competitor_2_analysis" if has_two else ""},
      "your_verdict": "Winning" or "Losing" or "Tied",
      "specific_feedback": "Exactly what to change and why"
    }},
    {{
      "dimension": "Introduction / Hook",
      ... same structure ...
    }},
    {{
      "dimension": "Content Depth & Examples",
      ...
    }},
    {{
      "dimension": "Data & Evidence",
      ...
    }},
    {{
      "dimension": "Structure & Headings",
      ...
    }},
    {{
      "dimension": "Writing Quality & Voice",
      ...
    }},
    {{
      "dimension": "SEO Signals",
      ...
    }},
    {{
      "dimension": "Visuals & Formatting",
      ...
    }},
    {{
      "dimension": "Call-to-Action & Conversion",
      ...
    }},
    {{
      "dimension": "Conclusion & Takeaways",
      ...
    }},
    {{
      "dimension": "AI vs Human Quality",
      ...
    }},
    {{
      "dimension": "Unique Value Proposition",
      ...
    }}
  ],

  "action_items": [
    "Specific, prioritized action item 1 — what exactly to change/add/remove and why",
    "Action item 2...",
    "... up to 10 action items, ordered by impact"
  ]
}}

IMPORTANT RULES:
- Every "your_analysis" and "competitor_X_analysis" must reference SPECIFIC content from the articles, not generic statements.
- "your_verdict" must be exactly "Winning", "Losing", or "Tied".
- "specific_feedback" must tell the user EXACTLY what to change — cite the section, the sentence, the issue, and the fix.
- Be harsh and honest. If the user's content is worse, say so clearly.
- All scores are 0-100 integers."""

    user_prompt = f"""Compare these articles in depth.

--- YOUR CONTENT ({your_wc} words) ---
{your_content[:your_max]}
--- END YOUR CONTENT ---
{competitor_blocks}

Perform a deep, section-by-section comparison. Return the complete JSON analysis."""

    try:
        result = await llm_review(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model_to_use,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM comparison failed: {e}")

    if result.get("parse_error"):
        raise HTTPException(status_code=500, detail="Failed to parse LLM comparison response.")

    result["model_used"] = model_to_use
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8500)

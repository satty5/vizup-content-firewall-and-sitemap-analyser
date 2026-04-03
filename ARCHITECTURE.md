# Vizup Soul — Technical Architecture & Feature Documentation

> **Purpose**: This document is a comprehensive reference for production developers converting this prototype into a production-ready feature set. It covers every module, endpoint, data flow, design decision, and known limitation.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Tech Stack](#tech-stack)
3. [Directory Structure](#directory-structure)
4. [Core Architecture](#core-architecture)
5. [Feature 1: Content Quality Firewall](#feature-1-content-quality-firewall)
6. [Feature 2: Competitor / SERP Analysis](#feature-2-competitor--serp-analysis)
7. [Feature 3: Ranking-Aware Comparison Tool](#feature-3-ranking-aware-comparison-tool)
8. [Feature 4: Content Gap Analyzer](#feature-4-content-gap-analyzer)
9. [LLM Infrastructure](#llm-infrastructure)
10. [Content Extraction Pipeline](#content-extraction-pipeline)
11. [Export System](#export-system)
12. [Frontend Architecture](#frontend-architecture)
13. [API Reference](#api-reference)
14. [Data Flows](#data-flows)
15. [Production Considerations](#production-considerations)
16. [Known Limitations](#known-limitations)

---

## System Overview

Vizup Soul is a content intelligence platform with four integrated tools:

1. **Content Quality Firewall** — Pre-publish editorial review with 6 AI review layers
2. **Competitor / SERP Analysis** — Single-URL competitor content analysis with content brief generation
3. **Ranking-Aware Comparison** — Multi-URL analysis that learns from what Google actually ranks
4. **Content Gap Analyzer** — Sitemap-based content gap detection with topic inference and content calendar

All tools share a common LLM gateway, content extraction pipeline, and export infrastructure.

---

## Tech Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| **Backend** | Python 3.12 + FastAPI | Single `app.py` monolith (~3600 lines) |
| **Server** | Uvicorn (ASGI) | Async, handles SSE streaming |
| **Frontend** | Vanilla HTML/CSS/JS | Single `static/index.html` (~3200 lines) |
| **LLM Providers** | OpenAI (GPT-5.2), Anthropic (Claude Opus 4.6, Sonnet 4.6) | Switchable per request |
| **LLM Gateway** | `engine/llm_judge.py` | Unified interface, streaming for Anthropic |
| **HTML Parsing** | BeautifulSoup4 | Content extraction, metadata extraction |
| **XML Parsing** | xml.etree.ElementTree (stdlib) | Sitemap parsing |
| **HTTP Client** | httpx | Async, with browser-like headers |
| **Excel Export** | openpyxl | Multi-sheet, styled workbooks |
| **PDF Export** | fpdf2 | Branded PDF reports |
| **Config** | python-dotenv | `.env` file for API keys |

### Dependencies (`requirements.txt`)

```
openai>=1.14.0
anthropic>=0.42.0
pydantic>=2.6.0
httpx>=0.27.0
rich>=13.7.0
python-dotenv>=1.0.0
tiktoken>=0.6.0
fastapi>=0.115.0
uvicorn>=0.34.0
python-multipart>=0.0.18
python-docx>=1.1.0
pdfplumber>=0.11.0
beautifulsoup4>=4.12.0
openpyxl>=3.1.0
fpdf2>=2.7.0
```

---

## Directory Structure

```
Vizup_Soul/
├── app.py                    # FastAPI application — ALL endpoints + tool logic
├── run.py                    # CLI entry point
├── firewall.py               # ContentFirewall orchestrator
├── config/
│   ├── settings.py           # API keys, model registry, thresholds
│   ├── taxonomy.py           # Issue categories and severity definitions
│   └── banned_phrases.py     # Hardcoded banned phrase lists
├── engine/
│   ├── llm_judge.py          # LLM gateway (OpenAI + Anthropic, streaming)
│   ├── review_orchestrator.py # Concurrent reviewer execution
│   ├── rule_engine.py        # Deterministic pre-publish rules
│   └── scoring.py            # Score aggregation and publish decision
├── models/
│   ├── brand_context.py      # BrandContext and ContentBrief Pydantic models
│   ├── content_unit.py       # ContentUnit model
│   └── review_result.py      # ReviewResult, Issue models
├── reviewers/
│   ├── base_reviewer.py      # Abstract base class for all reviewers
│   ├── context_fidelity.py   # Layer 1: Brief adherence
│   ├── brand_safety.py       # Layer 2: Competitor/brand leakage
│   ├── ai_slop_detector.py   # Layer 3: AI-generated content detection
│   ├── logic_checker.py      # Layer 4: Logic and truth verification
│   ├── editorial_quality.py  # Layer 5: Depth, specificity, POV
│   └── structural_quality.py # Layer 6: Structure quality
├── outputs/
│   ├── redline_report.py     # Line-by-line issue report
│   ├── repair_engine.py      # Targeted fix suggestions
│   └── root_cause.py         # Cross-review pattern dashboard
├── parsers/
│   ├── file_parser.py        # .md, .txt, .html, .docx, .pdf parsing
│   └── content_parser.py     # Content normalization
├── static/
│   └── index.html            # Single-page frontend (HTML + CSS + JS)
├── examples/
│   ├── sample_brand_context.json
│   ├── sample_brief.json
│   └── sample_content.md
├── reports/                  # Generated report JSON files (gitignored)
├── .env.example              # Environment variable template
└── requirements.txt
```

---

## Core Architecture

### Request Flow

```
Browser (index.html)
    │
    ├─ POST /api/review         → ContentFirewall → 6 Reviewers → Verdict
    ├─ POST /api/content-compare → Fetch URLs → Extract Context → LLM Analysis
    ├─ POST /api/content-gap/analyze → SSE Stream → 6 Stages Progressive
    └─ POST /api/competitor-analysis → Fetch URL → LLM Analysis
    │
    ▼
LLM Gateway (engine/llm_judge.py)
    │
    ├─ OpenAI  (GPT-5.2)         — chat.completions.create
    └─ Anthropic (Claude 4.6)    — messages.stream (streaming)
```

### Model Selection

Models are configured in `config/settings.py` via `MODEL_REGISTRY`. Each API request can specify a `model` parameter. The gateway resolves provider automatically from model ID prefix (`gpt*` → OpenAI, `claude*` → Anthropic).

### LLM Response Handling

All LLM calls return JSON. The gateway:
1. Strips markdown code fences (` ```json ... ``` `)
2. Parses JSON
3. On parse failure: returns `{"raw_response": "...", "parse_error": true}`
4. Callers implement JSON repair (close truncated braces) as fallback

---

## Feature 1: Content Quality Firewall

### Purpose
Pre-publish editorial review that catches AI slop, brand violations, logic gaps, and quality issues before content goes live.

### Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/review` | POST | Run full/quick/audit review |
| `/api/upload` | POST | Parse uploaded file (.md, .txt, .html, .docx, .pdf) |
| `/api/fetch-url` | POST | Fetch and extract content from URL |
| `/api/dashboard` | GET | Root cause dashboard from review history |
| `/api/sample-brand` | GET | Sample brand context JSON |
| `/api/sample-brief` | GET | Sample content brief JSON |
| `/api/sample-content` | GET | Sample content text |

### Review Modes

| Mode | What runs | Output |
|------|-----------|--------|
| `full` | Rule engine + all 6 LLM reviewers | Decision + issues + scores |
| `quick` | Rule engine + brand safety + slop detection | Faster, fewer layers |
| `audit` | Full review + repair engine | Full review + targeted fix suggestions |

### The 6 Review Layers

Each reviewer is a class in `reviewers/` inheriting from `BaseReviewer`:

1. **Context Fidelity** (`context_fidelity.py`) — Does content match the brief?
2. **Brand Safety** (`brand_safety.py`) — Competitor mentions, brand dilution (BLOCKER-level)
3. **AI Slop Detection** (`ai_slop_detector.py`) — Template language, empty abstractions, robotic transitions
4. **Logic & Truth** (`logic_checker.py`) — Unsupported claims, contradictions, causal jumps
5. **Editorial Quality** (`editorial_quality.py`) — Specificity, depth, POV, reader value
6. **Structural Quality** (`structural_quality.py`) — Earned vs imposed structure, keyword stuffing

### Scoring

- Each reviewer produces issues with severity (blocker/major/minor/style)
- `engine/scoring.py` aggregates into overall score (0-100)
- Publish decision: PASS (≥80), PASS WITH REVISIONS (≥50), FAIL (<50)
- Blocker count > 0 always → FAIL

### Data Flow

```
POST /api/review
  ├─ Parse brand context JSON → BrandContext model
  ├─ Parse brief JSON → ContentBrief model (optional)
  ├─ Create ContentFirewall(brand, brief)
  ├─ firewall.review(content, model) or firewall.quick_review(content, model)
  │   ├─ Rule engine (deterministic, fast)
  │   └─ 6 LLM reviewers (concurrent via asyncio)
  ├─ Get redline JSON (issues, scores, summaries)
  ├─ If audit mode + not pass: get_repairs()
  └─ Return JSON response
```

---

## Feature 2: Competitor / SERP Analysis

### Purpose
Analyze a single competitor URL to understand its content strategy, then generate a content creation brief.

### Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/competitor-analysis` | POST | Analyze single competitor URL |
| `/api/generate-content-brief` | POST | Generate creation brief from analysis |

### Analysis Output
- Search intent identification
- Writing style analysis
- Content structure breakdown
- AI-written probability score (0-1)
- Reading level assessment
- Actionable recommendations

### Brief Output
Structured JSON with: page kind, content goal, primary keyword, headline options, secondary keywords, tonality, depth, components (FAQ, tables, embeds, CTAs, images), writing persona, custom instructions, suggested outline, competitor gaps, differentiation strategy.

---

## Feature 3: Ranking-Aware Comparison Tool

### Purpose
**Not a checklist SEO audit.** This tool learns from what Google is actually ranking and evaluates your page against those learned preferences.

### Key Design Principle

> The tool studies top-ranking pages as ground truth, extracts Google's preferences, back-calculates ideal ranges, and evaluates the main page against ranking reality — not SEO best practices.

### UI Structure

```
┌─────────────────────────────────────┐
│  YOUR PAGE (to analyze)             │
│  [URL input] [Position dropdown]    │
├─────────────────────────────────────┤
│  COMPETITORS (currently ranking)    │
│  1. [URL] [Position] [×]           │
│  2. [URL] [Position] [×]           │
│  + Add Competitor                   │
├─────────────────────────────────────┤
│  [Analyze Against Rankings]         │
└─────────────────────────────────────┘
```

### Position Buckets

| Value | Label | Meaning |
|-------|-------|---------|
| `top3` | Top 3 | Position 1-3 |
| `top10` | Page 1 | Position 4-10 |
| `page2` | Page 2 | Position 11-20 |
| `page3_5` | Page 3-5 | Position 21-50 |
| `page5plus` | Page 5+ | Position 50+ |
| `not_ranking` | Not ranking | Not in index or beyond page 10 |

### Endpoint

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/content-compare` | POST | Ranking-aware analysis |

### Form Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `main_url` | string | No | Your page URL |
| `main_position` | string | No | Your page's ranking position bucket |
| `competitor_urls` | JSON array | Yes | Competitor page URLs |
| `competitor_positions` | JSON array | No | Position bucket for each competitor |
| `model` | string | No | LLM model to use |

### Analysis Methodology

The prompt (`_build_ranking_analysis_prompt`) instructs the LLM to:

1. **Study ranking pages** — Top-ranking competitors are treated as "what Google wants"
2. **Build a Google Preference Model** — Extract patterns:
   - Preferred word count range
   - Preferred heading density
   - Preferred page type
   - Preferred content style
   - Trust signals that matter
   - What ranking pages have in common
   - What ranking pages do NOT do
3. **Score against ranking reality** — NOT against SEO checklists
4. **Diagnose ranking gaps** — Why the main page ranks where it does

### Scoring Dimensions

| Dimension | What it measures |
|-----------|-----------------|
| Query Intent Match | Does the page give searchers what they need? |
| Content Efficiency | Value per word, not total words |
| Page Type Fit | Does the format match what Google rewards? |
| Trust & Authority Signals | Real trust, not just schema/tags |
| User Decision Support | Can users compare, decide, act? |
| Content Focus vs Bloat | Focused = high, bloated = low |
| Conversion UX | Clear path to action |
| Over-Optimization Risk | Natural = high, over-optimized = low |
| SERP Click Appeal | Title + description click-worthiness |
| Overall Ranking Potential | Holistic ranking prediction |

### Anti-Patterns the Prompt Explicitly Penalizes

- More words ≠ better (8000 words can score lower than 2000)
- More headings ≠ better (33 H2s can be worse than 8)
- More schema ≠ better (triple schema on page-45 = meaningless)
- Unicode tricks in titles are NOT ranking signals
- Over-optimization is a NEGATIVE signal

### Output Schema

```json
{
  "articles": [...],
  "google_preference_model": {
    "query_intent": "...",
    "preferred_page_type": "...",
    "preferred_word_count_range": "...",
    "preferred_heading_density": "...",
    "preferred_content_style": "...",
    "trust_signals_that_matter": "...",
    "what_ranking_pages_have_in_common": "...",
    "what_ranking_pages_do_NOT_do": "..."
  },
  "overall_verdict": "...",
  "comparative_scores": {...},
  "section_comparisons": [...],
  "ranking_gap_diagnosis": {
    "primary_reason": "...",
    "secondary_reasons": [...],
    "site_level_concerns": "...",
    "over_optimization_flags": [...],
    "intent_mismatch_details": "..."
  },
  "action_items": [...]
}
```

### Content Extraction Pipeline

For each URL, the system extracts two layers:

1. **Page Content** (`_fetch_and_extract`) — Cleaned text (headings, paragraphs, lists, tables, code, images)
2. **Page Context** (`_extract_page_context`) — Structural metadata:
   - SERP data: title tag, meta description, URL slug/depth, OG tags, canonical, robots
   - Structure: heading counts/outline, image counts, link counts
   - Linking: internal/external link counts, anchor text samples, nofollow/sponsored
   - Trust: author, date, schema types, TOC, FAQ, social, comments

Both layers are sent to the LLM for analysis.

---

## Feature 4: Content Gap Analyzer

### Purpose
Discover what topics competitors cover that you don't, using intelligent sitemap crawling, LLM-powered topic inference, and a publishable content calendar.

### Key Design Principle

> Results stream progressively via SSE. Users see each stage as it completes — no waiting for the entire pipeline.

### Endpoint

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/content-gap/discover` | POST | Single domain sitemap discovery (preview) |
| `/api/content-gap/analyze` | POST | Full 6-stage pipeline (SSE stream) |
| `/api/content-gap/download-text` | POST | Plain text export |
| `/api/content-gap/download-excel` | POST | 7-sheet Excel export |
| `/api/content-gap/download-pdf` | POST | Branded PDF export |

### The 6 Stages

#### Stage 1: Sitemap Intelligence

**Multi-strategy sitemap discovery:**

```
Strategy 1: robots.txt
  └─ Parse all Sitemap: directives

Strategy 2: CMS-Aware Probing
  ├─ Detect CMS from homepage HTML (WordPress/Yoast/Rank Math/Shopify/Ghost/etc.)
  └─ Probe CMS-specific sitemap paths

Strategy 3: Fallback Common Paths
  └─ /sitemap.xml, /sitemap_index.xml, /sitemap.xml.gz, etc.
```

**Sitemap parsing (`_parse_sitemap`):**
- Handles sitemap index files (recursive, up to depth 3)
- Handles direct URL set files
- Handles gzipped sitemaps
- XML namespace aware (`{http://www.sitemaps.org/schemas/sitemap/0.9}`)
- Extracts: URL, lastmod, changefreq, priority
- Deduplicates across child sitemaps
- No URL cap — processes all pages

**CMS detection (`_detect_cms`):**

| CMS | Detection Signal | Sitemap Paths |
|-----|-----------------|---------------|
| WordPress Core | `wp-content`, `wp-includes` | `/wp-sitemap.xml` |
| WordPress + Yoast | `yoast` in HTML | `/sitemap_index.xml` → `/post-sitemap.xml` |
| WordPress + Rank Math | `rank math` in HTML | `/sitemap_index.xml` |
| Shopify | `cdn.shopify` | `/sitemap.xml` → `sitemap_products_1.xml` |
| Ghost | `ghost` generator meta | `/sitemap.xml` → `sitemap-posts.xml` |
| Squarespace | `squarespace` | `/sitemap.xml` |
| Wix | `wix.com` | `/sitemap.xml` |
| Webflow | `webflow` | `/sitemap.xml` |

**URL classification (`_classify_gap_urls`):**
- Blog posts: `/blog/`, `/news/`, `/articles/`, date patterns, depth ≥ 3
- Landing pages: `/services/`, `/products/`, `/pricing/`, depth ≤ 2
- Other: tags, categories, authors, media, utility pages

**Output:** Domain cards with CMS, discovery method, URL counts by type, freshness stats

#### Stage 2: Site Architecture Profile

**Parallel metadata extraction (`_batch_extract_metadata`):**
- `asyncio.Semaphore(10)` for concurrency control
- Per URL: title, H1, meta description, meta keywords, published date, canonical, word count estimate
- Published date from: `article:published_time`, `<time>` element, JSON-LD `datePublished`
- 15s timeout per URL, failures skipped gracefully

**Architecture profile (`_build_arch_profile`):**
- Publishing cadence: posts per month (12-month trend), frequency classification
- Freshness distribution: pages updated in last 30/90/180/365 days
- Meta quality: average title/description length, missing description %
- URL depth analysis
- Recent content list (last 90 days)

#### Stage 3: Topic & Keyword Landscape (LLM Phase 1)

**`_llm_infer_topics`** — Sends batched metadata (50 pages per LLM call) to infer:
- Topic clusters (5-15 per domain)
- Primary keyword per page
- Search intent per page (informational/transactional/commercial/navigational)
- Volume tier (high/medium/low)
- Content type (blog_post/landing_page/guide/comparison/listicle/how_to)

#### Stage 4 & 5: Gap Analysis & Recommendations (LLM Phase 2)

**`_llm_gap_analysis`** — Compares topic inventories across domains:
- **Gap matrix**: Topics as rows, coverage status per domain
- **Gap types**: missing (no page), thin (weak coverage), outdated (stale)
- **Opportunity score**: 1-10, combining volume + gap type + competitor coverage
- **Recommendations**: 10-15 specific content pieces with title, keyword, intent, outline
- **Quick wins**: Low-effort, fast-impact actions
- **Strategic moves**: High-effort, high-impact plays

#### Stage 6: Content Calendar (LLM Phase 3)

**`_llm_content_calendar`** — User-configurable:

| Parameter | Options |
|-----------|---------|
| Frequency | Daily, 3x/week, 2x/week, Weekly, Bi-weekly, Monthly |
| Horizon | 1, 3, 6, or 12 months |
| Start date | Any date (defaults to next Monday) |

**Calendar features:**
- Week-by-week schedule with specific publish dates
- Pillar pages scheduled before cluster posts (dependencies)
- Related topics grouped into content sprints
- Monthly themes
- Milestone markers
- Per entry: date, title, type, keyword, intent, priority, effort, brief points

### SSE Streaming Architecture

The `/api/content-gap/analyze` endpoint returns a `StreamingResponse` with `text/event-stream` content type.

**Server side:**
```python
async def _stream():
    # Stage 1
    stage_1_data = await do_stage_1()
    yield f"event: stage_1\ndata: {json.dumps(stage_1_data)}\n\n"

    # Stage 2
    stage_2_data = await do_stage_2()
    yield f"event: stage_2\ndata: {json.dumps(stage_2_data)}\n\n"

    # ... stages 3-6 ...

    yield f"event: complete\ndata: {json.dumps(final_meta)}\n\n"

return StreamingResponse(_stream(), media_type="text/event-stream")
```

**Client side:**
```javascript
const response = await fetch('/api/content-gap/analyze', { method: 'POST', body: fd });
const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = '';

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // Parse SSE events, render each stage panel immediately
}
```

**SSE Events:**

| Event | Data | Frontend Action |
|-------|------|----------------|
| `stage_1` | Sitemap intelligence | Render sitemap panel, advance stepper |
| `stage_2` | Architecture profiles | Render architecture panel |
| `stage_3` | Topic clusters | Render topic landscape panel |
| `stage_4_5` | Gap matrix + recommendations | Render gaps + action plan panels |
| `stage_6` | Content calendar | Render calendar panel |
| `complete` | Report ID, metadata | Enable download buttons |

---

## LLM Infrastructure

### File: `engine/llm_judge.py`

**Key function:** `llm_review(system_prompt, user_prompt, model, temperature, max_tokens)`

| Parameter | Default | Notes |
|-----------|---------|-------|
| `model` | From `settings.py` | Can be overridden per call |
| `temperature` | `REVIEW_TEMPERATURE` (0.1) | Low for deterministic JSON |
| `max_tokens` | 16384 | Dynamically scaled for large outputs |

**Provider dispatch:**
- OpenAI: `client.chat.completions.create()` with `max_completion_tokens`
- Anthropic: `client.messages.stream()` (streaming to avoid 10-min timeout)

**Token scaling for comparisons:**
```python
token_budget = 8192 + (num_pages * 3000)
token_budget = min(token_budget, 32768)
```

**JSON repair fallback:**
When LLM output is truncated JSON, the system attempts repair by closing unclosed braces/brackets before raising an error.

### Why Anthropic Uses Streaming

The Anthropic SDK raises a timeout error for non-streaming requests that take longer than 10 minutes. Large comparison prompts (5+ URLs with page context + content) can exceed this. Streaming (`client.messages.stream()`) keeps the connection alive indefinitely.

---

## Content Extraction Pipeline

### `_fetch_and_extract(url)` → `(text, word_count)`

Extracts clean, readable text from any URL:

1. Fetch with browser-like headers (Chrome UA, Accept, Sec-Fetch-*)
2. Strip `<script>`, `<style>`, `<noscript>`, `<iframe>`, `<svg>`
3. Find content root (priority order):
   - `<article>`, `<main>`, `div[role=main]`
   - Class-based content divs (`.post-content`, `.entry-content`, etc.)
   - Largest paragraph-rich block
   - `<body>` fallback
4. Clean noise: nav, footer, aside, sidebar, related posts, social, ads
5. Walk DOM: headings → `# H1`, paragraphs, lists → `- item`, tables → markdown, code → fenced
6. Post-process: remove affiliate lines, boilerplate patterns
7. Deduplicate repeated text blocks

### `_extract_page_context(html, url)` → `dict`

Extracts structural/SEO metadata (not content text):

- **SERP**: title (with length check), meta description (with length check), URL slug/depth, OG tags, canonical, robots
- **Structure**: heading counts (H1-H6) with outline, image counts (total + with alt)
- **Links**: internal count, external count, nofollow/sponsored counts, top external domains, anchor text samples
- **Trust**: author presence, publish date, schema markup types, TOC, FAQ, social share, comments
- **Rich content**: lists, tables, blockquotes, code blocks, videos

---

## Export System

### Comparison Reports

| Format | Endpoint | Features |
|--------|----------|----------|
| Text | `/api/comparison-report/download-text` | Formatted plain text |
| Excel | `/api/comparison-report/download-excel` | 4-sheet workbook: Overview, Scores, Section Analysis, Action Items |
| PDF | `/api/comparison-report/download-pdf` | Branded PDF with cover page, score tables, section analysis, action items |

### Content Gap Reports

| Format | Endpoint | Features |
|--------|----------|----------|
| Text | `/api/content-gap/download-text` | All 6 stages as formatted text |
| Excel | `/api/content-gap/download-excel` | 7-sheet workbook: Summary, Sitemap, Architecture, Topics, Gaps, Recommendations, Calendar |
| PDF | `/api/content-gap/download-pdf` | Branded PDF with all 6 stages |

### PDF Generation Notes
- Uses `fpdf2` with core Helvetica fonts
- `_sanitize_for_pdf()` converts Unicode to ASCII-safe equivalents (em dash → --, smart quotes → straight quotes, etc.) to prevent core font crashes
- Color palette: purple (#6D28D9) primary, with green/orange/red for scores

---

## Frontend Architecture

### Single Page Application

`static/index.html` — ~3200 lines of inline HTML + CSS + JS. No build tools, no frameworks.

### Layout

```
┌──────────────────────────────────────────────┐
│  Header: Vizup Soul | Load Sample | Dashboard│
├──────────────────┬───────────────────────────┤
│  LEFT PANEL      │  RIGHT PANEL              │
│                  │                            │
│  Mode Selector   │  Tabs:                     │
│  Content Input   │  Issues | Scores |         │
│  Brand Context   │  Summaries | Repairs |     │
│  Content Brief   │  Compare | Content Gap     │
│  Competitor      │                            │
│  Model Selector  │  (tab content)             │
│  [Run Review]    │                            │
└──────────────────┴───────────────────────────┘
```

### Tab System

| Tab | data-tab | Content |
|-----|----------|---------|
| Issues | `issues` | Filter bar + expandable issue cards |
| Scores | `scores` | Score grid with bar fills |
| Summaries | `summaries` | Per-reviewer summary cards |
| Repairs | `repairs` | Repair suggestion cards |
| Compare | `compare` | Main URL + competitors + ranking analysis |
| Content Gap | `content-gap` | Domain inputs + 6 streaming insight panels |

### Key JavaScript Functions

| Function | Purpose |
|----------|---------|
| `runReview()` | Execute content review via `/api/review` |
| `runComparison()` | Execute ranking analysis via `/api/content-compare` |
| `runGapAnalysis()` | Execute gap analysis via SSE stream |
| `discoverSitemap()` | Preview sitemap discovery for a domain |
| `renderComparison(data)` | Render comparison results |
| `_handleGapSSE(event, data)` | Handle SSE events, render stages incrementally |
| `_renderGapStage1..6()` | Individual stage rendering functions |

### CSS Design System

- Dark theme with CSS variables (`:root`)
- Purple accent (`--accent: #6c5ce7`)
- Component classes: `.btn`, `.issue-card`, `.score-grid`, `.gap-panel`, `.gap-table`, etc.
- Responsive: single column below 900px

---

## API Reference

### Content Review

```
POST /api/review
  content: string (required)
  brand_context_json: string (required)
  brief_json: string (optional)
  mode: "full" | "quick" | "audit" (default: "full")
  model: string (optional)
```

### Content Compare (Ranking-Aware)

```
POST /api/content-compare
  main_url: string (optional — your page URL)
  main_position: string (optional — position bucket)
  competitor_urls: JSON array of strings (required)
  competitor_positions: JSON array of strings (optional)
  model: string (optional)
```

### Content Gap Analyze (SSE Stream)

```
POST /api/content-gap/analyze
  your_domain: string (optional)
  your_urls: JSON array (optional — manual URL list)
  competitor_domains: JSON array (required — 1-5 domains)
  model: string (optional)
  publishing_frequency: string (default: "weekly")
  calendar_horizon: string (default: "3")
  calendar_start_date: string (optional — YYYY-MM-DD)

Response: text/event-stream (SSE)
```

### Content Gap Discover (Preview)

```
POST /api/content-gap/discover
  domain: string (required)

Response: JSON with stage_1_sitemap data
```

---

## Data Flows

### Comparison Flow

```
User enters Main URL + Competitor URLs + Positions
    │
    ▼
Frontend collects data, sends POST /api/content-compare
    │
    ▼
Backend fetches all URLs in parallel (asyncio.gather)
  ├─ _fetch_page_html(url) → raw HTML
  ├─ _fetch_and_extract(url) → clean text + word count
  └─ _extract_page_context(html, url) → structural metadata
    │
    ▼
Build prompt with page blocks (content + context + position)
    │
    ▼
_build_ranking_analysis_prompt() → system + user prompt
    │
    ▼
llm_review(prompts, max_tokens=dynamic) → JSON result
    │
    ▼
JSON repair if needed → save report → return to frontend
    │
    ▼
renderComparison(data) → display results
```

### Content Gap Flow (SSE)

```
User enters domains + calendar settings
    │
    ▼
POST /api/content-gap/analyze → StreamingResponse
    │
    ├─ Stage 1: _discover_sitemaps() + _parse_sitemap() → yield SSE
    ├─ Stage 2: _batch_extract_metadata() → yield SSE
    ├─ Stage 3: _llm_infer_topics() per domain → yield SSE
    ├─ Stage 4+5: _llm_gap_analysis() → yield SSE
    ├─ Stage 6: _llm_content_calendar() → yield SSE
    └─ Complete: save report → yield SSE
    │
    ▼
Frontend: _handleGapSSE() → render each stage panel incrementally
```

---

## Production Considerations

### 1. Split the Monolith

`app.py` is ~3600 lines. For production:

```
app/
├── main.py                 # FastAPI app + router registration
├── routers/
│   ├── review.py           # /api/review, /api/upload, /api/dashboard
│   ├── compare.py          # /api/content-compare
│   ├── gap_analyzer.py     # /api/content-gap/*
│   └── exports.py          # /api/*/download-*
├── services/
│   ├── content_extractor.py    # _fetch_and_extract, _extract_page_context
│   ├── sitemap_crawler.py      # _discover_sitemaps, _parse_sitemap
│   ├── metadata_extractor.py   # _batch_extract_metadata
│   └── llm_prompts.py          # All prompt builders
└── utils/
    ├── pdf_generator.py
    ├── excel_generator.py
    └── json_repair.py
```

### 2. Background Job Queue

The content gap analyzer runs 3-4 LLM calls + hundreds of HTTP fetches. For production:
- Use Celery/Redis or similar task queue
- Store results in database (PostgreSQL)
- WebSocket or polling for progress updates (instead of SSE on POST)
- Job timeout handling and retry logic

### 3. Database

Currently uses:
- `review_history.jsonl` for review history
- `reports/*.json` for saved reports
- In-memory for everything else

For production: PostgreSQL with tables for reviews, reports, gap analyses, content calendars.

### 4. Caching

- Cache sitemap discovery results (sitemaps don't change hourly)
- Cache metadata extraction (page titles/descriptions are stable)
- Cache LLM results with content hash keys
- TTL: sitemaps 24h, metadata 6h, LLM results 1h

### 5. Rate Limiting

- LLM API rate limits (OpenAI, Anthropic)
- Target website rate limiting (currently `asyncio.Semaphore(10)`)
- Consider adding per-domain delay (0.5s between requests to same host)
- User-level rate limiting for the API

### 6. Authentication

Currently no auth. For production:
- API key or JWT authentication
- User accounts with usage quotas
- Team/workspace support

### 7. Frontend Framework

Currently vanilla HTML/CSS/JS. For production:
- React/Next.js or Vue/Nuxt
- Component library for consistency
- State management for complex data flows
- Proper error boundaries

### 8. Error Handling

Current error handling is basic (`try/except → HTTPException`). For production:
- Structured error responses
- Sentry or similar error tracking
- Graceful degradation (partial results on partial failures)
- User-friendly error messages

### 9. Observability

- Structured logging (JSON format)
- Request tracing (correlation IDs)
- LLM call metrics (latency, token usage, cost)
- Sitemap crawl metrics (URLs found, failures, timing)

---

## Known Limitations

### Content Extraction
- JavaScript-rendered content (SPAs) is not extracted — BeautifulSoup only sees server-rendered HTML
- Some sites block scraping despite browser-like headers
- Very large pages (100K+ chars) are truncated for LLM context limits

### Sitemap Discovery
- Sites without sitemaps cannot be analyzed (no crawler fallback)
- Password-protected or login-required sitemaps are skipped
- Some CDN/WAF configurations block sitemap requests

### LLM Analysis
- Topic inference quality depends on metadata quality (missing titles/descriptions reduce accuracy)
- Gap analysis is inference-based, not keyword-volume-based (no external keyword API)
- Content calendar is suggestive, not data-driven (no seasonal trend data)
- Large prompts (5+ competitors) can approach model context limits

### Comparison Tool
- Position data is user-provided (not verified against actual SERPs)
- Cannot detect site-level trust issues without broader crawl data
- Backlink data is not available (would need Ahrefs/Moz API)
- User interaction signals (CTR, bounce rate, dwell time) are not available

### Exports
- PDF uses core fonts (Helvetica) — Unicode characters are replaced with ASCII equivalents
- Excel styling is functional but not pixel-perfect
- Very large reports may produce large file sizes

---

## Environment Variables

```bash
# Required — at least one provider
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Model selection
OPENAI_MODEL=gpt-5.2
ANTHROPIC_MODEL=claude-sonnet-4-6
ANTHROPIC_MODEL_OPUS=claude-opus-4-6
ANTHROPIC_MODEL_SONNET=claude-sonnet-4-6
DEFAULT_PROVIDER=openai
DEFAULT_MODEL=gpt-5.2

# Tuning
MAX_CONCURRENT_REVIEWERS=6
REVIEW_TEMPERATURE=0.1
```

---

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start the server
python app.py
# → http://localhost:8500

# Or use the CLI
python run.py review content.md --brand brand.json
```

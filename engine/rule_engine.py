"""
Deterministic rule engine — fast, cheap, reliable first pass.
Catches banned phrases, competitor names, structural anomalies,
repeated n-grams, and regex-based risky patterns.
"""

from __future__ import annotations

import re
from collections import Counter
from models.content_unit import ContentUnit, ParsedContent, UnitType
from models.review_result import ReviewIssue
from models.brand_context import BrandContext
from config.taxonomy import IssueCategory, Severity
from config.banned_phrases import (
    AI_SLOP_PHRASES,
    ROBOTIC_TRANSITIONS,
    GENERIC_TRIADS,
    EMPTY_ABSTRACTIONS,
)


def _compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


_SLOP_PATTERNS = _compile_patterns(AI_SLOP_PHRASES)
_TRANSITION_PATTERNS = _compile_patterns(ROBOTIC_TRANSITIONS)
_TRIAD_PATTERNS = _compile_patterns(GENERIC_TRIADS)
_ABSTRACTION_PATTERNS = _compile_patterns(EMPTY_ABSTRACTIONS)


def check_banned_phrases(unit: ContentUnit) -> list[ReviewIssue]:
    issues: list[ReviewIssue] = []
    text = unit.text

    for pat in _SLOP_PATTERNS:
        if pat.search(text):
            issues.append(ReviewIssue(
                line_number=unit.line_number,
                line_text=text,
                section=unit.parent_heading,
                category=IssueCategory.AI_SLOP,
                severity=Severity.MAJOR,
                explanation=f"Matched AI slop pattern: {pat.pattern}",
                suggested_fix="Remove or rewrite with specific, concrete language.",
                confidence=0.95,
                reviewer="rule_engine",
            ))

    for pat in _TRANSITION_PATTERNS:
        if pat.search(text):
            issues.append(ReviewIssue(
                line_number=unit.line_number,
                line_text=text,
                section=unit.parent_heading,
                category=IssueCategory.ROBOTIC_TRANSITION,
                severity=Severity.MINOR,
                explanation=f"Matched robotic transition: {pat.pattern}",
                suggested_fix="Rewrite transition naturally or remove if unnecessary.",
                confidence=0.90,
                reviewer="rule_engine",
            ))

    for pat in _TRIAD_PATTERNS:
        if pat.search(text):
            issues.append(ReviewIssue(
                line_number=unit.line_number,
                line_text=text,
                section=unit.parent_heading,
                category=IssueCategory.GENERIC_FRAMEWORK,
                severity=Severity.MAJOR,
                explanation=f"Generic triad detected: {pat.pattern}",
                suggested_fix="Replace with specific, evidence-backed attributes relevant to the topic.",
                confidence=0.90,
                reviewer="rule_engine",
            ))

    for pat in _ABSTRACTION_PATTERNS:
        if pat.search(text):
            issues.append(ReviewIssue(
                line_number=unit.line_number,
                line_text=text,
                section=unit.parent_heading,
                category=IssueCategory.EMPTY_ABSTRACTION,
                severity=Severity.MAJOR,
                explanation=f"Empty abstraction detected: {pat.pattern}",
                suggested_fix="Delete or replace with a concrete, factual statement.",
                confidence=0.90,
                reviewer="rule_engine",
            ))

    return issues


def check_competitor_mentions(
    unit: ContentUnit,
    brand_context: BrandContext,
) -> list[ReviewIssue]:
    issues: list[ReviewIssue] = []
    banned = brand_context.get_competitor_names()
    text_lower = unit.text.lower()

    for name in banned:
        pattern = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
        if pattern.search(unit.text):
            issues.append(ReviewIssue(
                line_number=unit.line_number,
                line_text=unit.text,
                section=unit.parent_heading,
                category=IssueCategory.COMPETITOR_PROMOTION,
                severity=Severity.BLOCKER,
                explanation=f"Unauthorized competitor mention: '{name}' found in content.",
                suggested_fix=f"Remove mention of '{name}' or replace with brand-owned language.",
                confidence=1.0,
                reviewer="rule_engine",
            ))

    return issues


def check_semantic_repetition(
    parsed: ParsedContent,
    ngram_size: int = 4,
    threshold: int = 3,
) -> list[ReviewIssue]:
    """Detect repeated n-grams across the entire content."""
    issues: list[ReviewIssue] = []
    paragraphs = [u for u in parsed.all_units if u.unit_type == UnitType.PARAGRAPH]

    all_words: list[str] = []
    for p in paragraphs:
        all_words.extend(p.text.lower().split())

    if len(all_words) < ngram_size:
        return issues

    ngram_counter: Counter[str] = Counter()
    for i in range(len(all_words) - ngram_size + 1):
        ngram = " ".join(all_words[i : i + ngram_size])
        ngram_counter[ngram] += 1

    repeated = {ng: count for ng, count in ngram_counter.items() if count >= threshold}

    for ngram, count in list(repeated.items())[:10]:
        for p in paragraphs:
            if ngram in p.text.lower():
                issues.append(ReviewIssue(
                    line_number=p.line_number,
                    line_text=p.text[:120],
                    section=p.parent_heading,
                    category=IssueCategory.SEMANTIC_REPETITION,
                    severity=Severity.MINOR,
                    explanation=f"Phrase '{ngram}' repeated {count} times across content.",
                    suggested_fix="Vary language or consolidate the repeated idea.",
                    confidence=0.85,
                    reviewer="rule_engine",
                ))
                break

    return issues


def check_structural_rules(parsed: ParsedContent) -> list[ReviewIssue]:
    """Deterministic structural checks."""
    issues: list[ReviewIssue] = []

    hierarchy = parsed.heading_hierarchy
    if hierarchy:
        prev_level = 0
        for level, text in hierarchy:
            if prev_level > 0 and level > prev_level + 1:
                issues.append(ReviewIssue(
                    line_number=0,
                    line_text=text,
                    category=IssueCategory.HEADING_HIERARCHY_BROKEN,
                    severity=Severity.MINOR,
                    explanation=f"Heading '{text}' (H{level}) skips from H{prev_level} — hierarchy broken.",
                    suggested_fix="Fix heading levels to maintain proper hierarchy.",
                    confidence=1.0,
                    reviewer="rule_engine",
                ))
            prev_level = level

    for unit in parsed.all_units:
        if unit.unit_type == UnitType.PARAGRAPH and unit.word_count > 150:
            issues.append(ReviewIssue(
                line_number=unit.line_number,
                line_text=unit.text[:100] + "...",
                section=unit.parent_heading,
                category=IssueCategory.OVERLONG_PARAGRAPH,
                severity=Severity.MINOR,
                explanation=f"Paragraph is {unit.word_count} words — too long for readability.",
                suggested_fix="Break into 2-3 shorter paragraphs.",
                confidence=0.95,
                reviewer="rule_engine",
            ))

    for section in parsed.sections:
        if len(section.list_items) > 10:
            heading_text = section.heading.text if section.heading else "Untitled"
            issues.append(ReviewIssue(
                line_number=section.heading.line_number if section.heading else 0,
                line_text=heading_text,
                section=heading_text,
                category=IssueCategory.LIST_BLOAT,
                severity=Severity.MINOR,
                explanation=f"Section '{heading_text}' has {len(section.list_items)} list items — list bloat.",
                suggested_fix="Reduce to 5-7 most important items or group into sub-sections.",
                confidence=0.90,
                reviewer="rule_engine",
            ))

    return issues


def run_rule_engine(
    parsed: ParsedContent,
    brand_context: BrandContext,
) -> list[ReviewIssue]:
    """Run all deterministic checks. Fast first pass."""
    all_issues: list[ReviewIssue] = []

    for unit in parsed.all_units:
        if unit.unit_type in {UnitType.SENTENCE, UnitType.PARAGRAPH, UnitType.LIST_ITEM, UnitType.TITLE, UnitType.META_TITLE, UnitType.META_DESCRIPTION}:
            all_issues.extend(check_banned_phrases(unit))
            all_issues.extend(check_competitor_mentions(unit, brand_context))

    all_issues.extend(check_semantic_repetition(parsed))
    all_issues.extend(check_structural_rules(parsed))

    return all_issues

"""
Content parser — decomposes raw draft text into reviewable units.
Handles markdown-style content with headings, paragraphs, lists, FAQs.
"""

from __future__ import annotations

import re
from models.content_unit import (
    ContentUnit,
    ContentSection,
    ParsedContent,
    UnitType,
)


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_LIST_ITEM_RE = re.compile(r"^[\s]*[-*+]\s+(.+)$", re.MULTILINE)
_NUMBERED_LIST_RE = re.compile(r"^[\s]*\d+[.)]\s+(.+)$", re.MULTILINE)
_SENTENCE_SPLIT_RE = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z"\u201c])|(?<=[.!?])$', re.MULTILINE
)
_BLOCKQUOTE_RE = re.compile(r"^>\s+(.+)$", re.MULTILINE)
_META_TITLE_RE = re.compile(r"^(?:meta\s*title|title\s*tag)\s*[:]\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_META_DESC_RE = re.compile(r"^(?:meta\s*description)\s*[:]\s*(.+)$", re.IGNORECASE | re.MULTILINE)


def _heading_level(marker: str) -> UnitType:
    level = len(marker)
    return {
        1: UnitType.HEADING_H1,
        2: UnitType.HEADING_H2,
        3: UnitType.HEADING_H3,
    }.get(level, UnitType.HEADING_H4)


def _split_sentences(text: str) -> list[str]:
    raw = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in raw if s.strip()]


def _line_number_of(full_text: str, substring: str, start_search: int = 0) -> int:
    idx = full_text.find(substring, start_search)
    if idx == -1:
        return 0
    return full_text[:idx].count("\n") + 1


def parse_content(
    raw_text: str,
    title: str = "",
    meta_title: str = "",
    meta_description: str = "",
) -> ParsedContent:
    """Parse raw markdown/text content into structured reviewable units."""

    all_units: list[ContentUnit] = []
    heading_hierarchy: list[tuple[int, str]] = []
    lines = raw_text.split("\n")

    title_unit = None
    if title:
        title_unit = ContentUnit(
            unit_type=UnitType.TITLE,
            text=title,
            line_number=0,
        )
        all_units.append(title_unit)

    meta_title_unit = None
    if meta_title:
        meta_title_unit = ContentUnit(
            unit_type=UnitType.META_TITLE,
            text=meta_title,
            line_number=0,
        )
        all_units.append(meta_title_unit)
    else:
        m = _META_TITLE_RE.search(raw_text)
        if m:
            meta_title_unit = ContentUnit(
                unit_type=UnitType.META_TITLE,
                text=m.group(1).strip(),
                line_number=_line_number_of(raw_text, m.group(0)),
            )
            all_units.append(meta_title_unit)

    meta_desc_unit = None
    if meta_description:
        meta_desc_unit = ContentUnit(
            unit_type=UnitType.META_DESCRIPTION,
            text=meta_description,
            line_number=0,
        )
        all_units.append(meta_desc_unit)
    else:
        m = _META_DESC_RE.search(raw_text)
        if m:
            meta_desc_unit = ContentUnit(
                unit_type=UnitType.META_DESCRIPTION,
                text=m.group(1).strip(),
                line_number=_line_number_of(raw_text, m.group(0)),
            )
            all_units.append(meta_desc_unit)

    sections: list[ContentSection] = []
    current_section = ContentSection()
    current_heading_text = ""
    section_idx = 0

    for line_idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue

        heading_match = _HEADING_RE.match(stripped)
        if heading_match:
            if current_section.heading or current_section.paragraphs:
                sections.append(current_section)
                section_idx += 1

            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            unit_type = _heading_level(heading_match.group(1))

            heading_unit = ContentUnit(
                unit_type=unit_type,
                text=heading_text,
                line_number=line_idx,
                section_index=section_idx,
            )
            all_units.append(heading_unit)
            heading_hierarchy.append((level, heading_text))
            current_heading_text = heading_text

            current_section = ContentSection(heading=heading_unit)
            continue

        blockquote_match = _BLOCKQUOTE_RE.match(stripped)
        if blockquote_match:
            unit = ContentUnit(
                unit_type=UnitType.BLOCKQUOTE,
                text=blockquote_match.group(1).strip(),
                line_number=line_idx,
                section_index=section_idx,
                parent_heading=current_heading_text,
            )
            all_units.append(unit)
            continue

        list_match = _LIST_ITEM_RE.match(stripped) or _NUMBERED_LIST_RE.match(stripped)
        if list_match:
            item_text = list_match.group(1).strip()
            unit = ContentUnit(
                unit_type=UnitType.LIST_ITEM,
                text=item_text,
                line_number=line_idx,
                section_index=section_idx,
                parent_heading=current_heading_text,
            )
            current_section.list_items.append(unit)
            all_units.append(unit)
            continue

        para_unit = ContentUnit(
            unit_type=UnitType.PARAGRAPH,
            text=stripped,
            line_number=line_idx,
            section_index=section_idx,
            parent_heading=current_heading_text,
        )
        current_section.paragraphs.append(para_unit)
        all_units.append(para_unit)

        sentences = _split_sentences(stripped)
        for sent in sentences:
            sent_unit = ContentUnit(
                unit_type=UnitType.SENTENCE,
                text=sent,
                line_number=line_idx,
                section_index=section_idx,
                parent_heading=current_heading_text,
            )
            current_section.sentences.append(sent_unit)
            all_units.append(sent_unit)

    if current_section.heading or current_section.paragraphs or current_section.list_items:
        sections.append(current_section)

    total_paragraphs = sum(
        1 for u in all_units if u.unit_type == UnitType.PARAGRAPH
    )
    total_sentences = sum(
        1 for u in all_units if u.unit_type == UnitType.SENTENCE
    )
    total_headings = sum(
        1 for u in all_units
        if u.unit_type in {UnitType.HEADING_H1, UnitType.HEADING_H2, UnitType.HEADING_H3, UnitType.HEADING_H4}
    )
    total_words = sum(
        u.word_count for u in all_units if u.unit_type == UnitType.PARAGRAPH
    )

    return ParsedContent(
        raw_text=raw_text,
        title=title_unit,
        meta_title=meta_title_unit,
        meta_description=meta_desc_unit,
        sections=sections,
        all_units=all_units,
        total_words=total_words,
        total_sentences=total_sentences,
        total_paragraphs=total_paragraphs,
        total_headings=total_headings,
        heading_hierarchy=heading_hierarchy,
    )

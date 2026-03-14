"""
Content units — the atomic pieces the firewall reviews.
Every draft is decomposed into these before any reviewer touches it.
"""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class UnitType(str, Enum):
    TITLE = "title"
    META_TITLE = "meta_title"
    META_DESCRIPTION = "meta_description"
    HEADING_H1 = "h1"
    HEADING_H2 = "h2"
    HEADING_H3 = "h3"
    HEADING_H4 = "h4"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    LIST_ITEM = "list_item"
    BLOCKQUOTE = "blockquote"
    FAQ_QUESTION = "faq_question"
    FAQ_ANSWER = "faq_answer"
    CTA = "cta"
    IMAGE_ALT = "image_alt"
    CAPTION = "caption"


class ContentUnit(BaseModel):
    """A single reviewable unit of content."""

    unit_type: UnitType
    text: str
    line_number: int = 0
    section_index: int = 0
    parent_heading: str = ""
    word_count: int = 0
    char_count: int = 0

    def model_post_init(self, __context: object) -> None:
        if not self.word_count:
            self.word_count = len(self.text.split())
        if not self.char_count:
            self.char_count = len(self.text)


class ContentSection(BaseModel):
    """A section under a heading, containing its child units."""

    heading: ContentUnit | None = None
    paragraphs: list[ContentUnit] = Field(default_factory=list)
    sentences: list[ContentUnit] = Field(default_factory=list)
    list_items: list[ContentUnit] = Field(default_factory=list)
    sub_sections: list[ContentSection] = Field(default_factory=list)


class ParsedContent(BaseModel):
    """Fully decomposed content ready for review."""

    raw_text: str
    title: ContentUnit | None = None
    meta_title: ContentUnit | None = None
    meta_description: ContentUnit | None = None
    sections: list[ContentSection] = Field(default_factory=list)
    all_units: list[ContentUnit] = Field(default_factory=list)
    total_words: int = 0
    total_sentences: int = 0
    total_paragraphs: int = 0
    total_headings: int = 0
    heading_hierarchy: list[tuple[int, str]] = Field(default_factory=list)

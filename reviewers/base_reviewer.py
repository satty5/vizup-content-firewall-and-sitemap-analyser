"""
Base reviewer — abstract contract all reviewers implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from models.content_unit import ParsedContent
from models.review_result import ReviewerOutput
from models.brand_context import BrandContext, ContentBrief


class BaseReviewer(ABC):
    """Every reviewer inherits this and implements `review`."""

    name: str = "base"

    @abstractmethod
    async def review(
        self,
        parsed: ParsedContent,
        brand_context: BrandContext,
        brief: ContentBrief | None = None,
    ) -> ReviewerOutput:
        ...

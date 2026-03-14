"""
Review result models — what every reviewer outputs.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from config.taxonomy import IssueCategory, Severity, PublishDecision


class ReviewIssue(BaseModel):
    """A single flagged issue from any reviewer."""

    line_number: int = 0
    line_text: str = ""
    section: str = ""
    category: IssueCategory
    severity: Severity
    explanation: str
    suggested_fix: str = ""
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    reviewer: str = ""


class ReviewerOutput(BaseModel):
    """Output from a single reviewer pass."""

    reviewer_name: str
    issues: list[ReviewIssue] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    summary: str = ""
    passed: bool = True

    @property
    def blocker_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.BLOCKER)

    @property
    def major_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.MAJOR)


class FirewallVerdict(BaseModel):
    """Final output of the entire firewall."""

    decision: PublishDecision
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
    total_issues: int = 0
    blocker_count: int = 0
    major_count: int = 0
    minor_count: int = 0
    style_count: int = 0
    reviewer_outputs: list[ReviewerOutput] = Field(default_factory=list)
    all_issues: list[ReviewIssue] = Field(default_factory=list)
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    summary: str = ""

    @property
    def blockers(self) -> list[ReviewIssue]:
        return [i for i in self.all_issues if i.severity == Severity.BLOCKER]

    @property
    def majors(self) -> list[ReviewIssue]:
        return [i for i in self.all_issues if i.severity == Severity.MAJOR]

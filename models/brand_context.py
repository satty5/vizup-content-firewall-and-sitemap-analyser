"""
Brand context pack — the reviewer's memory of truth.
Without this, the reviewer cannot reliably detect drift.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CompetitorPolicy(BaseModel):
    name: str
    mention_allowed: bool = False
    comparison_allowed: bool = False
    notes: str = ""


class BrandContext(BaseModel):
    """Everything the firewall needs to know about the brand to judge content."""

    company_name: str = ""
    product_name: str = ""
    product_description: str = ""
    what_product_does: str = ""

    raw_description: str = ""

    icp_description: str = ""
    icp_company_size: str = ""
    icp_industry: list[str] = Field(default_factory=list)
    icp_job_titles: list[str] = Field(default_factory=list)
    icp_pain_points: list[str] = Field(default_factory=list)
    icp_geography: list[str] = Field(default_factory=list)

    brand_voice: str = ""
    brand_personality: list[str] = Field(default_factory=list)
    messaging_pillars: list[str] = Field(default_factory=list)
    tagline: str = ""
    mission: str = ""
    approved_claims: list[str] = Field(default_factory=list)
    no_go_claims: list[str] = Field(default_factory=list)
    proof_points: list[str] = Field(default_factory=list)

    competitors: list[CompetitorPolicy] = Field(default_factory=list)
    banned_competitor_names: list[str] = Field(default_factory=list)

    writing_style: str = ""
    tone_guidelines: str = ""
    dos: list[str] = Field(default_factory=list)
    donts: list[str] = Field(default_factory=list)

    funnel_stage: str = ""
    target_action: str = ""
    author_pov: str = ""

    def get_competitor_names(self) -> list[str]:
        names = list(self.banned_competitor_names)
        for c in self.competitors:
            if not c.mention_allowed:
                names.append(c.name)
        return list(set(names))

    def to_context_string(self) -> str:
        if self.raw_description and not self.product_name and not self.what_product_does:
            header = f"Company: {self.company_name}\n" if self.company_name else ""
            return f"{header}Brand Context (provided as text):\n{self.raw_description}"

        parts = [
            f"Company: {self.company_name}" if self.company_name else "",
            f"Product: {self.product_name}" if self.product_name else "",
            f"What it does: {self.what_product_does}" if self.what_product_does else "",
            f"ICP: {self.icp_description}" if self.icp_description else "",
            f"ICP company size: {self.icp_company_size}" if self.icp_company_size else "",
            f"ICP industries: {', '.join(self.icp_industry)}" if self.icp_industry else "",
            f"ICP job titles: {', '.join(self.icp_job_titles)}" if self.icp_job_titles else "",
            f"ICP pain points: {', '.join(self.icp_pain_points)}" if self.icp_pain_points else "",
            f"ICP geography: {', '.join(self.icp_geography)}" if self.icp_geography else "",
            f"Brand voice: {self.brand_voice}" if self.brand_voice else "",
            f"Messaging pillars: {', '.join(self.messaging_pillars)}" if self.messaging_pillars else "",
            f"Tagline: {self.tagline}" if self.tagline else "",
            f"Mission: {self.mission}" if self.mission else "",
            f"Approved claims: {'; '.join(self.approved_claims)}" if self.approved_claims else "",
            f"No-go claims: {'; '.join(self.no_go_claims)}" if self.no_go_claims else "",
            f"Proof points: {'; '.join(self.proof_points)}" if self.proof_points else "",
            f"Writing style: {self.writing_style}" if self.writing_style else "",
            f"DOs: {'; '.join(self.dos)}" if self.dos else "",
            f"DON'Ts: {'; '.join(self.donts)}" if self.donts else "",
            f"Funnel stage: {self.funnel_stage}" if self.funnel_stage else "",
            f"Target action: {self.target_action}" if self.target_action else "",
            f"Author POV: {self.author_pov}" if self.author_pov else "",
        ]
        if self.raw_description:
            parts.append(f"Additional context: {self.raw_description}")
        return "\n".join(p for p in parts if p)


class ContentBrief(BaseModel):
    """The input brief the content was generated from."""

    target_keyword: str = ""
    secondary_keywords: list[str] = Field(default_factory=list)
    topic: str = ""
    content_type: str = ""
    funnel_stage: str = ""
    target_audience: str = ""
    intent: str = ""
    angle: str = ""
    required_sections: list[str] = Field(default_factory=list)
    required_mentions: list[str] = Field(default_factory=list)
    word_count_target: int = 0
    notes: str = ""

    def to_context_string(self) -> str:
        parts = [
            f"Target keyword: {self.target_keyword}" if self.target_keyword else "",
            f"Secondary keywords: {', '.join(self.secondary_keywords)}" if self.secondary_keywords else "",
            f"Topic: {self.topic}" if self.topic else "",
            f"Content type: {self.content_type}" if self.content_type else "",
            f"Funnel stage: {self.funnel_stage}" if self.funnel_stage else "",
            f"Target audience: {self.target_audience}" if self.target_audience else "",
            f"Intent: {self.intent}" if self.intent else "",
            f"Angle: {self.angle}" if self.angle else "",
            f"Required sections: {', '.join(self.required_sections)}" if self.required_sections else "",
            f"Required mentions: {', '.join(self.required_mentions)}" if self.required_mentions else "",
            f"Word count target: {self.word_count_target}" if self.word_count_target else "",
            f"Notes: {self.notes}" if self.notes else "",
        ]
        return "\n".join(p for p in parts if p)

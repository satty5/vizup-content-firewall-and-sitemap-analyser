"""
Content Quality Firewall — Main entry point.
The generator should not decide if content is good.
This system decides if content is publishable.

Usage:
    from firewall import ContentFirewall

    firewall = ContentFirewall(brand_context=brand, brief=brief)
    verdict = await firewall.review(content, model="claude-opus-4-6")
    print(verdict.decision)  # pass | pass_with_revisions | fail
"""

from __future__ import annotations

import asyncio
from typing import Callable

from models.content_unit import ParsedContent
from models.review_result import FirewallVerdict
from models.brand_context import BrandContext, ContentBrief
from parsers.content_parser import parse_content
from engine.llm_judge import set_model, set_provider
from engine.review_orchestrator import run_full_review, run_quick_review
from outputs.redline_report import generate_redline_report, generate_redline_json
from outputs.repair_engine import generate_repairs, format_repairs_report
from outputs.root_cause import RootCauseDashboard


OnProgress = Callable[[str, str], None]


class ContentFirewall:
    """
    The Content Quality Firewall.

    Sits after generation, before publish.
    Every draft passes through 6 review layers + deterministic rules.
    Produces: publish decision, redline report, repair suggestions, root cause tracking.
    """

    def __init__(
        self,
        brand_context: BrandContext,
        brief: ContentBrief | None = None,
        history_path: str = "review_history.jsonl",
        on_progress: OnProgress | None = None,
    ):
        self.brand_context = brand_context
        self.brief = brief
        self.on_progress = on_progress
        self.dashboard = RootCauseDashboard(history_path=history_path)
        self._last_verdict: FirewallVerdict | None = None
        self._last_parsed: ParsedContent | None = None

    def _configure_model(self, model: str | None = None, provider: str | None = None) -> None:
        """Set the active model. Provider is inferred from the model ID."""
        if model:
            set_model(model)
        elif provider:
            set_provider(provider)

    async def review(
        self,
        content: str,
        title: str = "",
        meta_title: str = "",
        meta_description: str = "",
        content_id: str = "",
        skip_reviewers: list[str] | None = None,
        model: str | None = None,
        provider: str | None = None,
    ) -> FirewallVerdict:
        """
        Run the full firewall review on a piece of content.

        Args:
            model: Model ID (e.g. "gpt-5-2", "claude-opus-4-6", "claude-sonnet-4-6").
                   Provider is inferred automatically.
        """
        self._configure_model(model, provider)

        self._last_parsed = parse_content(
            raw_text=content,
            title=title,
            meta_title=meta_title,
            meta_description=meta_description,
        )

        self._last_verdict = await run_full_review(
            parsed=self._last_parsed,
            brand_context=self.brand_context,
            brief=self.brief,
            on_progress=self.on_progress,
            skip_reviewers=skip_reviewers,
        )

        self.dashboard.record_review(self._last_verdict, content_id=content_id)

        return self._last_verdict

    async def quick_review(
        self,
        content: str,
        title: str = "",
        meta_title: str = "",
        meta_description: str = "",
        model: str | None = None,
        provider: str | None = None,
    ) -> FirewallVerdict:
        """Quick pre-screen: rules + brand safety + slop detection only."""
        self._configure_model(model, provider)

        self._last_parsed = parse_content(
            raw_text=content,
            title=title,
            meta_title=meta_title,
            meta_description=meta_description,
        )

        self._last_verdict = await run_quick_review(
            parsed=self._last_parsed,
            brand_context=self.brand_context,
            brief=self.brief,
            on_progress=self.on_progress,
        )

        return self._last_verdict

    def get_redline_report(self) -> str:
        if not self._last_verdict:
            return "No review has been run yet."
        return generate_redline_report(self._last_verdict)

    def get_redline_json(self) -> dict:
        if not self._last_verdict:
            return {"error": "No review has been run yet."}
        return generate_redline_json(self._last_verdict)

    async def get_repairs(self) -> dict:
        if not self._last_verdict:
            return {"error": "No review has been run yet."}
        return await generate_repairs(
            self._last_verdict,
            self.brand_context,
            self.brief,
        )

    async def get_repairs_report(self) -> str:
        repairs = await self.get_repairs()
        return format_repairs_report(repairs)

    def get_dashboard(self, last_n: int = 50) -> str:
        return self.dashboard.generate_dashboard(last_n=last_n)

    async def full_audit(
        self,
        content: str,
        title: str = "",
        meta_title: str = "",
        meta_description: str = "",
        content_id: str = "",
        include_repairs: bool = True,
        model: str | None = None,
        provider: str | None = None,
    ) -> dict:
        """Run complete audit: review + redline + repairs."""
        verdict = await self.review(
            content=content,
            title=title,
            meta_title=meta_title,
            meta_description=meta_description,
            content_id=content_id,
            model=model,
            provider=provider,
        )

        result = {
            "decision": verdict.decision.value,
            "overall_score": verdict.overall_score,
            "redline_report": self.get_redline_report(),
            "redline_json": self.get_redline_json(),
            "verdict": verdict,
        }

        if include_repairs and verdict.decision.value != "pass":
            result["repairs"] = await self.get_repairs()
            result["repairs_report"] = format_repairs_report(result["repairs"])

        return result

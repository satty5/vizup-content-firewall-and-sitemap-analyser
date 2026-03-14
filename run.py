"""
CLI runner for the Content Quality Firewall.

Usage:
    # Review a file
    python run.py review content.md --brand brand_context.json

    # Review with brief
    python run.py review content.md --brand brand.json --brief brief.json

    # Quick scan (rules + brand safety + slop only)
    python run.py quick content.md --brand brand.json

    # Full audit with repairs
    python run.py audit content.md --brand brand.json

    # Show root cause dashboard
    python run.py dashboard

    # Review from stdin
    cat content.md | python run.py review --brand brand.json --stdin
"""

from __future__ import annotations

import asyncio
import json
import sys
import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from models.brand_context import BrandContext, ContentBrief
from firewall import ContentFirewall


console = Console()


def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _load_brand_context(path: str) -> BrandContext:
    data = _load_json(path)
    return BrandContext(**data)


def _load_brief(path: str) -> ContentBrief:
    data = _load_json(path)
    return ContentBrief(**data)


def _load_content(path: str | None, from_stdin: bool = False) -> str:
    if from_stdin:
        return sys.stdin.read()
    if path:
        return Path(path).read_text(encoding="utf-8")
    console.print("[red]Error: No content provided. Use a file path or --stdin[/red]")
    sys.exit(1)


def _on_progress(stage: str, detail: str) -> None:
    color = {
        "rule_engine": "yellow",
        "context_fidelity": "cyan",
        "brand_safety": "red",
        "ai_slop_detector": "magenta",
        "logic_checker": "blue",
        "editorial_quality": "green",
        "structural_quality": "white",
        "llm_reviewers": "bold cyan",
        "complete": "bold green",
    }.get(stage, "white")
    console.print(f"  [{color}][{stage}][/{color}] {detail}")


async def cmd_review(args: argparse.Namespace) -> None:
    brand = _load_brand_context(args.brand)
    brief = _load_brief(args.brief) if args.brief else None
    content = _load_content(args.content, args.stdin)

    console.print(Panel("Content Quality Firewall — Full Review", style="bold blue"))
    console.print(f"  Content length: {len(content)} chars / ~{len(content.split())} words")
    console.print(f"  Brand: {brand.company_name}")
    if brief:
        console.print(f"  Topic: {brief.topic}")
        console.print(f"  Keyword: {brief.target_keyword}")
    console.print()

    firewall = ContentFirewall(
        brand_context=brand,
        brief=brief,
        on_progress=_on_progress,
    )

    verdict = await firewall.review(content)

    console.print()
    console.print(firewall.get_redline_report())

    if args.output:
        output_path = Path(args.output)
        report_data = firewall.get_redline_json()
        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2)
        console.print(f"\n  Report saved to: {output_path}")


async def cmd_quick(args: argparse.Namespace) -> None:
    brand = _load_brand_context(args.brand)
    brief = _load_brief(args.brief) if args.brief else None
    content = _load_content(args.content, args.stdin)

    console.print(Panel("Content Quality Firewall — Quick Scan", style="bold yellow"))

    firewall = ContentFirewall(
        brand_context=brand,
        brief=brief,
        on_progress=_on_progress,
    )

    verdict = await firewall.quick_review(content)
    console.print()
    console.print(firewall.get_redline_report())


async def cmd_audit(args: argparse.Namespace) -> None:
    brand = _load_brand_context(args.brand)
    brief = _load_brief(args.brief) if args.brief else None
    content = _load_content(args.content, args.stdin)

    console.print(Panel("Content Quality Firewall — Full Audit + Repairs", style="bold magenta"))
    console.print(f"  Content: ~{len(content.split())} words")
    console.print()

    firewall = ContentFirewall(
        brand_context=brand,
        brief=brief,
        on_progress=_on_progress,
    )

    result = await firewall.full_audit(content)

    console.print(result["redline_report"])

    if "repairs_report" in result:
        console.print()
        console.print(result["repairs_report"])

    if args.output:
        output_path = Path(args.output)
        output_data = {
            "decision": result["decision"],
            "overall_score": result["overall_score"],
            "redline": result["redline_json"],
        }
        if "repairs" in result:
            output_data["repairs"] = result["repairs"]
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        console.print(f"\n  Full audit saved to: {output_path}")


async def cmd_dashboard(args: argparse.Namespace) -> None:
    from outputs.root_cause import RootCauseDashboard

    dashboard = RootCauseDashboard(history_path=args.history or "review_history.jsonl")
    console.print(dashboard.generate_dashboard(last_n=args.last_n or 50))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Content Quality Firewall — Editorial Intelligence Layer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    review_parser = subparsers.add_parser("review", help="Full review of content")
    review_parser.add_argument("content", nargs="?", help="Path to content file")
    review_parser.add_argument("--brand", required=True, help="Path to brand context JSON")
    review_parser.add_argument("--brief", help="Path to content brief JSON")
    review_parser.add_argument("--output", "-o", help="Save JSON report to file")
    review_parser.add_argument("--stdin", action="store_true", help="Read content from stdin")

    quick_parser = subparsers.add_parser("quick", help="Quick scan (rules + safety + slop)")
    quick_parser.add_argument("content", nargs="?", help="Path to content file")
    quick_parser.add_argument("--brand", required=True, help="Path to brand context JSON")
    quick_parser.add_argument("--brief", help="Path to content brief JSON")
    quick_parser.add_argument("--stdin", action="store_true", help="Read content from stdin")

    audit_parser = subparsers.add_parser("audit", help="Full audit with repair suggestions")
    audit_parser.add_argument("content", nargs="?", help="Path to content file")
    audit_parser.add_argument("--brand", required=True, help="Path to brand context JSON")
    audit_parser.add_argument("--brief", help="Path to content brief JSON")
    audit_parser.add_argument("--output", "-o", help="Save full audit JSON to file")
    audit_parser.add_argument("--stdin", action="store_true", help="Read content from stdin")

    dash_parser = subparsers.add_parser("dashboard", help="Root cause dashboard")
    dash_parser.add_argument("--history", help="Path to review history file")
    dash_parser.add_argument("--last-n", type=int, default=50, help="Number of recent reviews")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    commands = {
        "review": cmd_review,
        "quick": cmd_quick,
        "audit": cmd_audit,
        "dashboard": cmd_dashboard,
    }

    asyncio.run(commands[args.command](args))


if __name__ == "__main__":
    main()

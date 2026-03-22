#!/usr/bin/env python3
"""Lumen — Autonomous Visual Audit Agent CLI (Rich impact table + optional capture)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import structlog
from dotenv import load_dotenv
from google import genai
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from engine.auditor import AuditGraph
from engine.vision import AutonomousObserver

console = Console()


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def _require_api_key() -> str:
    load_dotenv()
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        console.print(
            Panel(
                "Set [bold]GOOGLE_API_KEY[/bold] in the environment or a [bold].env[/bold] file.",
                title="Missing API key",
                border_style="red",
            )
        )
        raise SystemExit(1)
    return key


def cmd_audit(args: argparse.Namespace) -> None:
    api_key = _require_api_key()
    client = genai.Client(api_key=api_key)

    baseline = Path(args.baseline).read_bytes()
    current = Path(args.current).read_bytes()
    diff_text = Path(args.diff).read_text(encoding="utf-8", errors="replace")

    graph = AuditGraph(client)
    report = graph.run(
        baseline,
        current,
        diff_text,
        draw_annotated_report=not args.no_annotate,
    )

    console.print(Rule("[bold cyan]Lumen — Impact Table[/bold cyan]", style="cyan"))

    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Component", ratio=1)
    table.add_column("Expected change", ratio=2)
    table.add_column("Actual result", ratio=2)
    table.add_column("Risk", justify="center")
    table.add_column("Drift", justify="center")
    table.add_column("Root cause (heuristic)", ratio=2)

    for row in report.impact_rows:
        rc = row.root_cause
        if rc:
            rc_cell = (
                f"[dim]{rc.suggested_path}[/dim]\n"
                f"{rc.css_or_selector_hint or '—'}\n"
                f"[italic]{rc.line_hint or 'line ?'}[/italic]"
            )
        else:
            rc_cell = "—"
        row_style = "red" if row.risk_level in ("high", "critical") else None
        table.add_row(
            row.component,
            row.expected_change,
            row.actual_result,
            row.risk_level.upper(),
            row.drift,
            rc_cell,
            style=row_style,
        )

    console.print(table)

    if report.reconciliation.critical_drifts:
        console.print(
            Panel(
                "\n".join(report.reconciliation.critical_drifts),
                title="[bold red]Critical architectural drift[/bold red]",
                border_style="red",
            )
        )

    if report.annotated_current_png and args.annotate_out:
        out_p = Path(args.annotate_out)
        out_p.write_bytes(report.annotated_current_png)
        console.print(f"[dim]Annotated screenshot written to {out_p.resolve()}[/dim]")


def cmd_observe(args: argparse.Namespace) -> None:
    api_key = _require_api_key()
    client = genai.Client(api_key=api_key)
    obs = AutonomousObserver(client, headless=not args.headful)

    png, resolved = obs.capture_screenshot(
        args.url,
        selector=args.selector,
        full_page=args.full_page,
    )
    out = Path(args.out)
    out.write_bytes(png)
    console.print(f"[green]Screenshot[/green] {resolved} → [bold]{out.resolve()}[/bold]")

    if args.vcot:
        vc = obs.visual_chain_of_thought(
            png,
            task_context=args.task or "Describe UI health and rendering fidelity.",
        )
        console.print(Panel(vc.scan.model_dump_json(), title="Scan", border_style="blue"))
        console.print(
            Panel(vc.hypothesis.model_dump_json(), title="Hypothesis", border_style="yellow")
        )
        console.print(Panel(vc.verify.model_dump_json(), title="Verify", border_style="green"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lumen",
        description="Lumen: Autonomous Visual Audit Agent (Gemini + Playwright).",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Debug trace logging.")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("audit", help="Run AuditGraph: baseline vs current PNG + git diff.")
    a.add_argument("--baseline", required=True, type=str, help="Path to baseline screenshot PNG.")
    a.add_argument("--current", required=True, type=str, help="Path to current screenshot PNG.")
    a.add_argument("--diff", required=True, type=str, help="Path to git diff file.")
    a.add_argument(
        "--annotate-out",
        type=str,
        default="lumen_annotated_current.png",
        help="Where to save bbox-overlay PNG (skipped with --no-annotate).",
    )
    a.add_argument(
        "--no-annotate",
        action="store_true",
        help="Skip drawing Gemini boxes on the current screenshot.",
    )
    a.set_defaults(func=cmd_audit)

    o = sub.add_parser("observe", help="Resolve URL (port probe), capture screenshot, optional V-CoT.")
    o.add_argument("--url", required=True, help="Target URL (e.g. http://localhost:3000).")
    o.add_argument("--out", default="lumen_capture.png", help="Output PNG path.")
    o.add_argument("--selector", default=None, help="Optional Playwright selector (falls back to body).")
    o.add_argument("--full-page", action="store_true", help="Use full-page capture on final fallback.")
    o.add_argument("--headful", action="store_true", help="Run browser with UI (debug).")
    o.add_argument(
        "--vcot",
        action="store_true",
        help="Run Visual Chain-of-Thought on the captured image.",
    )
    o.add_argument(
        "--task",
        default="",
        help="Task context string for V-CoT when --vcot is set.",
    )
    o.set_defaults(func=cmd_observe)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)
    try:
        args.func(args)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

"""Build a static UPS demo report from light experiment summaries."""

import argparse
import glob
import shutil
import subprocess
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ups.eval.demo_scorecard import (
    collect_scorecard,
    render_scorecard_html,
    write_scorecard_json,
    write_scorecard_tsv,
)
from ups.eval.demo_plots import write_scorecard_plots


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _summary_paths(inputs: list[str], patterns: list[str]) -> list[Path]:
    paths = [Path(item) for item in inputs]
    for pattern in patterns:
        if Path(pattern).is_absolute():
            paths.extend(Path(item) for item in glob.glob(pattern))
        else:
            paths.extend(Path().glob(pattern))
    unique = sorted({path.resolve() for path in paths if path.exists()})
    if not unique:
        raise SystemExit("No summary files found")
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a static demo report")
    parser.add_argument("summaries", nargs="*", help="summary.json files")
    parser.add_argument("--glob", action="append", default=[], help="Glob pattern for summary files")
    parser.add_argument("--output-dir", default="reports/demo/latest")
    parser.add_argument("--title", default="UPS Demo Scorecard")
    parser.add_argument("--data-manifest", default="")
    parser.add_argument("--commit", default=None, help="Commit SHA to record; defaults to current HEAD")
    parser.add_argument("--promotion-rule", action="append", default=[])
    parser.add_argument("--baseline-run", default="", help="Run name to compare every row against")
    parser.add_argument("--baseline-metric", default="", help="Metric for baseline comparison; defaults to row main metric")
    parser.add_argument("--baseline-min-improvement", type=float, default=0.2)
    parser.add_argument(
        "--cost-json",
        action="append",
        default=[],
        help="Optional cost.json files keyed by run_name or summary_json",
    )
    parser.add_argument("--copy-summaries", action="store_true", help="Copy input summaries into output-dir/summaries")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _summary_paths(args.summaries, args.glob)
    scorecard = collect_scorecard(
        paths,
        data_manifest=args.data_manifest or None,
        commit=args.commit if args.commit is not None else _git_commit(),
        promotion_rules=args.promotion_rule,
        cost_paths=args.cost_json,
        baseline_run_name=args.baseline_run,
        baseline_metric_name=args.baseline_metric,
        baseline_min_improvement=args.baseline_min_improvement,
    )

    write_scorecard_tsv(scorecard, output_dir / "metrics.tsv")
    write_scorecard_json(scorecard, output_dir / "scorecard.json")
    plots = write_scorecard_plots(scorecard, output_dir)
    (output_dir / "index.html").write_text(
        render_scorecard_html(scorecard, title=args.title, plots=plots),
        encoding="utf-8",
    )

    if args.copy_summaries:
        summaries_dir = output_dir / "summaries"
        summaries_dir.mkdir(exist_ok=True)
        for path in paths:
            run_name = path.parent.name
            shutil.copy2(path, summaries_dir / f"{run_name}.summary.json")

    print(output_dir / "index.html")
    print(output_dir / "metrics.tsv")
    print(output_dir / "scorecard.json")


if __name__ == "__main__":
    main()

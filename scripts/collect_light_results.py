#!/usr/bin/env python
from __future__ import annotations

"""Collect UPS light experiment summaries into comparable scorecard artifacts."""

import argparse
import glob
import subprocess
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ups.eval.demo_scorecard import collect_scorecard, write_scorecard_json, write_scorecard_tsv


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
    parser = argparse.ArgumentParser(description="Collect light experiment summary.json files")
    parser.add_argument("summaries", nargs="*", help="summary.json files")
    parser.add_argument("--glob", action="append", default=[], help="Glob pattern for summary files")
    parser.add_argument("--output-tsv", default="reports/demo/metrics.tsv")
    parser.add_argument("--output-json", default="reports/demo/scorecard.json")
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
    args = parser.parse_args()

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
    write_scorecard_tsv(scorecard, args.output_tsv)
    write_scorecard_json(scorecard, args.output_json)
    print(f"wrote {args.output_tsv}")
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()

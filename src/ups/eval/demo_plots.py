from __future__ import annotations

"""Plot helpers for UPS demo reports."""

from pathlib import Path
from typing import Any

from ups.eval.demo_scorecard import Scorecard


def _numeric(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_scorecard_plots(scorecard: Scorecard, output_dir: str | Path) -> dict[str, str]:
    """Write compact scorecard plots and return label -> relative path."""

    import matplotlib.pyplot as plt

    out_dir = Path(output_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plots: dict[str, str] = {}

    run_names = [str(row.get("run_name", "")) for row in scorecard.rows]
    if not run_names:
        return plots

    for metric_key, label in (
        ("metric:decoded_rollout_nrmse", "decoded_rollout_nrmse"),
        ("metric:decoded_step1_nrmse", "decoded_step1_nrmse"),
        ("main_metric_value", "main_metric"),
    ):
        values = [_numeric(row.get(metric_key)) for row in scorecard.rows]
        if not any(value is not None for value in values):
            continue
        plot_values = [float("nan") if value is None else value for value in values]
        fig_width = max(6.0, min(14.0, 1.4 * len(run_names)))
        fig, ax = plt.subplots(figsize=(fig_width, 4.0), constrained_layout=True)
        ax.bar(run_names, plot_values, color="#34699a")
        ax.set_title(label)
        ax.set_ylabel("lower is better")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
        path = plots_dir / f"{label}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots[label] = str(path.relative_to(out_dir))
    return plots


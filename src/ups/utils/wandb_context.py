"""Centralized WandB context management for clean run lifecycle.

This module provides a single, clean interface for WandB logging that ensures:
- One WandB run per pipeline execution
- Proper data types (summary for scalars, log for time series)
- No hacky mode switching or run proliferation
- Easy testing via dependency injection
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


@dataclass
class WandBContext:
    """Centralized WandB run context - passed to all components.

    This context is created once by the orchestrator and passed to training,
    evaluation, and other components. This ensures all data goes to a single
    WandB run with proper organization.

    Attributes:
        run: The wandb.Run object (or None if disabled)
        run_id: Unique identifier for this run
        enabled: Whether WandB logging is enabled
    """

    run: Any  # wandb.Run object
    run_id: str
    enabled: bool = True

    def log_training_metric(
        self, stage: str, metric: str, value: float, step: int
    ) -> None:
        """Log training metrics as time series.

        Use this for per-epoch/per-step metrics that should appear as line charts.

        IMPORTANT: This logs BOTH the metric AND the step metric. WandB requires
        the step metric to exist as data for define_metric() to work correctly.
        Without logging the step metric, charts will appear empty!

        Args:
            stage: Training stage (operator, diffusion, consistency)
            metric: Metric name (loss, lr, grad_norm, etc.)
            value: Metric value
            step: Global step number for x-axis

        Example:
            ctx.log_training_metric("operator", "loss", 0.001, step=100)
            # Creates: training/operator/loss = 0.001 at step 100
            #          training/operator/step = 100 (for x-axis)
        """
        if not self.enabled or self.run is None:
            return

        key = f"training/{stage}/{metric}"
        step_key = f"training/{stage}/step"

        try:
            # Log both the metric AND the step metric for proper charting
            # WandB needs the step metric to exist as data for define_metric() to work
            # IMPORTANT: Do NOT pass step= parameter when using custom step metrics!
            # See: https://github.com/wandb/examples/blob/master/colabs/wandb-log/Customize_metric_logging_with_define_metric.ipynb
            self.run.log({key: value, step_key: step})
        except Exception:
            pass

    def log_eval_summary(
        self, metrics: Dict[str, float], prefix: str = "eval"
    ) -> None:
        """Log evaluation metrics as summary (single values, not time series).

        Use this for final evaluation results. These appear in the Summary tab,
        NOT as line charts. Perfect for single-point metrics like final NRMSE.

        Args:
            metrics: Dictionary of metric name -> value
            prefix: Namespace prefix (default: "eval")

        Example:
            ctx.log_eval_summary({"nrmse": 0.09, "mse": 0.001}, prefix="eval/baseline")
            # Creates summary values (not charts!):
            # eval/baseline/nrmse = 0.09
            # eval/baseline/mse = 0.001
        """
        if not self.enabled or self.run is None:
            return

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                summary_key = f"{prefix}/{key}"
                try:
                    self.run.summary[summary_key] = value
                except Exception:
                    pass

    def log_table(
        self, name: str, columns: List[str], data: List[List[Any]]
    ) -> None:
        """Log a table for multi-value comparisons.

        Use this for side-by-side comparisons, leaderboard entries, or any
        structured multi-column data.

        Args:
            name: Table name (will appear in WandB UI)
            columns: List of column names
            data: List of rows (each row is a list of values)

        Example:
            ctx.log_table(
                "eval/baseline_vs_ttc",
                columns=["Metric", "Baseline", "TTC", "Improvement"],
                data=[
                    ["NRMSE", 0.78, 0.09, "88.5%"],
                    ["MSE", 0.99, 0.01, "99.0%"],
                ]
            )
        """
        if not self.enabled or self.run is None or wandb is None:
            return

        try:
            table = wandb.Table(columns=columns, data=data)
            self.run.log({name: table}, commit=False)
        except Exception:
            pass

    def log_image(self, name: str, image_path: Path) -> None:
        """Log an image visualization.

        Args:
            name: Image name (will appear in WandB UI)
            image_path: Path to image file

        Example:
            ctx.log_image("eval/mse_histogram", Path("reports/mse_hist.png"))
        """
        if not self.enabled or self.run is None or wandb is None:
            return

        try:
            self.run.log({name: wandb.Image(str(image_path))}, commit=False)
        except Exception:
            pass

    def save_file(self, file_path: Path) -> None:
        """Save a file to WandB (checkpoints, configs, reports, etc.).

        Args:
            file_path: Path to file to upload

        Example:
            ctx.save_file(Path("checkpoints/operator.pt"))
        """
        if not self.enabled or self.run is None or wandb is None:
            return

        try:
            wandb.save(str(file_path), base_path=str(file_path.parent.parent))
        except Exception:
            pass

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update run config (for metadata, not metrics).

        Use this for hyperparameters and metadata that characterize the run.
        Do NOT use this for metrics (use log_eval_summary instead).

        Args:
            updates: Dictionary of config key -> value

        Example:
            ctx.update_config({
                "eval_samples": 3072,
                "eval_tau": 0.5,
                "eval_ttc_enabled": True
            })
        """
        if not self.enabled or self.run is None:
            return

        try:
            self.run.config.update(updates, allow_val_change=True)
        except Exception:
            pass

    def add_tags(self, tags: List[str]) -> None:
        """Add tags to run for organization/filtering.

        Args:
            tags: List of tag strings

        Example:
            ctx.add_tags(["fast-to-sota", "production", "baseline"])
        """
        if not self.enabled or self.run is None:
            return

        try:
            existing = list(self.run.tags) if self.run.tags else []
            self.run.tags = tuple(set(existing + tags))
        except Exception:
            pass

    def log_artifact(self, artifact: Any) -> None:
        """Log a WandB artifact (collection of related files).

        Args:
            artifact: wandb.Artifact object

        Example:
            artifact = wandb.Artifact("summary", type="fast_to_sota")
            artifact.add_file("summary.json")
            ctx.log_artifact(artifact)
        """
        if not self.enabled or self.run is None or wandb is None:
            return

        try:
            self.run.log_artifact(artifact)
        except Exception:
            pass

    def alert(self, title: str, text: str, level: str = "INFO") -> None:
        """Send a WandB alert (appears in dashboard notifications).

        Args:
            title: Alert title
            text: Alert message
            level: Alert level ("INFO", "WARN", "ERROR")

        Example:
            ctx.alert(
                "Performance Regression",
                "NRMSE increased by 15%",
                level="WARN"
            )
        """
        if not self.enabled or self.run is None or wandb is None:
            return

        try:
            alert_level = getattr(wandb.AlertLevel, level, wandb.AlertLevel.INFO)
            wandb.alert(title=title, text=text, level=alert_level)
        except Exception:
            pass

    def finish(self) -> None:
        """Finish the WandB run (call at end of pipeline)."""
        if not self.enabled or self.run is None:
            return

        try:
            self.run.finish()
        except Exception:
            pass


def create_wandb_context(
    config: Dict[str, Any],
    run_id: str,
    mode: str = "online",
) -> Optional[WandBContext]:
    """Create a single WandB run context for the entire pipeline.

    This should be called ONCE by the orchestrator at the start of a pipeline.
    The returned context is then passed to training, evaluation, and other
    components to ensure all data goes to the same WandB run.

    Args:
        config: Full training configuration (hyperparameters will be logged)
        run_id: Unique run identifier
        mode: WandB mode (online, offline, disabled)

    Returns:
        WandBContext object or None if WandB disabled/unavailable

    Example:
        ctx = create_wandb_context(cfg, run_id="pipeline-123", mode="online")
        train(cfg, wandb_ctx=ctx)
        evaluate(cfg, wandb_ctx=ctx)
        ctx.finish()
    """
    if not WANDB_AVAILABLE or mode == "disabled":
        return WandBContext(run=None, run_id=run_id, enabled=False)

    logging_cfg = config.get("logging", {}).get("wandb", {})
    if not logging_cfg.get("enabled", True):
        return WandBContext(run=None, run_id=run_id, enabled=False)

    # Extract key hyperparameters for wandb.config
    # (Don't dump entire config - just what's important)
    wandb_config = {
        "latent_dim": config["latent"]["dim"],
        "latent_tokens": config["latent"]["tokens"],
        "operator_hidden_dim": config["operator"]["pdet"]["hidden_dim"],
        "operator_num_heads": config["operator"]["pdet"]["num_heads"],
        "operator_depths": config["operator"]["pdet"]["depths"],
        "batch_size": config["training"]["batch_size"],
        "time_stride": config["training"]["time_stride"],
        "task": config["data"]["task"],
    }

    # Add training stage configs
    stages = config.get("stages", {})
    if "operator" in stages:
        wandb_config["operator_epochs"] = stages["operator"].get("epochs", 0)
        wandb_config["operator_lr"] = stages["operator"].get("optimizer", {}).get(
            "lr", 0
        )
    if "diff_residual" in stages:
        wandb_config["diffusion_epochs"] = stages["diff_residual"].get("epochs", 0)
        wandb_config["diffusion_lr"] = stages["diff_residual"].get("optimizer", {}).get(
            "lr", 0
        )
    if "consistency_distill" in stages:
        wandb_config["consistency_epochs"] = stages["consistency_distill"].get(
            "epochs", 0
        )

    # Add TTC config if enabled
    ttc_cfg = config.get("ttc", {})
    if ttc_cfg.get("enabled"):
        wandb_config["ttc_enabled"] = True
        wandb_config["ttc_candidates"] = ttc_cfg.get("candidates", 0)
        wandb_config["ttc_beam_width"] = ttc_cfg.get("beam_width", 0)

    try:
        run = wandb.init(
            project=logging_cfg.get("project", "universal-simulator"),
            entity=logging_cfg.get("entity"),
            name=logging_cfg.get("run_name", f"pipeline-{run_id}"),
            id=run_id,  # Explicit run ID for reproducibility
            config=wandb_config,
            tags=logging_cfg.get("tags", []),
            group=logging_cfg.get("group"),
            job_type="training-pipeline",
            mode=mode,
            reinit=False,  # ONE run per pipeline
        )

        # Define metric step relationships for proper charting
        # Each stage uses its own step metric for x-axis
        # Stage names: operator, diffusion_residual, consistency_distill, steady_prior
        if run is not None and wandb is not None:
            # Define step metrics to use the wandb step parameter
            wandb.define_metric("training/operator/step")
            wandb.define_metric("training/diffusion_residual/step")
            wandb.define_metric("training/consistency_distill/step")
            wandb.define_metric("training/steady_prior/step")

            # Define all other metrics in each namespace to use their stage's step
            wandb.define_metric("training/operator/*", step_metric="training/operator/step")
            wandb.define_metric("training/diffusion_residual/*", step_metric="training/diffusion_residual/step")
            wandb.define_metric("training/consistency_distill/*", step_metric="training/consistency_distill/step")
            wandb.define_metric("training/steady_prior/*", step_metric="training/steady_prior/step")

        return WandBContext(run=run, run_id=run_id, enabled=True)

    except Exception as e:
        print(f"⚠️  Failed to initialize WandB: {e}")
        return WandBContext(run=None, run_id=run_id, enabled=False)


def load_wandb_context(
    run_id: str, project: str, entity: Optional[str] = None
) -> Optional[WandBContext]:
    """Load an existing WandB run context (for separate processes).

    This is for edge cases where training runs in a separate process and needs
    to attach to an existing run. Prefer passing WandBContext object directly.

    Args:
        run_id: WandB run ID to resume
        project: WandB project name
        entity: WandB entity (optional)

    Returns:
        WandBContext object or None if unavailable

    Example:
        # In subprocess
        ctx = load_wandb_context(run_id="abc123", project="universal-simulator")
        ctx.log_training_metric("operator", "loss", 0.1, step=10)
    """
    if not WANDB_AVAILABLE:
        return WandBContext(run=None, run_id=run_id, enabled=False)

    try:
        run = wandb.init(
            id=run_id,
            project=project,
            entity=entity,
            resume="allow",
            reinit=True,
            settings=wandb.Settings(start_method="thread"),
        )
        return WandBContext(run=run, run_id=run_id, enabled=True)
    except Exception:
        return WandBContext(run=None, run_id=run_id, enabled=False)


def save_wandb_context(ctx: WandBContext, path: Path) -> None:
    """Save WandB context info to file for subprocess communication.

    Args:
        ctx: WandB context to save
        path: Path to save context JSON

    Example:
        ctx = create_wandb_context(cfg, "run-123")
        save_wandb_context(ctx, Path("wandb_context.json"))
        # Subprocess can load this and resume the run
    """
    if not ctx.enabled or ctx.run is None:
        return

    context_data = {
        "run_id": ctx.run_id,
        "project": ctx.run.project,
        "entity": ctx.run.entity,
        "url": ctx.run.url,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(context_data, indent=2), encoding="utf-8")


def load_wandb_context_from_file(path: Path) -> Optional[WandBContext]:
    """Load WandB context from file (for subprocesses).

    Args:
        path: Path to context JSON file

    Returns:
        WandBContext object or None if file doesn't exist

    Example:
        # In subprocess
        ctx = load_wandb_context_from_file(Path("wandb_context.json"))
        if ctx:
            ctx.log_training_metric("operator", "loss", 0.1, step=10)
    """
    if not path.exists():
        return None

    try:
        context_data = json.loads(path.read_text(encoding="utf-8"))
        return load_wandb_context(
            run_id=context_data["run_id"],
            project=context_data["project"],
            entity=context_data.get("entity"),
        )
    except Exception:
        return None


def load_wandb_context_from_env() -> Optional[WandBContext]:
    """Load WandB context from WANDB_CONTEXT_FILE environment variable.

    This is a convenience function for subprocesses to automatically load
    the context set by the parent orchestrator.

    Returns:
        WandBContext object or None if env var not set

    Example:
        # Parent process sets: export WANDB_CONTEXT_FILE=/path/to/context.json
        # Subprocess:
        ctx = load_wandb_context_from_env()
        if ctx:
            ctx.log_training_metric("operator", "loss", 0.1, step=10)
    """
    context_file = os.environ.get("WANDB_CONTEXT_FILE")
    if not context_file:
        return None

    return load_wandb_context_from_file(Path(context_file))

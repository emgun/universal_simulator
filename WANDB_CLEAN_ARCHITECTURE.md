# WandB Clean Architecture - Design Document

> **✅ STATUS: IMPLEMENTED**
> This design has been fully implemented. See `WANDB_IMPLEMENTATION_SUMMARY.md` for implementation details.
> Implementation completed: 2025-01-22

## Current Issues

### 1. Multiple WandB Runs Hack ❌
**Problem**: To avoid multiple wandb runs, evaluation is disabled with `WANDB_MODE=disabled` (run_fast_to_sota.py:880)

**Why it's hacky**:
- Orchestrator creates a run (line 665)
- Training creates its own run (train.py:1259)
- Evaluation would create another run, so it's disabled
- This prevents eval metrics from appearing in WandB dashboard

### 2. Scalars Logged as Time Series ❌
**Problem**: Single-point evaluation metrics logged with `wandb.log()` instead of `wandb.summary`

**Examples**:
- `evaluate.py:538` - Eval metrics logged as time series (creates useless single-point line charts)
- `evaluate.py:548` - Extra metadata logged as metrics (should be in config/summary)

**Why it's wrong**:
- WandB creates line charts for logged metrics
- Single data points appear as dots on empty charts
- Should use `wandb.summary` for final values

### 3. Run Proliferation ❌
**Problem**: Multiple `wandb.init()` calls throughout codebase

**Locations**:
- `monitoring.py:65` - Creates runs for each component
- `train.py:1259` - Creates training run
- `evaluate.py:524` - Tries to resume run
- `leaderboard.py:94` - Creates leaderboard run
- `run_fast_to_sota.py:665` - Creates orchestrator run

**Why it's messy**:
- Unclear which run contains what data
- Hard to find all metrics for a training session
- Requires complex resume logic
- Necessitates hacks like WANDB_MODE=disabled

---

## Clean Architecture Design

### Core Principle: **One Pipeline = One WandB Run**

```
┌─────────────────────────────────────────────────────────────┐
│                    Single WandB Run                         │
│                 (Created by Orchestrator)                   │
│                                                             │
│  Config:                                                    │
│    - latent_dim, batch_size, etc.                          │
│    - samples, tau, ttc (metadata)                          │
│                                                             │
│  Metrics (Time Series with step):                          │
│    - training/operator/loss (per epoch)                    │
│    - training/operator/lr (per epoch)                      │
│    - training/diffusion/loss (per epoch)                   │
│    - training/consistency/loss (per epoch)                 │
│                                                             │
│  Summary (Final Scalars):                                  │
│    - eval/baseline/nrmse (single value)                    │
│    - eval/baseline/mse (single value)                      │
│    - eval/ttc/nrmse (single value)                         │
│    - eval/ttc/mse (single value)                           │
│    - physics/conservation_gap (single value)               │
│    - physics/bc_violation (single value)                   │
│                                                             │
│  Tables:                                                    │
│    - eval/baseline_vs_ttc (comparison)                     │
│    - physics/violations (categorized)                      │
│    - training/stage_summary (all stages)                   │
│    - leaderboard/entry (this run's entry)                  │
│                                                             │
│  Images:                                                    │
│    - eval/mse_histogram                                    │
│    - eval/latent_heatmap                                   │
│    - eval/ttc_summary                                      │
│                                                             │
│  Files:                                                     │
│    - checkpoints/operator.pt                               │
│    - checkpoints/diffusion_residual.pt                     │
│    - reports/evaluation.json                               │
│                                                             │
│  Artifacts:                                                 │
│    - fast_to_sota_summary_{run_id}                         │
│      ├─ summary.json                                       │
│      ├─ leaderboard.csv                                    │
│      └─ metrics.json                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Data Type Guidelines

| Data Type | WandB Method | Use Case | Example |
|-----------|-------------|----------|---------|
| **Time Series** | `wandb.log({key: value}, step=N)` | Training metrics per epoch/step | `training/operator/loss` |
| **Final Scalars** | `wandb.summary[key] = value` | Single evaluation result | `eval/nrmse = 0.09` |
| **Metadata** | `wandb.config[key] = value` | Hyperparameters, dataset info | `latent_dim = 32` |
| **Comparisons** | `wandb.Table()` | Multi-value side-by-side | Baseline vs TTC table |
| **Categorical** | `wandb.plot.bar()` | Category-based metrics | Physics violations by type |
| **Images** | `wandb.Image()` | Visualizations | Heatmaps, histograms |
| **Files** | `wandb.save()` | Checkpoints, configs | `operator.pt` |
| **Collections** | `wandb.Artifact()` | Related file groups | Summary artifact |

### WandB Run Lifecycle

```python
# 1. Orchestrator creates ONE run
run = wandb.init(
    project="universal-simulator",
    name=f"pipeline-{run_id}",
    config={
        "latent_dim": 32,
        "batch_size": 12,
        "task": "burgers1d",
        # ... all hyperparameters from config
    }
)

# 2. Pass run to training (NO new wandb.init())
train.main(config, wandb_run=run)
  # Logs: wandb.log({"training/operator/loss": loss}, step=global_step)

# 3. Pass run to evaluation (NO new wandb.init())
evaluate.main(config, wandb_run=run)
  # Summary: wandb.summary["eval/nrmse"] = 0.09
  # Tables: wandb.log({"eval/comparison": table})

# 4. Orchestrator finalizes
wandb.log_artifact(summary_artifact)
wandb.finish()
```

---

## Implementation Plan

### Phase 1: Centralized WandB Context Manager

**Purpose**: Single source of truth for WandB run management

**File**: `src/ups/utils/wandb_context.py` (NEW)

```python
"""Centralized WandB context management for clean run lifecycle."""

from __future__ import annotations
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import json

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


@dataclass
class WandBContext:
    """Centralized WandB run context - passed to all components."""

    run: Any  # wandb.Run object
    run_id: str
    enabled: bool = True

    def log_training_metric(self, stage: str, metric: str, value: float, step: int) -> None:
        """Log training metrics as time series."""
        if not self.enabled or self.run is None:
            return
        self.run.log({f"training/{stage}/{metric}": value}, step=step)

    def log_eval_summary(self, metrics: Dict[str, float], prefix: str = "eval") -> None:
        """Log evaluation metrics as summary (single values, not time series)."""
        if not self.enabled or self.run is None:
            return
        # Use summary for single-point metrics (no line charts!)
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.run.summary[f"{prefix}/{key}"] = value

    def log_table(self, name: str, columns: List[str], data: List[List[Any]]) -> None:
        """Log a table for multi-value comparisons."""
        if not self.enabled or self.run is None or wandb is None:
            return
        table = wandb.Table(columns=columns, data=data)
        self.run.log({name: table}, commit=False)

    def log_image(self, name: str, image_path: Path) -> None:
        """Log an image."""
        if not self.enabled or self.run is None or wandb is None:
            return
        self.run.log({name: wandb.Image(str(image_path))}, commit=False)

    def save_file(self, file_path: Path) -> None:
        """Save a file to WandB."""
        if not self.enabled or self.run is None or wandb is None:
            return
        wandb.save(str(file_path), base_path=str(file_path.parent.parent))

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update run config (for metadata, not metrics)."""
        if not self.enabled or self.run is None:
            return
        self.run.config.update(updates, allow_val_change=True)

    def add_tags(self, tags: List[str]) -> None:
        """Add tags to run."""
        if not self.enabled or self.run is None:
            return
        self.run.tags = tuple(set(list(self.run.tags) + tags))

    def finish(self) -> None:
        """Finish the run."""
        if not self.enabled or self.run is None:
            return
        self.run.finish()


def create_wandb_context(
    config: Dict[str, Any],
    run_id: str,
    mode: str = "online",
) -> Optional[WandBContext]:
    """Create a single WandB run context for the entire pipeline.

    Args:
        config: Full training configuration (will be logged to wandb.config)
        run_id: Unique run identifier
        mode: WandB mode (online, offline, disabled)

    Returns:
        WandBContext or None if WandB disabled/unavailable
    """
    if not WANDB_AVAILABLE or mode == "disabled":
        return WandBContext(run=None, run_id=run_id, enabled=False)

    logging_cfg = config.get("logging", {}).get("wandb", {})
    if not logging_cfg.get("enabled", True):
        return WandBContext(run=None, run_id=run_id, enabled=False)

    # Extract hyperparameters for config (not all of config!)
    wandb_config = {
        "latent_dim": config["latent"]["dim"],
        "latent_tokens": config["latent"]["tokens"],
        "operator_hidden_dim": config["operator"]["pdet"]["hidden_dim"],
        "operator_num_heads": config["operator"]["pdet"]["num_heads"],
        "operator_depths": config["operator"]["pdet"]["depths"],
        "batch_size": config["training"]["batch_size"],
        "time_stride": config["training"]["time_stride"],
        "task": config["data"]["task"],
        # Add more key hyperparameters as needed
    }

    # Metadata goes in config too (samples, tau, ttc should NOT be metrics)
    # These will be set later during evaluation

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
        if run is not None:
            wandb.define_metric("training/operator/*", step_metric="training/operator/step")
            wandb.define_metric("training/diffusion/*", step_metric="training/diffusion/step")
            wandb.define_metric("training/consistency/*", step_metric="training/consistency/step")

        return WandBContext(run=run, run_id=run_id, enabled=True)

    except Exception as e:
        print(f"⚠️  Failed to initialize WandB: {e}")
        return WandBContext(run=None, run_id=run_id, enabled=False)


def load_wandb_context(run_id: str, project: str, entity: Optional[str] = None) -> Optional[WandBContext]:
    """Load an existing WandB run context (for separate processes).

    This is only for edge cases where training runs in a separate process.
    Prefer passing WandBContext object directly.
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
        )
        return WandBContext(run=run, run_id=run_id, enabled=True)
    except Exception:
        return WandBContext(run=None, run_id=run_id, enabled=False)
```

### Phase 2: Update Training Script

**File**: `scripts/train.py`

**Changes**:

1. **Remove `wandb.init()` call** (line 1259)
2. **Accept WandBContext parameter**:

```python
def train_all_stages(cfg: dict, wandb_ctx: Optional[WandBContext] = None) -> None:
    """Run all training stages with provided WandB context."""

    # Remove all wandb.init() calls - use provided context

    # Training loops log via context
    for epoch in range(epochs):
        loss = train_epoch(...)

        if wandb_ctx:
            wandb_ctx.log_training_metric(
                stage="operator",
                metric="loss",
                value=loss,
                step=global_step
            )
            wandb_ctx.log_training_metric(
                stage="operator",
                metric="lr",
                value=optimizer.param_groups[0]["lr"],
                step=global_step
            )

    # Checkpoint saving
    if wandb_ctx:
        wandb_ctx.save_file(operator_path)
```

3. **Update main function**:

```python
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # Don't create wandb run here - orchestrator will provide it
    # For standalone use, create one
    wandb_ctx = None
    if args.wandb_enabled:  # New flag
        from ups.utils.wandb_context import create_wandb_context
        wandb_ctx = create_wandb_context(cfg, run_id=f"train-{timestamp}")

    try:
        if args.stage == "all":
            train_all_stages(cfg, wandb_ctx=wandb_ctx)
        elif args.stage == "operator":
            train_operator_stage(cfg, wandb_ctx=wandb_ctx)
        # ... other stages
    finally:
        if wandb_ctx and args.wandb_enabled:
            wandb_ctx.finish()
```

### Phase 3: Update Evaluation Script

**File**: `scripts/evaluate.py`

**Changes**:

1. **Remove wandb.log() for metrics** (line 538)
2. **Use wandb.summary for scalars**:

```python
def evaluate_and_log(
    cfg: Dict,
    operator: LatentOperator,
    diffusion: Optional[DiffusionResidual],
    wandb_ctx: Optional[WandBContext] = None,
    output_prefix: Path = Path("reports/evaluation"),
) -> MetricReport:
    """Run evaluation and log to WandB context."""

    # Run evaluation
    report = evaluate_latent_operator(cfg, operator, diffusion=diffusion, ...)

    # Generate outputs
    outputs = _write_outputs(report, output_prefix, cfg, details)

    if wandb_ctx:
        # 1. Log final metrics to SUMMARY (not as time series!)
        wandb_ctx.log_eval_summary(report.metrics, prefix="eval/baseline")

        # 2. Log metadata to CONFIG (not as metrics!)
        if report.extra:
            wandb_ctx.update_config({
                "eval_samples": report.extra.get("samples"),
                "eval_tau": report.extra.get("tau"),
                "eval_ttc_enabled": report.extra.get("ttc", False),
            })

        # 3. Log images
        for key, path in outputs.items():
            if key.startswith("plot_"):
                wandb_ctx.log_image(f"eval/{key}", path)

        # 4. Log comparison table (baseline vs TTC)
        if baseline_metrics and ttc_metrics:
            comparison_rows = []
            for metric_name in baseline_metrics.keys():
                baseline_val = baseline_metrics[metric_name]
                ttc_val = ttc_metrics.get(metric_name, baseline_val)
                improvement = ((baseline_val - ttc_val) / baseline_val * 100) if baseline_val != 0 else 0

                comparison_rows.append([
                    metric_name,
                    f"{baseline_val:.4f}",
                    f"{ttc_val:.4f}",
                    f"{improvement:.1f}%"
                ])

            wandb_ctx.log_table(
                "eval/baseline_vs_ttc",
                columns=["Metric", "Baseline", "TTC", "Improvement"],
                data=comparison_rows
            )

        # 5. Save files
        for path in outputs.values():
            wandb_ctx.save_file(path)

    return report
```

### Phase 4: Update Orchestrator

**File**: `scripts/run_fast_to_sota.py`

**Changes**:

1. **Create single WandB context** (replace line 665):

```python
def main() -> None:
    args = parse_args()

    # ... setup ...

    # Create ONE WandB context for entire pipeline
    wandb_ctx = None
    if args.wandb_sync:
        from ups.utils.wandb_context import create_wandb_context
        wandb_ctx = create_wandb_context(
            config=cfg,
            run_id=run_id,
            mode=args.wandb_mode
        )

        if wandb_ctx and wandb_ctx.enabled:
            wandb_ctx.add_tags(["fast-to-sota", "orchestrator"] + args.wandb_tags)

    try:
        # 2. Save WandB context for subprocess communication
        if wandb_ctx and wandb_ctx.enabled:
            context_file = run_root / "wandb_context.json"
            context_file.write_text(json.dumps({
                "run_id": wandb_ctx.run_id,
                "project": wandb_ctx.run.project,
                "entity": wandb_ctx.run.entity,
            }))

        # 3. Call training with context (NO MORE WANDB_MODE=disabled!)
        train_cmd = [
            PYTHON, "scripts/train.py",
            "--config", str(resolved_train_config),
            "--stage", "all",
        ]

        # Pass context file path instead of disabling wandb
        train_env = {
            "WANDB_CONTEXT_FILE": str(context_file) if wandb_ctx else "",
        }

        _run_command(train_cmd, env=train_env, desc="train")

        # 4. Call evaluation with same context (NO MORE WANDB_MODE=disabled!)
        eval_cmd = [
            PYTHON, "scripts/evaluate.py",
            "--config", str(resolved_eval_config),
            "--operator", str(operator_ckpt),
            "--output-prefix", str(eval_prefix),
        ]

        eval_env = {
            "WANDB_CONTEXT_FILE": str(context_file) if wandb_ctx else "",
        }

        _run_command(eval_cmd, env=eval_env, desc="evaluate")

        # 5. Leaderboard logging (within same run)
        if wandb_ctx and wandb_ctx.enabled:
            # Log this run's leaderboard entry as a table
            leaderboard_row = build_leaderboard_row(metrics, run_id, ...)
            wandb_ctx.log_table(
                "leaderboard/this_run",
                columns=list(leaderboard_row.keys()),
                data=[list(leaderboard_row.values())]
            )

        # 6. Finalize
        if wandb_ctx:
            wandb_ctx.log_artifact(summary_artifact)
            wandb_ctx.finish()

    except Exception as e:
        if wandb_ctx:
            wandb_ctx.finish()
        raise
```

2. **Remove WANDB_MODE=disabled hack** (line 880):

```python
# DELETE THIS BLOCK (lines 879-881):
# eval_env = {
#     "WANDB_MODE": "disabled",  # ❌ HACK - NO MORE!
# }
```

### Phase 5: Update Subprocess Context Loading

For subprocesses (train.py, evaluate.py called by orchestrator):

```python
def load_context_from_env() -> Optional[WandBContext]:
    """Load WandB context from environment variable (for subprocesses)."""
    context_file = os.environ.get("WANDB_CONTEXT_FILE")
    if not context_file or not Path(context_file).exists():
        return None

    context_data = json.loads(Path(context_file).read_text())
    return load_wandb_context(
        run_id=context_data["run_id"],
        project=context_data["project"],
        entity=context_data.get("entity")
    )

# In train.py and evaluate.py main():
def main() -> None:
    # Try to load context from orchestrator
    wandb_ctx = load_context_from_env()

    # If not found, create own (standalone mode)
    if wandb_ctx is None and args.wandb_enabled:
        wandb_ctx = create_wandb_context(cfg, run_id=...)

    # ... rest of code uses wandb_ctx
```

---

## Benefits of Clean Architecture

### ✅ Simplicity
- ONE `wandb.init()` call per pipeline
- Clear data ownership
- No mode switching hacks

### ✅ Correct Data Types
- Training metrics → Time series with `wandb.log()`
- Eval metrics → Summary scalars with `wandb.summary`
- Metadata → Config with `wandb.config`
- Comparisons → Tables with `wandb.Table()`

### ✅ Better UX
- All metrics in one run
- Proper chart types (no single-point line charts)
- Easy comparison across runs
- Clear organization (training/, eval/, physics/)

### ✅ Maintainability
- Centralized WandB logic in one module
- Type-safe context passing
- Easy to test (mock WandBContext)
- No global state

---

## Migration Strategy

### Step 1: Add New Module (No Breaking Changes)
- Create `src/ups/utils/wandb_context.py`
- Add unit tests
- Deploy without using it yet

### Step 2: Update Orchestrator
- Modify `run_fast_to_sota.py` to use WandBContext
- Remove WANDB_MODE=disabled hack
- Keep backward compatibility with subprocess context loading

### Step 3: Update Training
- Modify `train.py` to accept WandBContext
- Remove old `wandb.init()` call
- Support both standalone and orchestrated modes

### Step 4: Update Evaluation
- Modify `evaluate.py` to use summary/config instead of log()
- Remove old resume logic
- Use WandBContext for all logging

### Step 5: Deprecate Old Patterns
- Remove `MonitoringSession` (replaced by WandBContext)
- Clean up old wandb.init() calls
- Update tests

---

## Testing Strategy

### Unit Tests

`tests/unit/test_wandb_context.py`:
```python
def test_wandb_context_disabled():
    ctx = WandBContext(run=None, run_id="test", enabled=False)
    ctx.log_training_metric("operator", "loss", 0.1, step=1)  # Should not crash

def test_wandb_context_training_metrics(mock_wandb_run):
    ctx = WandBContext(run=mock_wandb_run, run_id="test", enabled=True)
    ctx.log_training_metric("operator", "loss", 0.1, step=5)
    mock_wandb_run.log.assert_called_with({"training/operator/loss": 0.1}, step=5)

def test_wandb_context_eval_summary(mock_wandb_run):
    ctx = WandBContext(run=mock_wandb_run, run_id="test", enabled=True)
    ctx.log_eval_summary({"nrmse": 0.09, "mse": 0.001}, prefix="eval")
    assert mock_wandb_run.summary["eval/nrmse"] == 0.09
    assert mock_wandb_run.summary["eval/mse"] == 0.001
```

### Integration Tests

`tests/integration/test_full_pipeline_clean_wandb.py`:
- Test full pipeline with single wandb run
- Verify no duplicate runs created
- Verify correct data types (summary vs log)
- Verify subprocess context loading

### Manual Testing

1. Run full pipeline:
   ```bash
   python scripts/run_fast_to_sota.py \
     --train-config configs/train_burgers_32dim.yaml \
     --small-eval-config configs/small_eval_burgers.yaml \
     --full-eval-config configs/full_eval_burgers.yaml \
     --wandb-sync \
     --wandb-mode online
   ```

2. Verify in WandB dashboard:
   - ✅ Only ONE run created
   - ✅ Training metrics show as line charts (multi-point)
   - ✅ Eval metrics show in Summary tab (single values)
   - ✅ Metadata in Config tab (samples, tau, ttc)
   - ✅ Tables show comparisons
   - ✅ No empty/single-point line charts

3. Check run structure:
   ```
   Run: pipeline-run_20250122_143022
   ├─ Config
   │  ├─ latent_dim: 32
   │  ├─ batch_size: 12
   │  ├─ eval_samples: 3072
   │  └─ eval_ttc_enabled: true
   ├─ Charts (Time Series)
   │  ├─ training/operator/loss (line chart, 25 points)
   │  ├─ training/operator/lr (line chart, 25 points)
   │  ├─ training/diffusion/loss (line chart, 8 points)
   │  └─ training/consistency/loss (line chart, 8 points)
   ├─ Summary (Scalars)
   │  ├─ eval/baseline/nrmse: 0.78
   │  ├─ eval/baseline/mse: 0.99
   │  ├─ eval/ttc/nrmse: 0.09
   │  ├─ eval/ttc/mse: 0.01
   │  ├─ physics/conservation_gap: 8.08
   │  └─ physics/bc_violation: 0.82
   ├─ Tables
   │  ├─ eval/baseline_vs_ttc (4 columns, N rows)
   │  ├─ physics/violations (3 columns, 3 rows)
   │  └─ leaderboard/this_run (20+ columns, 1 row)
   ├─ Images
   │  ├─ eval/mse_histogram
   │  ├─ eval/latent_heatmap
   │  └─ eval/ttc_summary
   └─ Files
      ├─ checkpoints/operator.pt
      ├─ checkpoints/diffusion_residual.pt
      └─ reports/evaluation.json
   ```

---

## Summary

This clean architecture:

1. **Eliminates hacks**: No more `WANDB_MODE=disabled`
2. **Proper data types**: Summary for scalars, log for time series
3. **Single run**: One wandb run per pipeline execution
4. **Simple flow**: Context passed explicitly, no global state
5. **Type-safe**: WandBContext class with clear methods
6. **Testable**: Easy to mock and unit test
7. **Maintainable**: Centralized logic in one module

The result is a clean, professional WandB integration that follows best practices and provides an excellent user experience.

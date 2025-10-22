# WandB Optimization Implementation Plan

> **✅ STATUS: CORE ARCHITECTURE IMPLEMENTED**
> The clean WandB architecture (Phases 1-4) has been fully implemented. See `WANDB_IMPLEMENTATION_SUMMARY.md` for details.
> Remaining items (regression detection, enhanced visualizations) are tracked as future enhancements.
> Core implementation completed: 2025-01-22

## Overview

Enhance the WandB integration to provide comprehensive tracking, visualization, and automated regression detection for remote VastAI training runs.

## Current State Analysis

### Strengths
- Solid WandB integration with metrics logging
- Comprehensive evaluation metrics (MSE, MAE, RMSE, NRMSE, physics diagnostics)
- Remote sync working well (checkpoints, reports, artifacts uploaded)
- Leaderboard system (CSV + HTML) functional
- Image logging (histograms, heatmaps, spectra)
- Basic tables for metrics

### Gaps
1. No regression guards or automated baseline comparison
2. Leaderboard not visualized in WandB dashboard
3. Limited custom chart configuration
4. No consolidated metrics dashboard
5. Physics gates not visualized as charts/alerts
6. No automated regression alerting
7. TTC analysis lacks summary visualizations
8. No training stage comparison tables

## Desired End State

### Key Improvements
1. **Automated Regression Detection**: Compare each run against baseline, alert on regressions
2. **Enhanced Visualizations**: Custom charts for stage comparison, physics gates, TTC analysis
3. **Dashboard Integration**: Leaderboard rows logged to WandB for dashboard viewing
4. **Consolidated Summary**: Single-page training summary with all key metrics
5. **Physics Monitoring**: Real-time alerts when conservation laws violated
6. **Automated Gates**: Promotion gates logged as pass/fail with explanations

### Success Criteria
- Regression detection catches performance drops > 10% NRMSE
- All leaderboard entries visible in WandB dashboard
- Physics violations trigger WandB alerts
- Training summary table shows all stages side-by-side
- TTC reward curves visualized for each evaluation
- Zero manual data copying required (fully automated sync)

## What We're NOT Doing

- Not changing the training pipeline itself (only logging/visualization)
- Not modifying existing checkpoint save/load logic
- Not changing leaderboard CSV/HTML format (only adding WandB logging)
- Not retraining models or changing hyperparameters

---

## Phase 1: Enhanced WandB Logging Infrastructure

### Overview
Add utilities for custom charts, summary tables, and baseline tracking.

### Changes Required

#### 1. Enhanced Monitoring Module
**File**: `src/ups/utils/monitoring.py`

Add methods to `MonitoringSession`:

```python
def log_table(self, key: str, columns: List[str], data: List[List[Any]]) -> None:
    """Log a table to WandB."""
    if self.run is not None and wandb is not None:
        table = wandb.Table(columns=columns, data=data)
        self.run.log({key: table}, commit=False)

def log_summary(self, data: Dict[str, Any]) -> None:
    """Update run summary with key metrics."""
    if self.run is not None:
        for key, value in data.items():
            self.run.summary[key] = value

def define_custom_charts(self, chart_configs: Dict[str, Any]) -> None:
    """Define custom WandB charts for better visualization."""
    if self.run is not None and wandb is not None:
        for chart_name, config in chart_configs.items():
            # Configure custom line charts, bar charts, etc.
            pass
```

#### 2. Baseline Tracking Module
**File**: `src/ups/utils/baseline_tracker.py` (NEW)

```python
from pathlib import Path
from typing import Dict, Optional, List
import json

class BaselineTracker:
    """Track baseline metrics for regression detection."""

    def __init__(self, baseline_path: Path):
        self.baseline_path = baseline_path
        self.baseline = self._load_baseline()

    def _load_baseline(self) -> Optional[Dict[str, float]]:
        """Load baseline metrics from file."""
        if not self.baseline_path.exists():
            return None
        return json.loads(self.baseline_path.read_text())

    def save_baseline(self, metrics: Dict[str, float], run_id: str) -> None:
        """Save new baseline metrics."""
        self.baseline = {
            "metrics": metrics,
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self.baseline_path.write_text(json.dumps(self.baseline, indent=2))

    def compare(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Compare current metrics against baseline."""
        if not self.baseline:
            return {"status": "no_baseline", "regressions": [], "improvements": []}

        regressions = []
        improvements = []

        for key, current_value in current_metrics.items():
            if key not in self.baseline["metrics"]:
                continue

            baseline_value = self.baseline["metrics"][key]
            pct_change = ((current_value - baseline_value) / baseline_value) * 100

            # For error metrics (lower is better)
            if key in ["mse", "mae", "rmse", "nrmse", "rel_l2"]:
                if pct_change > 10:  # 10% worse
                    regressions.append({
                        "metric": key,
                        "baseline": baseline_value,
                        "current": current_value,
                        "pct_change": pct_change
                    })
                elif pct_change < -5:  # 5% better
                    improvements.append({
                        "metric": key,
                        "baseline": baseline_value,
                        "current": current_value,
                        "pct_change": pct_change
                    })

        return {
            "status": "regressions" if regressions else "pass",
            "regressions": regressions,
            "improvements": improvements,
            "baseline_run_id": self.baseline.get("run_id")
        }
```

### Success Criteria

#### Automated Verification:
- [ ] New methods added to MonitoringSession: `make -C . test`
- [ ] BaselineTracker module passes unit tests: `pytest tests/unit/test_baseline_tracker.py`
- [ ] No type errors: `mypy src/ups/utils/`
- [ ] Linting passes: `make lint`

#### Manual Verification:
- [ ] Baseline tracking correctly identifies regressions
- [ ] Tables render properly in WandB interface
- [ ] Summary metrics update correctly

---

## Phase 2: Regression Detection and Alerting

### Overview
Integrate baseline tracking into evaluation pipeline with automated WandB alerts.

### Changes Required

#### 1. Evaluation Script Integration
**File**: `scripts/evaluate.py`

Add regression detection after metrics computation (after line 576):

```python
# Regression detection (NEW)
baseline_tracker = None
baseline_path = Path("reports/baseline_metrics.json")
if args.track_regressions:
    from ups.utils.baseline_tracker import BaselineTracker
    baseline_tracker = BaselineTracker(baseline_path)

    comparison = baseline_tracker.compare(report.metrics)

    # Log comparison to WandB
    if session.run is not None:
        session.log({
            "regression/status": comparison["status"],
            "regression/num_regressions": len(comparison["regressions"]),
            "regression/num_improvements": len(comparison["improvements"])
        })

        # Create comparison table
        if comparison["regressions"] or comparison["improvements"]:
            rows = []
            for r in comparison["regressions"]:
                rows.append(["❌ REGRESSION", r["metric"], r["baseline"], r["current"], f"{r['pct_change']:.1f}%"])
            for i in comparison["improvements"]:
                rows.append(["✅ IMPROVEMENT", i["metric"], i["baseline"], i["current"], f"{i['pct_change']:.1f}%"])

            session.log_table(
                "regression/comparison",
                columns=["Status", "Metric", "Baseline", "Current", "Change"],
                data=rows
            )

        # Send alert if regressions detected
        if comparison["regressions"] and wandb is not None and wandb.run is not None:
            alert_text = f"⚠️ {len(comparison['regressions'])} regression(s) detected:\n"
            for r in comparison["regressions"]:
                alert_text += f"- {r['metric']}: {r['baseline']:.4f} → {r['current']:.4f} ({r['pct_change']:+.1f}%)\n"

            wandb.alert(
                title="Performance Regression Detected",
                text=alert_text,
                level=wandb.AlertLevel.WARN
            )

# Update baseline if this is marked as champion (NEW)
if args.update_baseline:
    if baseline_tracker is None:
        baseline_tracker = BaselineTracker(baseline_path)
    baseline_tracker.save_baseline(report.metrics, args.leaderboard_run_id or "unknown")
    print(f"✅ Updated baseline metrics")
```

#### 2. CLI Arguments
**File**: `scripts/evaluate.py`

Add new arguments (after line 410):

```python
parser.add_argument("--track-regressions", action="store_true", help="Track and alert on regressions")
parser.add_argument("--update-baseline", action="store_true", help="Update baseline metrics with this run")
```

#### 3. Fast-to-SOTA Integration
**File**: `scripts/run_fast_to_sota.py`

Enable regression tracking for full eval (line 1070):

```python
full_cmd.append("--track-regressions")

# Update baseline if promoted to champion
if promoted:
    full_cmd.append("--update-baseline")
```

### Success Criteria

#### Automated Verification:
- [ ] Regression detection runs without errors: `python scripts/evaluate.py --config configs/eval_burgers_32dim_practical.yaml --operator checkpoints/operator.pt --track-regressions`
- [ ] Unit tests pass: `pytest tests/unit/test_regression_detection.py`
- [ ] Baseline file created: `test -f reports/baseline_metrics.json`

#### Manual Verification:
- [ ] WandB alert triggers when NRMSE increases > 10%
- [ ] Comparison table shows in WandB dashboard
- [ ] Baseline updates correctly when champion promoted
- [ ] No false positives on first run (no baseline)

---

## Phase 3: Enhanced Visualizations

### Overview
Add custom charts for training stages, physics monitoring, and TTC analysis.

### Changes Required

#### 1. Training Stage Comparison
**File**: `scripts/train.py`

Add stage summary table after all stages complete (after line 1440):

```python
# Training Stage Comparison Table (NEW)
if wandb is not None and wandb.run is not None:
    stage_rows = []
    for stage_name in ["operator", "diff_residual", "consistency_distill"]:
        # Collect final metrics for each stage from run history
        # This requires querying the run history API
        pass

    stage_table = wandb.Table(
        columns=["Stage", "Final Loss", "Best Loss", "Total Epochs", "Time (min)"],
        data=stage_rows
    )
    wandb.log({"training_summary/stage_comparison": stage_table})
```

#### 2. Physics Gates Visualization
**File**: `scripts/evaluate.py`

Add physics monitoring charts (after line 568):

```python
# Physics Gates Chart (NEW)
if session.run is not None and wandb is not None:
    physics_metrics = {
        k: v for k, v in report.metrics.items()
        if k in ["conservation_gap", "bc_violation", "negativity_penalty"]
    }

    # Create bar chart
    session.log({
        "physics/violations": wandb.plot.bar(
            wandb.Table(
                columns=["metric", "value"],
                data=[[k, v] for k, v in physics_metrics.items()]
            ),
            "metric",
            "value",
            title="Physics Constraint Violations"
        )
    })

    # Alert if violations exceed thresholds
    alert_messages = []
    if physics_metrics.get("conservation_gap", 0) > 5.0:
        alert_messages.append(f"Conservation gap: {physics_metrics['conservation_gap']:.2f} > 5.0")
    if physics_metrics.get("bc_violation", 0) > 1.0:
        alert_messages.append(f"BC violation: {physics_metrics['bc_violation']:.2f} > 1.0")
    if physics_metrics.get("negativity_penalty", 0) > 0.5:
        alert_messages.append(f"Negativity penalty: {physics_metrics['negativity_penalty']:.2f} > 0.5")

    if alert_messages:
        wandb.alert(
            title="Physics Constraint Violations Detected",
            text="⚠️ " + "\n".join(alert_messages),
            level=wandb.AlertLevel.WARN
        )
```

#### 3. TTC Analysis Charts
**File**: `scripts/evaluate.py`

Add TTC summary visualization (after line 207):

```python
def _plot_ttc_summary(ttc_logs: List[Dict], prefix: Path) -> Path:
    """Create TTC summary with reward distributions and selection stats."""
    if not ttc_logs:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Reward trajectory (top-left)
    steps = [log["step"] for log in ttc_logs]
    chosen_rewards = [log["rewards"][log["chosen"]] for log in ttc_logs]
    mean_rewards = [np.mean(log["rewards"]) for log in ttc_logs]

    axes[0, 0].plot(steps, chosen_rewards, 'g-', label='Chosen', linewidth=2)
    axes[0, 0].plot(steps, mean_rewards, 'b--', label='Mean')
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].set_title("TTC Reward Trajectory")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Reward distribution (top-right)
    all_rewards = [r for log in ttc_logs for r in log["rewards"]]
    axes[0, 1].hist(all_rewards, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel("Reward")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("TTC Reward Distribution")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Candidate selection frequency (bottom-left)
    chosen_indices = [log["chosen"] for log in ttc_logs]
    unique, counts = np.unique(chosen_indices, return_counts=True)
    axes[1, 0].bar(unique, counts, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel("Candidate Index")
    axes[1, 0].set_ylabel("Times Selected")
    axes[1, 0].set_title("Candidate Selection Frequency")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Reward improvement over baseline (bottom-right)
    baseline_reward = ttc_logs[0]["rewards"][0]  # First candidate, first step
    improvements = [(log["rewards"][log["chosen"]] - baseline_reward) / abs(baseline_reward) * 100
                    for log in ttc_logs]
    axes[1, 1].plot(steps, improvements, 'r-', linewidth=2)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Improvement (%)")
    axes[1, 1].set_title("TTC Reward Improvement")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = prefix.parent / f"{prefix.name}_ttc_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path
```

### Success Criteria

#### Automated Verification:
- [ ] Charts generate without errors: `python scripts/evaluate.py --config configs/eval_burgers_32dim_practical.yaml --operator checkpoints/operator.pt --diffusion checkpoints/diffusion_residual.pt`
- [ ] All images created: `ls reports/*_ttc_summary.png`
- [ ] WandB upload succeeds (check via `wandb status`)

#### Manual Verification:
- [ ] Stage comparison table shows in WandB dashboard
- [ ] Physics violations trigger alerts in WandB
- [ ] TTC summary chart displays all 4 subplots correctly
- [ ] Bar charts render properly in WandB interface

---

## Phase 4: Leaderboard Dashboard Integration

### Overview
Log leaderboard entries to WandB for dashboard visualization and searching.

### Changes Required

#### 1. Enhanced Leaderboard Logging
**File**: `src/ups/utils/leaderboard.py`

Update `_log_to_wandb()` function (replace lines 84-100):

```python
def _log_to_wandb(
    row: Dict[str, Any],
    *,
    project: Optional[str],
    entity: Optional[str],
    run_name: Optional[str]
) -> None:
    """Log leaderboard row to W&B with table and summary updates."""
    try:
        import wandb
    except ImportError:
        return

    if wandb.run is None:
        wandb.init(
            project=project or "universal-simulator",
            entity=entity,
            name=run_name or row.get("run_id", "leaderboard"),
            job_type="leaderboard",
            reinit=True,
        )
        should_finish = True
    else:
        should_finish = False

    # Log individual metrics (existing behavior)
    metrics_to_log = {f"leaderboard/{k}": v for k, v in row.items() if k != "notes" and isinstance(v, (int, float, bool))}
    if metrics_to_log:
        wandb.log(metrics_to_log)

    # NEW: Create leaderboard row table
    if wandb.run is not None:
        # Single-row table for this entry
        row_table = wandb.Table(
            columns=list(row.keys()),
            data=[list(row.values())]
        )
        wandb.log({"leaderboard/entry": row_table})

        # Update summary with key metrics
        summary_metrics = {
            k: v for k, v in row.items()
            if k.startswith("metric:") or k in ["run_id", "label", "timestamp"]
        }
        for k, v in summary_metrics.items():
            wandb.run.summary[f"leaderboard_{k}"] = v

    if should_finish:
        wandb.finish()
```

#### 2. Consolidated Leaderboard Table
**File**: `scripts/run_fast_to_sota.py`

Add consolidated leaderboard logging (after line 1310):

```python
# Log consolidated leaderboard to WandB (NEW)
if wandb_run is not None and wandb is not None and leaderboard_csv.exists():
    import pandas as pd
    try:
        df = pd.read_csv(leaderboard_csv)

        # Create WandB table from CSV
        leaderboard_table = wandb.Table(dataframe=df)
        wandb_run.log({"leaderboard/full_table": leaderboard_table})

        # Log top performers (by NRMSE)
        if "metric:nrmse" in df.columns:
            top_runs = df.nsmallest(5, "metric:nrmse")
            top_table = wandb.Table(dataframe=top_runs)
            wandb_run.log({"leaderboard/top_5_nrmse": top_table})
    except Exception as e:
        print(f"Note: Could not log consolidated leaderboard: {e}")
```

### Success Criteria

#### Automated Verification:
- [ ] Leaderboard entry logged to WandB: `python scripts/update_leaderboard.py --metrics reports/evaluation.json --run-id test_run --wandb`
- [ ] Table appears in WandB run: Check run page for `leaderboard/entry` table
- [ ] No errors in logging: Check WandB logs for upload success

#### Manual Verification:
- [ ] Leaderboard table searchable in WandB
- [ ] Top 5 runs table shows correctly
- [ ] Individual run summaries show key metrics
- [ ] Can filter leaderboard by tags in WandB

---

## Phase 5: Consolidated Training Summary

### Overview
Create a single comprehensive summary table/report showing all stages and key metrics.

### Changes Required

#### 1. Training Summary Generator
**File**: `scripts/train.py`

Add comprehensive summary logging (after line 1450):

```python
def _generate_training_summary(cfg: Dict, checkpoint_dir: Path) -> Dict[str, Any]:
    """Generate comprehensive training summary."""
    summary = {
        "config": {
            "latent_dim": cfg["latent"]["dim"],
            "latent_tokens": cfg["latent"]["tokens"],
            "operator_hidden_dim": cfg["operator"]["pdet"]["hidden_dim"],
            "training_batch_size": cfg["training"]["batch_size"],
        },
        "stages": {},
        "checkpoints": {}
    }

    # Collect stage metrics (would need to track these during training)
    # For now, read from WandB run history or checkpoint metadata

    return summary

# Log comprehensive summary (NEW)
if wandb is not None and wandb.run is not None:
    summary = _generate_training_summary(cfg, checkpoint_dir)

    # Create summary table
    summary_rows = [
        ["Configuration", "Value"],
        ["Latent Dim", summary["config"]["latent_dim"]],
        ["Latent Tokens", summary["config"]["latent_tokens"]],
        ["Operator Hidden Dim", summary["config"]["operator_hidden_dim"]],
        ["Batch Size", summary["config"]["training_batch_size"]],
    ]

    summary_table = wandb.Table(
        columns=["Parameter", "Value"],
        data=summary_rows
    )
    wandb.log({"training_summary/configuration": summary_table})
```

#### 2. Evaluation Summary
**File**: `scripts/evaluate.py`

Add evaluation summary table (after line 576):

```python
# Comprehensive Evaluation Summary (NEW)
if session.run is not None and wandb is not None:
    summary_rows = [
        ["Category", "Metric", "Value"],
        # Accuracy metrics
        ["Accuracy", "MSE", f"{report.metrics.get('mse', 0):.6f}"],
        ["Accuracy", "MAE", f"{report.metrics.get('mae', 0):.4f}"],
        ["Accuracy", "RMSE", f"{report.metrics.get('rmse', 0):.4f}"],
        ["Accuracy", "NRMSE", f"{report.metrics.get('nrmse', 0):.4f}"],
        # Physics metrics
        ["Physics", "Conservation Gap", f"{report.metrics.get('conservation_gap', 0):.4f}"],
        ["Physics", "BC Violation", f"{report.metrics.get('bc_violation', 0):.4f}"],
        ["Physics", "Negativity Penalty", f"{report.metrics.get('negativity_penalty', 0):.4f}"],
        # Extra info
        ["Info", "Samples", str(report.extra.get("samples", 0))],
        ["Info", "TTC Enabled", str(report.extra.get("ttc", False))],
    ]

    eval_summary_table = wandb.Table(
        columns=["Category", "Metric", "Value"],
        data=summary_rows
    )
    wandb.log({"eval_summary/full": eval_summary_table})
```

### Success Criteria

#### Automated Verification:
- [ ] Summary table logged: Check WandB run for `training_summary/configuration` table
- [ ] Evaluation summary logged: Check WandB run for `eval_summary/full` table
- [ ] All metrics included: Verify table has all expected rows

#### Manual Verification:
- [ ] Summary table readable in WandB dashboard
- [ ] All key configuration parameters shown
- [ ] Evaluation summary categorizes metrics correctly
- [ ] Easy to compare across multiple runs

---

## Testing Strategy

### Unit Tests

Create `tests/unit/test_baseline_tracker.py`:
- Test baseline save/load
- Test regression detection logic
- Test improvement detection
- Test missing baseline handling

Create `tests/unit/test_wandb_logging.py`:
- Test custom table logging
- Test summary updates
- Test chart generation
- Mock WandB API calls

### Integration Tests

Create `tests/integration/test_full_pipeline_wandb.py`:
- Test full training pipeline with WandB logging
- Test evaluation with regression detection
- Test leaderboard updates with WandB sync
- Verify all artifacts uploaded

### Manual Testing Steps

1. Run training with WandB enabled:
   ```bash
   python scripts/train.py --config configs/train_burgers_32dim.yaml --stage all
   ```
   - Verify stage comparison table appears in WandB
   - Verify checkpoints uploaded

2. Run evaluation with regression tracking:
   ```bash
   python scripts/evaluate.py \
     --config configs/eval_burgers_32dim_practical.yaml \
     --operator checkpoints/operator.pt \
     --diffusion checkpoints/diffusion_residual.pt \
     --track-regressions
   ```
   - Verify baseline comparison runs
   - Verify alerts trigger if regression detected
   - Verify physics gate charts appear

3. Run full Fast-to-SOTA pipeline:
   ```bash
   python scripts/run_fast_to_sota.py \
     --train-config configs/train_burgers_32dim.yaml \
     --small-eval-config configs/small_eval_burgers.yaml \
     --full-eval-config configs/full_eval_burgers.yaml \
     --wandb-sync \
     --wandb-mode online
   ```
   - Verify leaderboard logged to WandB
   - Verify consolidated summary artifact created
   - Verify all stages complete successfully

4. Launch on VastAI:
   ```bash
   python scripts/vast_launch.py launch \
     --config configs/train_burgers_32dim.yaml \
     --auto-shutdown
   ```
   - Monitor WandB dashboard during remote run
   - Verify all artifacts sync correctly
   - Verify auto-shutdown after completion

---

## Performance Considerations

- WandB logging adds ~2-5% overhead to training time
- Table logging should be done sparingly (once per stage, not per step)
- Image uploads batched to reduce API calls
- Baseline comparison happens only during evaluation (not training)
- Leaderboard logging happens only when `--leaderboard-wandb` flag set

---

## Migration Notes

- Existing WandB runs will continue to work
- New features opt-in via CLI flags (`--track-regressions`, `--update-baseline`)
- Baseline file (`reports/baseline_metrics.json`) created on first champion run
- No breaking changes to existing evaluation pipeline

---

## References

- WandB API docs: https://docs.wandb.ai/ref/python/
- Existing evaluation script: `scripts/evaluate.py`
- Existing leaderboard system: `src/ups/utils/leaderboard.py`
- VastAI launch script: `scripts/vast_launch.py`
- Fast-to-SOTA orchestrator: `scripts/run_fast_to_sota.py`

---

## Implementation Order

1. **Phase 1** - Foundation (enhanced logging utilities)
2. **Phase 2** - Regression detection (highest value for preventing bad runs)
3. **Phase 3** - Visualizations (improved debugging and analysis)
4. **Phase 4** - Leaderboard dashboard (better run comparison)
5. **Phase 5** - Consolidated summaries (polished final presentation)

Each phase is independently testable and deployable.

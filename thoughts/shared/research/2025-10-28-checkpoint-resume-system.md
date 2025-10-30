---
date: 2025-10-29T02:41:19+0000
researcher: Emery Gunselman
git_commit: ebe5320e3d80ca714dab1765a19dc33305d9dea4
branch: feature--UPT
repository: universal_simulator
topic: "Checkpoint and Resume System for Remote Training Instances"
tags: [research, codebase, checkpoints, resume, vastai, wandb, training-pipeline]
status: complete
last_updated: 2025-10-28
last_updated_by: Emery Gunselman
---

# Research: Checkpoint and Resume System for Remote Training Instances

**Date**: 2025-10-29T02:41:19+0000
**Researcher**: Emery Gunselman
**Git Commit**: ebe5320e3d80ca714dab1765a19dc33305d9dea4
**Branch**: feature--UPT
**Repository**: universal_simulator

## Research Question

How can we implement optimal resuming of remote training instances based on the data/checkpoints in the instance and from wandb runs/checkpoints? The system should:
1. Resume the current training stage or any other training stage
2. Resume the rest of the pipeline (e.g., evaluation stages)
3. Support running specific stages in isolation (e.g., just eval on a given checkpoint)
4. Update wandb runs and data accordingly

## Summary

The current codebase has a **partial checkpoint and resume system** with distinct mechanisms for different use cases:

1. **CheckpointManager-based resume** (modern approach): Downloads checkpoints from WandB runs and resumes training with WandB run continuity
2. **Metadata-based caching**: Skips completed pipeline phases (training, small eval, full eval) based on checkpoint metadata
3. **Stage-specific execution**: Can run individual training stages via `--stage` flag, but requires manual config editing for granular resumption
4. **Standalone evaluation**: Fully supports running evaluation independently on any checkpoint

**Key Gap**: There is **no automatic stage detection** to determine which training stage to resume from. The system relies on:
- Checkpoint file existence for implicit stage completion detection
- Config `epochs: 0` to manually skip completed stages
- Metadata `trained: true` flag for full training completion

**Recommendation**: Implement an explicit **stage completion tracking system** to enable intelligent resumption of multi-stage pipelines on remote instances.

## Detailed Findings

### 1. Checkpoint Saving and Loading System

#### Checkpoint Manager Utility
**Location**: `src/ups/utils/checkpoint_manager.py:11-233`

The `CheckpointManager` class provides the core resumption infrastructure:

**Download checkpoints from WandB** (`checkpoint_manager.py:36-119`):
- Downloads specific checkpoint files from a WandB run
- Default checkpoints:
  - `checkpoints/operator.pt`
  - `checkpoints/operator_ema.pt`
  - `checkpoints/diffusion_residual.pt`
  - `checkpoints/diffusion_residual_ema.pt`
  - `checkpoints/scale/input_stats.pt`
  - `checkpoints/scale/latent_stats.pt`
- Skips files that already exist locally (unless `force=True`)
- Returns list of downloaded file paths

**Setup WandB resumption** (`checkpoint_manager.py:121-134`):
- Sets `WANDB_RUN_ID` environment variable
- Sets `WANDB_RESUME` mode (allow/must/never)
- Enables WandB to append to existing run instead of creating new one

**Verify checkpoints exist** (`checkpoint_manager.py:136-166`):
- Checks that required checkpoint files are present
- Default requirement: `operator.pt` and `operator_ema.pt`
- Raises error if critical files missing

#### Training Script Checkpoint Management
**Location**: `scripts/train.py`

**Checkpoint saving pattern** (used by all stages):
1. Track best model during training: `best_loss`, `best_state`, `best_ema_state`
2. Restore best model before saving: `operator.load_state_dict(best_state)`
3. Save to disk: `torch.save(operator.state_dict(), operator_path)`
4. Upload to WandB: `wandb_ctx.save_file(operator_path)`

**Stage-specific checkpoint locations**:
- **Operator** (`train.py:669-682`):
  - `checkpoints/operator.pt`
  - `checkpoints/operator_ema.pt`
- **Diffusion** (`train.py:858-868`):
  - `checkpoints/diffusion_residual.pt`
  - `checkpoints/diffusion_residual_ema.pt`
- **Consistency Distillation** (`train.py:1177-1187`):
  - Overwrites `diffusion_residual.pt` (by design - it's an improved version)
- **Steady Prior** (`train.py:1293-1299`):
  - `checkpoints/steady_prior.pt`

**Checkpoint loading pattern**:
1. Load from disk: `state = torch.load(path, map_location="cpu")`
2. Strip compiled prefixes: `state = _strip_compiled_prefix(state)`
3. Load into model: `model.load_state_dict(state)`
4. Move to device: `model.to(device)`

**Resume from WandB** (`train.py:1762-1796`):
```python
# CLI: python scripts/train.py --resume-from-wandb <run_id> --resume-mode allow
if args.resume_from_wandb:
    manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
    downloaded_files = manager.download_checkpoints_from_run(
        run_id=args.resume_from_wandb,
        checkpoint_files=None,
        force=False
    )
    manager.setup_wandb_resume(
        run_id=args.resume_from_wandb,
        resume_mode=args.resume_mode
    )
    manager.verify_checkpoints(["operator.pt", "operator_ema.pt"])
```

#### Checkpoint File Naming Conventions

**Standard checkpoints**:
- `operator.pt` - Best operator checkpoint
- `operator_ema.pt` - EMA version (preferred for evaluation)
- `diffusion_residual.pt` - Best diffusion checkpoint
- `diffusion_residual_ema.pt` - EMA version
- `steady_prior.pt` - Steady state prior checkpoint

**Metadata**:
- `checkpoints/metadata.json` - Training status, config hash, architecture fingerprint

**Scale statistics** (for multi-scale training):
- `checkpoints/scale/input_stats.pt`
- `checkpoints/scale/latent_stats.pt`

**Key insight**: Checkpoint names are **fixed** - there are no versioned checkpoints like `operator_v1.pt`, `operator_v2.pt`. Each stage overwrites its checkpoint on completion.

---

### 2. Multi-Stage Pipeline Progress Tracking

#### Stage Enumeration
**Location**: `scripts/train.py:1738-1742`

Four training stages are hardcoded:
1. `operator` - Deterministic latent evolution model
2. `diff_residual` - Diffusion residual for uncertainty/correction
3. `consistency_distill` - Few-step diffusion distillation
4. `steady_prior` - Steady-state prior (optional)
5. `all` - Run complete pipeline sequentially

#### Stage Configuration
**Location**: `scripts/train.py:391-397`

Each stage can be enabled/disabled via config:
```yaml
stages:
  operator:
    epochs: 25          # Run this stage
  diff_residual:
    epochs: 8           # Run this stage
  consistency_distill:
    epochs: 0           # Skip this stage
  steady_prior:
    epochs: 0           # Skip this stage
```

Function `_stage_epochs(cfg, stage)` reads this value and returns 0 for disabled stages.

#### Stage Completion Detection
**Current mechanism**: **Implicit checkpoint file existence**

The codebase does NOT have explicit stage status tracking (no `StageStatus` enum). Instead, completion is inferred from:

1. **Checkpoint file existence** (primary signal):
   - Operator complete: `operator.pt` or `operator_ema.pt` exists
   - Diffusion complete: `diffusion_residual.pt` or `diffusion_residual_ema.pt` exists
   - Steady prior complete: `steady_prior.pt` exists

2. **Metadata `trained` flag** (global completion only):
   - `checkpoints/metadata.json` → `"trained": true` indicates ALL training stages complete
   - Used by orchestrator to skip training phase entirely

3. **Config `epochs: 0`** (disabled stage):
   - Stage will be skipped during `train_all_stages()` execution

**Example implementation of stage detection** (not in codebase, but pattern used):
```python
def get_completed_stages(checkpoint_dir: Path) -> dict:
    completed = {
        "operator": False,
        "diff_residual": False,
        "consistency_distill": False,
        "steady_prior": False,
    }

    if (checkpoint_dir / "operator.pt").exists() or (checkpoint_dir / "operator_ema.pt").exists():
        completed["operator"] = True

    if (checkpoint_dir / "diffusion_residual.pt").exists() or (checkpoint_dir / "diffusion_residual_ema.pt").exists():
        completed["diff_residual"] = True
        completed["consistency_distill"] = True  # Overwrites same file

    if (checkpoint_dir / "steady_prior.pt").exists():
        completed["steady_prior"] = True

    return completed
```

#### Checkpoint Discovery Between Stages
**Location**: `scripts/run_fast_to_sota.py:343-348`

Stages discover each other's checkpoints via shared checkpoint directory:

```python
def _find_checkpoint(directory: Path, names: List[str]) -> Optional[Path]:
    """Find first existing checkpoint from a prioritized list."""
    for name in names:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None

# Usage:
operator_ckpt = _find_checkpoint(checkpoint_dir, ["operator_ema.pt", "operator.pt"])
diffusion_ckpt = _find_checkpoint(checkpoint_dir, ["diffusion_residual_ema.pt", "diffusion_residual.pt"])
```

**Pattern**: EMA checkpoints are always preferred over non-EMA versions.

#### Sequential Stage Execution
**Location**: `scripts/train.py:1516-1732`

The `train_all_stages()` function orchestrates sequential execution:

```python
def train_all_stages(cfg: dict, wandb_ctx=None) -> None:
    global_step = 0

    # Stage 1: Operator
    op_epochs = _stage_epochs(cfg, "operator")
    if op_epochs > 0:
        train_operator(cfg, wandb_ctx=wandb_ctx, global_step=global_step)
        global_step += op_epochs
    else:
        print("Skipping Operator (epochs<=0)")

    # Stage 2: Diffusion
    diff_epochs = _stage_epochs(cfg, "diff_residual")
    if diff_epochs > 0:
        train_diffusion(cfg, wandb_ctx=wandb_ctx, global_step=global_step)
        global_step += diff_epochs

    # ... similar for consistency and steady prior
```

**Key insight**: `global_step` counter provides continuous WandB x-axis across all stages in a single run.

---

### 3. VastAI Instance Resume Capabilities

#### Fresh Launch (Always Clears Checkpoints)
**Location**: `scripts/vast_launch.py:183-185` (in `generate_onstart_script()`)

**Behavior**:
```bash
rm -rf data/latent_cache checkpoints/scale || true
rm -f checkpoints/*.pt checkpoints/*.pth checkpoints/*.ckpt 2>/dev/null || true
rm -rf checkpoints || true
mkdir -p checkpoints data/latent_cache
```

**Key finding**: Fresh launches **ALWAYS clear checkpoints**. There is **no detection logic** for preserving existing checkpoints on fresh instances.

#### Resume Command (Existing Instances)
**Location**: `scripts/vast_launch.py:328-450`

The `resume` command is for **existing running instances**, not fresh launches:

**CLI**:
```bash
python scripts/vast_launch.py resume \
  --instance-id <INSTANCE_ID> \
  --config <CONFIG_PATH> \
  --resume-from-wandb <WANDB_RUN_ID> \
  --resume-mode allow \
  --stage all \
  --auto-shutdown
```

**Behavior**:
1. Gets SSH connection details for existing instance
2. Generates resume script that:
   - Pulls latest code (`git pull`)
   - **Does NOT clear checkpoints** (assumes instance already has them)
   - Calls `train.py` with `--resume-from-wandb` flags
3. Uploads and executes script via SSH/SCP

**Key insight**: Resume command assumes checkpoints are **already on the instance**. It's not for fresh launches.

#### Onstart Script Variations

**Regular onstart (fresh launch)** - Generated by `generate_onstart_script()`:
- Clears all checkpoints
- Downloads training data from B2
- Runs training from scratch

**Resume onstart** (`.vast/onstart_resume_checkpoint_manager.sh`):
- Does NOT clear cache: `# DO NOT RESET CACHE - We're resuming training!`
- Creates directories without clearing
- Calls `train.py` with `--resume-from-wandb` and `--resume-mode` flags
- CheckpointManager handles checkpoint downloading

**Manual resume onstart** (`.vast/onstart_resume_h200.sh`):
- Does NOT clear cache
- Contains inline Python script to download checkpoints from WandB
- Manually sets `WANDB_RUN_ID` and `WANDB_RESUME` environment variables
- Calls `run_fast_to_sota.py` instead of `train.py`

#### Checkpoint Metadata System
**Location**: `scripts/run_fast_to_sota.py:92-109`

**Created at pipeline start** (`_write_checkpoint_metadata`):
```json
{
  "schema_version": 1,
  "created_at": "2025-10-28T12:00:00",
  "config_hash": "abc123...",
  "config_path": "/path/to/config.yaml",
  "arch": {
    "latent_dim": 32,
    "latent_tokens": 64,
    "operator_hidden_dim": 96
  },
  "trained": false
}
```

**Updated throughout pipeline** (`_update_checkpoint_metadata`):
```json
{
  "trained": true,
  "trained_at": "2025-10-28T12:30:00",
  "last_small_eval": {"metric:nrmse": 0.09, ...},
  "last_small_eval_at": "2025-10-28T12:35:00",
  "last_full_eval": {"metric:nrmse": 0.08, ...},
  "last_full_eval_at": "2025-10-28T12:40:00",
  "training_wandb": {"id": "abc123", "project": "ups", ...}
}
```

**Used for skipping** (`run_fast_to_sota.py:687-693`):
```python
metadata_trained = bool(metadata.get("trained"))
skip_training_due_to_metadata = metadata_trained and not args.force_train
if skip_training_due_to_metadata:
    print("Skipping training: checkpoint metadata indicates training already completed.")
```

---

### 4. WandB Run and Checkpoint Tracking

#### WandBContext Class
**Location**: `src/ups/utils/wandb_context.py:27-271`

**Core logging methods**:
- `log_training_metric(stage, metric, value, step)` - Time series logging
- `log_eval_summary(metrics, prefix="eval")` - Final scalar summaries
- `log_table(name, columns, data)` - Table logging
- `log_image(name, image_or_path)` - Image visualization
- `save_file(path)` - File upload (used for checkpoints)
- `log_artifact(artifact)` - Artifact logging
- `update_config(updates)` - Config metadata updates
- `finish()` - Run cleanup

#### Checkpoint Upload to WandB
**Pattern** (repeated across all training stages):

1. Train stage and save best model to disk
2. Upload checkpoint to WandB:
   ```python
   operator_path = checkpoint_dir / "operator.pt"
   torch.save(operator.state_dict(), operator_path)
   wandb_ctx.save_file(operator_path)  # Upload to WandB
   ```

**Locations**:
- Operator: `train.py:669-682`
- Diffusion: `train.py:858-868`
- Consistency: `train.py:1177-1187`
- Steady prior: `train.py:1293-1299`

#### Checkpoint Download from WandB
**Implementation**: `checkpoint_manager.py:36-119`

Uses WandB API to download checkpoint files:
```python
api = wandb.Api()
run_path = f"{entity}/{project}/{run_id}"
run = api.run(run_path)

for file_obj in run.files():
    if file_obj.name in checkpoint_files:
        local_path = checkpoint_dir / file_obj.name
        if not local_path.exists() or force:
            file_obj.download(root=str(checkpoint_dir))
```

**Default checkpoint files downloaded**:
- `checkpoints/operator.pt`
- `checkpoints/operator_ema.pt`
- `checkpoints/diffusion_residual.pt`
- `checkpoints/diffusion_residual_ema.pt`
- `checkpoints/scale/input_stats.pt`
- `checkpoints/scale/latent_stats.pt`

#### WandB Run Resumption
**Environment variable setup** (`checkpoint_manager.py:121-134`):
```python
def setup_wandb_resume(self, run_id: str, resume_mode: str = "allow"):
    os.environ["WANDB_RUN_ID"] = run_id
    os.environ["WANDB_RESUME"] = resume_mode
```

**Resume modes**:
- `"allow"` - Resume if run exists, create new if not
- `"must"` - Must resume existing run, error if not found
- `"never"` - Never resume, always create new run

**WandB behavior**: When `WANDB_RUN_ID` and `WANDB_RESUME="allow"` are set, `wandb.init()` will append to the existing run instead of creating a new one.

#### Context File Communication (Subprocess Pattern)
**Location**: `scripts/run_fast_to_sota.py`

**Problem**: Orchestrator spawns training and evaluation as subprocesses. How to share WandB run?

**Solution**: Context file + environment variable

**Orchestrator saves context** (`train.py:1540-1543`):
```python
context_file_path = os.environ.get("WANDB_CONTEXT_FILE")
if context_file_path and wandb_ctx and wandb_ctx.enabled:
    save_wandb_context(wandb_ctx, Path(context_file_path))
```

**Orchestrator sets environment** (`run_fast_to_sota.py:706`):
```python
train_env = {
    "WANDB_CONTEXT_FILE": str(wandb_context_file)
}
_run_command(train_cmd, env=train_env, desc="train")
```

**Evaluation loads context** (`evaluate.py:497-501`):
```python
context_file = os.environ.get("WANDB_CONTEXT_FILE")
if context_file:
    wandb_ctx = load_wandb_context_from_file(Path(context_file))
```

**Result**: Single WandB run shared across entire pipeline (no run proliferation).

---

### 5. Standalone Evaluation Stage Execution

#### Command-Line Interface
**Location**: `scripts/evaluate.py`

**Required parameters**:
- `--operator` - Path to operator checkpoint (e.g., `checkpoints/operator_ema.pt`)

**Core parameters**:
- `--config` - Config file describing data/latent setup
- `--diffusion` - Optional diffusion checkpoint path
- `--tau` - Tau value for applying diffusion residual (default: 0.5)
- `--device` - Device for evaluation (default: "cpu")

**Output parameters**:
- `--output-prefix` - Prefix for saved reports (default: `reports/evaluation`)
- `--log-path` - Path to append evaluation logs (default: `reports/eval_log.jsonl`)
- `--print-json` - Print metrics as JSON

**Leaderboard parameters**:
- `--leaderboard-run-id` - If provided, append metrics to leaderboard
- `--leaderboard-path` - Leaderboard CSV path
- `--leaderboard-html` - Leaderboard HTML path
- `--leaderboard-label` - Label to record (e.g., "small_eval", "full_eval")
- `--leaderboard-tag` - Additional key=value pairs (repeatable)
- `--leaderboard-notes` - Optional notes

#### Two Execution Modes

**Mode 1: Standalone** (no WandB context):
```bash
python scripts/evaluate.py \
  --operator checkpoints/operator_ema.pt \
  --config configs/train_burgers_golden.yaml \
  --device cuda \
  --output-prefix reports/my_eval
```

**Produces**:
- `reports/my_eval.json` - Metrics JSON
- `reports/my_eval.csv` - Metrics CSV
- `reports/my_eval.html` - Interactive report
- `reports/my_eval_*.png` - Various plots
- No WandB logging

**Mode 2: Orchestrated** (with WandB context):
```bash
# Called by run_fast_to_sota.py with WANDB_CONTEXT_FILE env var
python scripts/evaluate.py \
  --operator checkpoints/operator_ema.pt \
  --config configs/train_burgers_golden.yaml \
  --device cuda \
  --output-prefix reports/eval
```

**Produces**:
- All local outputs (JSON, CSV, HTML, plots)
- WandB summary metrics logged to existing run
- WandB tables and images logged

**Key difference**: Presence of `WANDB_CONTEXT_FILE` environment variable determines mode.

#### Checkpoint Loading
**Location**: `evaluate.py:39-61`

**Robust loading function** (`_load_state_dict_compat`):
1. Loads checkpoint: `ckpt = torch.load(ckpt_path, map_location="cpu")`
2. Strips compiled prefixes (e.g., `_orig_mod.` from `torch.compile()`)
3. Handles both direct state_dict and nested formats
4. Loads into model: `model.load_state_dict(state_dict)`

#### Metadata Verification
**Location**: `evaluate.py:82-123`

**Checks performed**:
1. **Architecture fingerprint**: Verifies latent_dim, tokens, operator hidden_dim, num_heads, depths match
2. **Training status**: Ensures `metadata.trained == true`
3. **Config hash**: If config name matches training config, verifies SHA256 hash

**Example metadata check**:
```python
arch_from_meta = metadata["arch"]
arch_from_config = _extract_arch_fingerprint(cfg)

if arch_from_meta != arch_from_config:
    raise ValueError(f"Architecture mismatch: {arch_from_meta} vs {arch_from_config}")
```

#### Outputs Generated

**Always generated**:
- `<prefix>.json` - Metrics JSON
- `<prefix>.csv` - Metrics CSV
- `<prefix>.html` - Interactive HTML report
- `<prefix>.config.yaml` - Config snapshot

**Conditional outputs**:
- `<prefix>_ttc_step_logs.json` - TTC step-by-step logs (if TTC enabled)
- `<prefix>_ttc_rewards.png` - TTC reward trajectory plot
- `<prefix>_mse_hist.png` - MSE histogram
- `<prefix>_mae_hist.png` - MAE histogram
- `<prefix>_metrics.png` - Metrics bar chart
- `<prefix>_latent_heatmap.png` - Target vs Predicted heatmap
- `<prefix>_latent_spectrum.png` - Frequency spectrum comparison
- `<prefix>_preview.npz` - NumPy arrays of predicted/target latents

#### How Orchestrator Calls Evaluation
**Location**: `scripts/run_fast_to_sota.py:812-849` (small eval), `951-988` (full eval)

**Pattern**:
1. Discover checkpoints first: `operator_ckpt = _find_checkpoint(...)`
2. Build evaluation command with discovered checkpoint paths
3. Set `WANDB_CONTEXT_FILE` environment variable
4. Execute subprocess: `_run_command(eval_cmd, env=eval_env)`
5. Load metrics from output JSON file
6. Update metadata with cached results

**Example**:
```python
small_cmd = [
    PYTHON, "scripts/evaluate.py",
    "--config", str(resolved_small_config),
    "--operator", str(operator_ckpt),
    "--diffusion", str(diffusion_ckpt),
    "--device", args.eval_device,
    "--output-prefix", str(small_prefix),
    "--leaderboard-run-id", f"{run_id}_small",
    "--leaderboard-label", "small_eval",
]

eval_env = {"WANDB_CONTEXT_FILE": str(wandb_context_file)}
_run_command(small_cmd, env=eval_env, desc="evaluate-small")
```

---

### 6. Multi-Stage Orchestration and Resumption

#### Pipeline Phases
**Location**: `scripts/run_fast_to_sota.py`

The orchestrator controls **pipeline phases**, not training stages:
1. Validation (`--skip-validation`)
2. Data check (`--skip-data-check`)
3. Dry-run cost estimate (`--skip-dry-run`)
4. **Training** (`--skip-training`, `--force-train`)
5. **Small evaluation** (`--skip-small-eval`, `--redo-small-eval`)
6. **Full evaluation** (`--skip-full-eval`, `--redo-full-eval`, `--force-full-eval`)
7. Analysis (`--skip-analysis`)
8. Comparison (`--skip-comparison`)

**Key insight**: Training stage control (`operator`, `diff_residual`, etc.) is delegated to `train.py` via `--stage` flag.

#### Metadata-Based Result Caching
**Pattern used for small and full evaluation**:

```python
# Check if cached results exist
reuse_small = (metadata.get("last_small_eval") and not args.redo_small_eval)

if reuse_small:
    print("Reusing previous small-eval metrics from checkpoint metadata")
    small_flat = metadata["last_small_eval"]
else:
    # Run evaluation subprocess
    _run_command(eval_cmd, env=eval_env)

    # Load results and cache in metadata
    small_flat = _load_metrics(small_metrics_path)
    _update_checkpoint_metadata(metadata_path,
        last_small_eval=small_flat,
        last_small_eval_at=timestamp)
```

**Benefit**: Expensive evaluations are cached - subsequent runs reuse results unless explicitly forced to re-run.

#### Training Invocation
**Location**: `run_fast_to_sota.py:694-712`

```python
train_cmd = [
    PYTHON,
    "scripts/train.py",
    "--config", str(resolved_train_config),
    "--stage", args.train_stage,  # Default: "all"
]
train_cmd.extend(args.train_extra_arg)  # User-provided extras

train_env = {
    "WANDB_MODE": args.wandb_mode,
    "FAST_TO_SOTA_WANDB_INFO": str(wandb_info_path),
    "WANDB_CONTEXT_FILE": str(wandb_context_file),
}

_run_command(train_cmd, env=train_env, desc="train")

# After training, load WandB info
if wandb_info_path.exists():
    training_wandb_info = json.loads(wandb_info_path.read_text())
```

**Key flags**:
- `--train-stage` - Which stage to run (default: "all")
- `--train-extra-arg` - Additional arguments passed through to train.py (repeatable)

#### Gate-Based Flow Control
**Location**: `run_fast_to_sota.py:892-916`

Small evaluation results determine whether full evaluation runs:

```python
passed_small, messages = _check_gates(
    candidate=small_flat,
    baseline=baseline_metrics,
    improvement_metric="metric:nrmse",
    min_delta=0.01,
    ratio_limits={"metric:conservation_gap": 1.0}
)

if not passed_small:
    print("Proxy evaluation gates failed.")
    if not args.force_full_eval:
        should_run_full = False  # Skip full eval
```

**Purpose**: Avoid expensive full evaluation if small evaluation shows poor results.

#### Resumption Capabilities Summary

**Currently supported**:
- ✅ Skip completed training phase via metadata `trained` flag
- ✅ Resume from WandB run via `train.py --resume-from-wandb`
- ✅ Run specific training stage via `train.py --stage <stage>`
- ✅ Reuse cached evaluation results via metadata
- ✅ Force re-run phases via `--force-train`, `--redo-small-eval`, `--redo-full-eval`
- ✅ Standalone evaluation on any checkpoint via `evaluate.py`

**NOT currently supported**:
- ❌ Automatic detection of which stage to resume from
- ❌ `--resume-from-stage <stage>` flag in orchestrator
- ❌ Epoch-level resumption within a stage
- ❌ Optimizer state persistence across restarts
- ❌ Checkpoint versioning (e.g., `operator_v1.pt`, `operator_v2.pt`)

---

## Code References

**Core checkpoint management**:
- `src/ups/utils/checkpoint_manager.py:11-233` - CheckpointManager class
- `scripts/train.py:669-682` - Operator checkpoint saving
- `scripts/train.py:858-868` - Diffusion checkpoint saving
- `scripts/train.py:1762-1796` - Resume from WandB logic

**Stage tracking**:
- `scripts/train.py:391-397` - `_stage_epochs()` function
- `scripts/train.py:1516-1732` - `train_all_stages()` orchestration
- `scripts/run_fast_to_sota.py:343-348` - `_find_checkpoint()` discovery

**VastAI integration**:
- `scripts/vast_launch.py:183-185` - Fresh launch checkpoint clearing
- `scripts/vast_launch.py:328-450` - Resume command for existing instances
- `.vast/onstart_resume_checkpoint_manager.sh` - Resume onstart script

**WandB integration**:
- `src/ups/utils/wandb_context.py:27-271` - WandBContext class
- `src/ups/utils/wandb_context.py:455-530` - Context file serialization
- `scripts/evaluate.py:497-501` - WandB context loading

**Metadata tracking**:
- `scripts/run_fast_to_sota.py:92-109` - `_write_checkpoint_metadata()`
- `scripts/run_fast_to_sota.py:112-120` - `_update_checkpoint_metadata()`
- `scripts/run_fast_to_sota.py:687-693` - Metadata-based training skip logic

**Evaluation**:
- `scripts/evaluate.py:39-61` - Checkpoint loading
- `scripts/evaluate.py:82-123` - Metadata verification
- `scripts/run_fast_to_sota.py:812-849` - Small eval invocation
- `scripts/run_fast_to_sota.py:951-988` - Full eval invocation

---

## Architecture Documentation

### Current Checkpoint Flow

**Training pipeline**:
1. `run_fast_to_sota.py` creates checkpoint metadata
2. Calls `train.py --stage all` via subprocess
3. `train.py` runs stages sequentially:
   - Operator: saves `operator.pt` + `operator_ema.pt`
   - Diffusion: loads operator, saves `diffusion_residual.pt` + `diffusion_residual_ema.pt`
   - Consistency: loads both, overwrites `diffusion_residual.pt`
   - Steady prior: saves `steady_prior.pt`
4. Each checkpoint uploaded to WandB via `wandb_ctx.save_file()`
5. Metadata updated: `trained: true`

**Evaluation pipeline**:
1. `run_fast_to_sota.py` discovers checkpoints via `_find_checkpoint()`
2. Calls `evaluate.py` with checkpoint paths via subprocess
3. Evaluation loads checkpoints, runs inference, generates reports
4. Results cached in metadata: `last_small_eval`, `last_full_eval`

**Resume flow** (WandB-based):
1. User provides `--resume-from-wandb <run_id>`
2. `CheckpointManager` downloads checkpoints from WandB run
3. Sets `WANDB_RUN_ID` and `WANDB_RESUME` environment variables
4. Training loads checkpoints from disk, WandB appends to existing run
5. Training continues from checkpoint state

### Implicit vs Explicit Tracking

**Implicit (current)**:
- Stage completion inferred from checkpoint file existence
- No explicit stage status enum or tracking database
- Relies on file system as source of truth

**Explicit (not implemented)**:
- Could have `checkpoints/status.json`:
  ```json
  {
    "operator": {"status": "completed", "checkpoint": "operator_ema.pt", "completed_at": "..."},
    "diff_residual": {"status": "completed", "checkpoint": "diffusion_residual_ema.pt", "completed_at": "..."},
    "consistency_distill": {"status": "in_progress", "epoch": 3, "total_epochs": 8},
    "steady_prior": {"status": "not_started"}
  }
  ```

**Recommendation**: Implement explicit stage tracking to enable intelligent resumption.

---

## Historical Context

### From thoughts/ Directory

**UPT Implementation Plan** (`thoughts/shared/plans/2025-01-23-upt-inverse-losses-implementation.md`):
- Documents `--stage` flag for stage-specific execution
- Mentions pause points after each implementation phase
- Discusses checkpoint resumption: "Can resume from Phase 1 checkpoint if dimensions match"
- Suggests adding `--resume-from <checkpoint>` option to training script
- Shows commands like: `python scripts/train.py --config <config> --stage operator`

**Key insight**: The plan document suggested `--resume-from <checkpoint>` flag, but the implemented solution was `--resume-from-wandb <run_id>` (downloads checkpoints from WandB instead of loading local path).

---

## Open Questions

### 1. How should automatic stage resumption work?

**Scenario**: Training crashes after operator completes. User relaunches with same config.

**Current behavior**: Reruns all stages from scratch (fresh launch clears checkpoints).

**Desired behavior**:
- Detect operator checkpoint exists
- Skip operator stage automatically
- Resume from diffusion stage

**Implementation options**:

**Option A: Explicit stage status file**
```json
// checkpoints/status.json
{
  "stages": {
    "operator": {"status": "completed", "completed_at": "2025-10-28T10:30:00"},
    "diff_residual": {"status": "not_started"}
  }
}
```

**Option B: Smart config override**
```python
# Auto-detect completed stages and set epochs=0
completed_stages = get_completed_stages(checkpoint_dir)
for stage, is_complete in completed_stages.items():
    if is_complete and cfg["stages"][stage]["epochs"] > 0:
        print(f"Detected {stage} checkpoint - setting epochs=0 to skip")
        cfg["stages"][stage]["epochs"] = 0
```

**Option C: Resume-from-checkpoint flag**
```bash
python scripts/train.py --config config.yaml --stage all --resume-from-checkpoint checkpoints/
# Automatically detects which stages are complete and skips them
```

### 2. Should we support epoch-level resumption within stages?

**Current**: Each stage restarts from epoch 0 on interruption.

**Pros of epoch-level resumption**:
- Save cost on long-running stages (e.g., 50-epoch operator training)
- More resilient to transient failures

**Cons**:
- Requires saving optimizer state, scheduler state, random seeds
- Larger checkpoint files
- More complex loading logic

**Recommendation**: Start with stage-level resumption, add epoch-level if needed.

### 3. How should VastAI fresh launches handle resume?

**Current**: Fresh launches always clear checkpoints.

**Options**:

**Option A: Add `--resume-from-wandb` to launch command**
```bash
python scripts/vast_launch.py launch \
  --config config.yaml \
  --resume-from-wandb abc123 \
  --auto-shutdown
# Onstart script downloads checkpoints before training
```

**Option B: Separate "resume launch" command**
```bash
python scripts/vast_launch.py resume-launch \
  --config config.yaml \
  --wandb-run abc123 \
  --auto-shutdown
# Uses specialized onstart script that preserves/downloads checkpoints
```

**Option C: Detect from config metadata**
```bash
# If config has metadata.json with wandb run info, automatically resume
python scripts/vast_launch.py launch --config config.yaml
# Checks if checkpoints/metadata.json exists with training_wandb info
# If yes, downloads checkpoints from that run
```

### 4. What's the best UX for running evaluation only?

**Current options**:

**Option A: Skip training flag**
```bash
python scripts/run_fast_to_sota.py \
  --train-config config.yaml \
  --skip-training \
  --small-eval-config small.yaml
```

**Option B: Standalone evaluate.py**
```bash
python scripts/evaluate.py \
  --operator checkpoints/operator_ema.pt \
  --diffusion checkpoints/diffusion_residual_ema.pt \
  --config config.yaml
```

**Option C: Dedicated eval-only command**
```bash
python scripts/run_fast_to_sota.py eval-only \
  --checkpoint-dir checkpoints/ \
  --small-eval-config small.yaml \
  --full-eval-config full.yaml
```

**Recommendation**: Option B (standalone) is most flexible, Option A (skip-training) is best for full pipeline integration.

---

## Related Research

No existing research documents found in `thoughts/shared/research/` on this topic.

Related documentation:
- `thoughts/shared/plans/2025-01-23-upt-inverse-losses-implementation.md` - UPT plan with stage execution patterns
- `CLAUDE.md` - Project instructions with VastAI workflow
- `PRODUCTION_WORKFLOW.md` - VastAI training workflow
- `docs/production_playbook.md` - Best practices

---

## Implementation Recommendations

Based on the comprehensive research above, here are concrete recommendations for implementing optimal resuming of remote training instances:

### 1. Add Explicit Stage Status Tracking

**File**: `src/ups/utils/stage_tracker.py` (new)

```python
@dataclass
class StageStatus:
    status: Literal["not_started", "in_progress", "completed", "failed"]
    checkpoint: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    epoch: Optional[int]
    total_epochs: Optional[int]

class StageTracker:
    def __init__(self, checkpoint_dir: Path):
        self.status_file = checkpoint_dir / "stage_status.json"

    def get_stage_status(self, stage: str) -> StageStatus:
        """Get current status of a training stage."""
        pass

    def mark_stage_started(self, stage: str, total_epochs: int):
        """Mark stage as started."""
        pass

    def mark_stage_completed(self, stage: str, checkpoint: str):
        """Mark stage as completed."""
        pass

    def get_next_stage_to_run(self, cfg: dict) -> Optional[str]:
        """Determine which stage should run next."""
        pass
```

**Update**: `scripts/train.py` to use StageTracker:
```python
# At start of train_all_stages()
tracker = StageTracker(checkpoint_dir)

# Before each stage
next_stage = tracker.get_next_stage_to_run(cfg)
if next_stage == "operator":
    tracker.mark_stage_started("operator", op_epochs)
    train_operator(cfg, wandb_ctx, global_step)
    tracker.mark_stage_completed("operator", "operator_ema.pt")
```

### 2. Add Resume-Aware VastAI Launch

**Update**: `scripts/vast_launch.py` to add `--resume-from-wandb` flag:

```python
parser.add_argument("--resume-from-wandb", type=str,
    help="WandB run ID to resume from (downloads checkpoints)")
parser.add_argument("--resume-mode", default="allow",
    choices=["allow", "must", "never"])
```

**Update**: `generate_onstart_script()` to conditionally skip checkpoint clearing:

```python
def generate_onstart_script(..., resume_from_wandb=None, resume_mode="allow"):
    if resume_from_wandb:
        script += """
# RESUME MODE - Download checkpoints from WandB
echo "Resuming from WandB run: {resume_from_wandb}"
python -c "
from ups.utils.checkpoint_manager import CheckpointManager
manager = CheckpointManager('checkpoints')
manager.download_checkpoints_from_run('{resume_from_wandb}')
manager.setup_wandb_resume('{resume_from_wandb}', '{resume_mode}')
"
"""
    else:
        script += """
# FRESH START - Clear existing checkpoints
rm -rf checkpoints data/latent_cache
mkdir -p checkpoints data/latent_cache
"""
```

### 3. Add Smart Stage Resumption to train.py

**Update**: `scripts/train.py` to add `--auto-resume` flag:

```python
parser.add_argument("--auto-resume", action="store_true",
    help="Automatically detect completed stages and skip them")
```

**Implementation**:
```python
if args.auto_resume:
    tracker = StageTracker(checkpoint_dir)

    # Override config epochs for completed stages
    for stage_name in ["operator", "diff_residual", "consistency_distill", "steady_prior"]:
        status = tracker.get_stage_status(stage_name)
        if status.status == "completed":
            print(f"Auto-resume: Detected {stage_name} checkpoint - skipping")
            if "stages" not in cfg:
                cfg["stages"] = {}
            if stage_name not in cfg["stages"]:
                cfg["stages"][stage_name] = {}
            cfg["stages"][stage_name]["epochs"] = 0
```

### 4. Add Eval-Only Mode to Orchestrator

**Update**: `scripts/run_fast_to_sota.py` to add eval-only subcommand:

```python
subparsers = parser.add_subparsers(dest="command")

# Main pipeline (default)
pipeline_parser = subparsers.add_parser("pipeline", help="Run full pipeline")
# ... existing arguments

# Eval-only mode
eval_parser = subparsers.add_parser("eval-only", help="Run evaluation only")
eval_parser.add_argument("--checkpoint-dir", required=True)
eval_parser.add_argument("--small-eval-config")
eval_parser.add_argument("--full-eval-config")
# ... other eval args
```

**Implementation**:
```python
if args.command == "eval-only":
    # Discover checkpoints
    checkpoint_dir = Path(args.checkpoint_dir)
    operator_ckpt = _find_checkpoint(checkpoint_dir, ["operator_ema.pt", "operator.pt"])
    diffusion_ckpt = _find_checkpoint(checkpoint_dir, ["diffusion_residual_ema.pt", "diffusion_residual.pt"])

    # Run small eval
    if args.small_eval_config:
        run_evaluation(operator_ckpt, diffusion_ckpt, args.small_eval_config, "small")

    # Run full eval
    if args.full_eval_config:
        run_evaluation(operator_ckpt, diffusion_ckpt, args.full_eval_config, "full")
```

### 5. Add Stage Status Dashboard

**New script**: `scripts/show_training_status.py`

```python
"""Show current training status for a checkpoint directory."""

def show_status(checkpoint_dir: Path):
    tracker = StageTracker(checkpoint_dir)
    metadata = load_metadata(checkpoint_dir)

    print("\nTraining Status")
    print("=" * 60)

    for stage in ["operator", "diff_residual", "consistency_distill", "steady_prior"]:
        status = tracker.get_stage_status(stage)
        print(f"{stage:20} {status.status:15} {status.checkpoint or 'N/A'}")

    print("\nPipeline Status")
    print("=" * 60)
    print(f"Training complete: {metadata.get('trained', False)}")
    print(f"Small eval run: {metadata.get('last_small_eval_at', 'Never')}")
    print(f"Full eval run: {metadata.get('last_full_eval_at', 'Never')}")
```

**CLI**:
```bash
python scripts/show_training_status.py checkpoints/
```

### 6. Update Documentation

**Create**: `docs/checkpoint_resume_guide.md`

**Sections**:
1. How checkpoints are saved and tracked
2. How to resume training after interruption
3. How to resume on VastAI instances
4. How to run evaluation only
5. How to skip completed stages
6. Troubleshooting common issues

### Summary of Recommendations

**Priority 1 (Essential)**:
1. ✅ Explicit stage status tracking (`StageTracker`)
2. ✅ Resume-aware VastAI launch (`--resume-from-wandb`)
3. ✅ Smart stage resumption (`--auto-resume`)

**Priority 2 (High value)**:
4. ✅ Eval-only mode in orchestrator
5. ✅ Stage status dashboard script

**Priority 3 (Nice to have)**:
6. ✅ Comprehensive checkpoint/resume documentation
7. ✅ Epoch-level resumption within stages
8. ✅ Checkpoint versioning system

These recommendations will enable **optimal resuming of remote training instances** with minimal manual intervention.

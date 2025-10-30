# Checkpoint and Resume Guide

This guide explains how to use the intelligent checkpoint resume system for training pipelines.

## Overview

The checkpoint resume system enables:
- Automatic detection of completed training stages
- Seamless resumption after crashes or interruptions
- Resume training on fresh VastAI instances from WandB checkpoints
- Clear observability of pipeline status

## Key Concepts

### Training Stages

Training happens in four sequential stages:
1. **operator** - Deterministic latent evolution model
2. **diff_residual** - Diffusion residual for uncertainty
3. **consistency_distill** - Few-step diffusion distillation
4. **steady_prior** - Steady-state prior (optional)

### Stage Status Tracking

Stage status is tracked in `checkpoints/stage_status.json`:

```json
{
  "schema_version": 1,
  "created_at": "2025-10-28T12:00:00Z",
  "stages": {
    "operator": {
      "status": "completed",
      "checkpoint": "operator_ema.pt",
      "completed_at": "2025-10-28T12:15:00Z"
    },
    "diff_residual": {
      "status": "in_progress",
      "started_at": "2025-10-28T12:16:00Z",
      "epoch": 3,
      "total_epochs": 8
    }
  }
}
```

Status values:
- `not_started` - Stage hasn't run yet
- `in_progress` - Stage is currently running
- `completed` - Stage finished successfully
- `failed` - Stage encountered an error

## Usage Scenarios

### Scenario 1: Resume After Local Crash

Training crashes after operator completes:

```bash
# First run (crashes after operator)
python scripts/train.py --config configs/train_burgers_32dim.yaml --stage all

# Check status
python scripts/show_training_status.py checkpoints/

# Resume with auto-resume flag
python scripts/train.py \
  --config configs/train_burgers_32dim.yaml \
  --stage all \
  --auto-resume
```

The `--auto-resume` flag:
- Detects completed stages from `stage_status.json`
- Automatically sets `epochs: 0` for completed stages
- Skips them and continues from next incomplete stage

### Scenario 2: Resume on Fresh VastAI Instance

Training was interrupted on a VastAI instance. Resume on a new instance:

```bash
# Get WandB run ID from previous training
# (check WandB dashboard or logs)

# Launch new instance with resume
python scripts/vast_launch.py launch \
  --config configs/train_burgers_32dim.yaml \
  --resume-from-wandb train-20251028_120000 \
  --resume-mode allow \
  --auto-shutdown
```

This will:
1. Launch fresh VastAI instance
2. Download checkpoints from WandB run
3. Set up WandB to resume existing run (not create new one)
4. Run training with `--auto-resume` flag
5. Auto-shutdown after completion

### Scenario 3: Check Training Status

View current pipeline status:

```bash
python scripts/show_training_status.py checkpoints/
```

Output:
```
================================================================================
TRAINING PIPELINE STATUS
================================================================================
Checkpoint Directory: /path/to/checkpoints

Training Stages:
--------------------------------------------------------------------------------
Stage                     Status          Checkpoint                Completed
--------------------------------------------------------------------------------
operator                  ✅ completed    operator_ema.pt           2025-10-28 12:15:00 UTC
diff_residual             🔄 in_progress  N/A                       N/A
consistency_distill       ⏸️  not_started  N/A                       N/A
steady_prior              ⏸️  not_started  N/A                       N/A

Pipeline Status:
--------------------------------------------------------------------------------
Training Complete:        ❌ No
Small Eval:               ❌ Not run
Full Eval:                ❌ Not run

WandB Run:
--------------------------------------------------------------------------------
Run ID:                   train-20251028_120000
Run Name:                 jolly-mountain-42
Project:                  universal-simulator/emgun
URL:                      https://wandb.ai/emgun/universal-simulator/runs/...

Checkpoint Files:
--------------------------------------------------------------------------------
  ✅ operator.pt                    (145.2 MB)
  ✅ operator_ema.pt                (145.2 MB)
  ❌ diffusion_residual.pt          (not found)
  ❌ diffusion_residual_ema.pt      (not found)
  ❌ steady_prior.pt                (not found)
================================================================================
```

### Scenario 4: Run Evaluation Only

Run evaluation without training:

**Option A: Standalone evaluation**
```bash
python scripts/evaluate.py \
  --operator checkpoints/operator_ema.pt \
  --diffusion checkpoints/diffusion_residual_ema.pt \
  --config configs/train_burgers_32dim.yaml \
  --device cuda \
  --output-prefix reports/my_eval
```

**Option B: Via orchestrator with skip-training**
```bash
python scripts/run_fast_to_sota.py \
  --train-config configs/train_burgers_32dim.yaml \
  --skip-training \
  --small-eval-config configs/small_eval.yaml \
  --full-eval-config configs/full_eval.yaml
```

## Resume Modes

When resuming from WandB, you can specify resume mode:

- `allow` (default) - Resume if run exists, create new if not
- `must` - Must resume existing run, error if not found
- `never` - Never resume, always create new run

Example:
```bash
python scripts/vast_launch.py launch \
  --resume-from-wandb train-20251028_120000 \
  --resume-mode must  # Fail if run doesn't exist
```

## Troubleshooting

### "Stage status file not found"

If you see this warning, it means the checkpoint directory is from before stage tracking was implemented. You have two options:

1. Start fresh with new training run
2. Manually create `checkpoints/stage_status.json` based on which checkpoints exist

### "Checkpoint architecture mismatch"

This error occurs when trying to evaluate a checkpoint with a different config:

```
ValueError: Architecture mismatch: latent_dim=32 vs latent_dim=64
```

**Solution**: Use the same config that was used for training.

### "WandB run not found"

When using `--resume-from-wandb`, ensure:
- WandB run ID is correct (check WandB dashboard)
- `WANDB_API_KEY` environment variable is set
- You have access to the WandB project

### Auto-resume not detecting completed stages

Check that:
- `checkpoints/stage_status.json` exists and is valid JSON
- Stage status is actually "completed" (not "in_progress" or "failed")
- You're using the `--auto-resume` flag

## Best Practices

1. **Always use --auto-resume for production runs**: This ensures training resumes correctly after any interruption

2. **Check status before resuming**: Run `show_training_status.py` to verify which stages are complete

3. **Use --resume-mode must for critical runs**: This ensures you don't accidentally create a new WandB run

4. **Keep WandB run IDs handy**: Save run IDs in experiment notes for easy resumption

5. **Test locally before VastAI**: Run a quick local training test with auto-resume before launching expensive VastAI instances

## Related Documentation

- [VastAI Workflow](../PRODUCTION_WORKFLOW.md) - VastAI training workflow
- [Production Playbook](production_playbook.md) - Best practices
- [Research Document](../thoughts/shared/research/2025-10-28-checkpoint-resume-system.md) - Technical details

# Overnight SOTA Push — 2025-10-20

**Launch Time:** 2025-10-20 23:55 UTC (initial attempt)  
**Relaunches:**
  - 2025-10-21 03:28 UTC (instance 27071621) — failed due to encoder device mismatch
  - 2025-10-21 04:02 UTC (instance 27072060) — includes encoder device fix
**Active Instance:** VastAI `27072060` (RTX 5880 Ada, 64 GB)  
**Entry Point:** `python scripts/vast_launch.py launch --gpu RTX_5880Ada --config configs/rerun_txxoc8a8_capacity.yaml --auto-shutdown ...`

## Objectives
- Train an enlarged 64‑dim latent teacher with extended operator/diffusion schedules.
- Target post-distill small/full eval `nRMSE ≤ 0.05`, with stretch goal `< 0.035`.
- Provide clean checkpoints for the follow-up distill-only sweep (`configs/rerun_txxoc8a8_distillstretch.yaml`).

## Config Highlights (`configs/rerun_txxoc8a8_capacity.yaml`)
- Latent: `dim=64`, `tokens=32`
- Operator: `hidden_dim=128`, `num_heads=8`, `group_size=16`
- LR schedule: cosine (`lr=7.5e-4`, `eta_min=1.5e-4`, `epochs=20`)
- Diffusion: `hidden_dim=128`, `epochs=10`
- Distill staging: `distill_micro_batch=6`, `distill_num_taus=6`
- Evaluation uses small/full configs from the baseline rerun (valid/test splits)

## Run Arguments
- `--wandb-run-name=rerun-capacity`
- `--tag=config=rerun_txxoc8a8_capacity`
- `--leaderboard-wandb` (project `universal-simulator`, entity `emgun-morpheus-space`)
- Explicit small/full eval config overrides

## Monitoring & Expected Timeline
| Stage | Duration (est.) | Notes |
|-------|-----------------|-------|
| Latent cache | 4–5 min | larger dataset download, watch rclone logs |
| Operator stage (20 epochs) | ~55 min | aim for `loss ≤ 3e-4`, monitor grad_norm after LR decay |
| Diffusion stage (10 epochs) | ~30 min | expect `best_loss ≤ 4e-3` |
| Consistency distill (6 epochs) | ~60 min | target `best_loss ≤ 5e-5` |
| Eval + artifact upload | ~25 min | small/full eval on valid/test splits |

Total runtime ≈ 2.5–3.0 hr. Instance auto-shutdown on completion.

## Follow-up Actions (Tomorrow)
1. Inspect orchestrator run (`rerun-capacity`) for:
   - Operator/diffusion losses
   - Small/full eval nRMSE and physics gates
   - TTC metrics table (now logged inline)
2. If teacher meets gates, run distill-only job:  
   `python scripts/run_fast_to_sota.py --train-config configs/rerun_txxoc8a8_distillstretch.yaml --skip-training --redo-small-eval --redo-full-eval --wandb-run-name rerun-distillstretch`
3. Queue TTC cadence sweep using the new checkpoints if full eval nRMSE ≤ 0.05.

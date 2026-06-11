# P1.2 Capacity Sweep Results: Rollout Stability, Not Capacity, Is the Bottleneck

Date: 2026-06-10

Status: research evidence for north-star roadmap Phase 1 (P1.2/P1.3). Validation-only; no held-out test data was read and no ledger was written. Not claim evidence.

## Protocol

- Data: medium-v1 (`b2://pdebench/medium-v1`), tasks advection1d/burgers1d/darcy2d, 512 train samples, 128 validation samples, test split never fetched.
- Eval: 16-step decoded rollout on validation, pure model prediction (`decoded_persistence_residual_alpha = 1.0` default), no roll-shift or context estimators.
- Training per tier: stages operator(12) / decoder(6) / operator_decoded(6) / joint_codec_operator(4) epochs, patience 3, AdamW lr 1e-3, batch 16 (tier_c 4, tier_d 2 for 24GB VRAM).
- Hardware: single RTX 4090 instances on Vast.ai; runner `scripts/run_remote_capacity_sweep.sh`; published artifact `b2://pdebench/remote-runs/capacity-sweep/capacity_sweep_medium-v1_20260610T235516Z.tar.gz`; metrics JSON committed at `docs/research/artifacts/p1_capacity_sweep_medium_v1_val.json`.

## Results (validation `decoded_rollout_nrmse`)

| run | params | overall | advection | burgers | darcy | h1 | h16 |
|---|---|---|---|---|---|---|---|
| persistence_medium_v1_val | — | 0.3826 | 0.5339 | 0.1509 | 0.2200 | 0.5240 | 0.3710 |
| current | 33.8K | 0.8006 | 0.9093 | 0.7088 | 0.5336 | 0.4831 | 0.8630 |
| tier_a | 199K | 0.7993 | 0.9137 | 0.7007 | 0.5283 | 0.5167 | 0.8201 |
| tier_b | 759K | 0.7449 | 0.8624 | 0.6322 | 0.5587 | 0.5021 | 0.7723 |
| tier_c | 4.3M | 0.7472 | 0.8570 | 0.6419 | 0.5872 | 0.5042 | 0.7749 |
| tier_d | 12.6M | 0.9275 | 1.0139 | 0.8519 | 0.7855 | 0.5034 | 1.1068 |

Operator parameter counts measured by instantiating `LatentOperator` locally; tier specs are in `scripts/run_remote_capacity_sweep.sh`.

## Findings

1. **Single-step competence, rollout collapse.** Every tier matches or slightly beats persistence at horizon 1 (≈0.48–0.52 vs 0.524) and then compounds error to 2–3× persistence by horizon 16 (0.77–1.11 vs 0.371). The learned operator can predict one step; autoregressive rollout destroys it. This is the same long-horizon signature as the historical advection phase failure, now measured across all three tasks.
2. **Capacity saturates at ~750K under this recipe.** current→tier_a (6× params) is flat; tier_b improves 7%; tier_b→tier_c (5.6× params, and 4× more gradient steps due to batch 4) is flat. Raw parameters and raw step count are both exhausted.
3. **tier_d regressed sharply** (0.9275, h16 1.107). Most plausible cause: batch 2 with unscaled lr 1e-3 destabilized training. Treat as a recipe artifact, not a capacity conclusion.
4. **Persistence-friendly tasks are where the model loses worst.** Burgers (persistence 0.151, best model 0.632) and Darcy (0.220 vs 0.528): the model injects drift into systems that barely change. Error accumulation, not expressiveness, is the gap.

## Decision

- Gate G1 (beat persistence at validation) will not fall to capacity scaling under the current training recipe. Capacity work is paused at tier_b/tier_c scale.
- The next Phase 1 experiment is a **rollout-stability recipe sweep at fixed tier_b capacity**: longer-horizon decoded rollout training pressure, semigroup/composition consistency (explore bet E1, levers already in `scripts/train.py`: `training_rollout_steps`, `rollout_loss_horizon_power`, `lambda_semigroup`), longer training budgets, and lr scaled to batch.
- tier_d should be retried only after a stable recipe exists, with lr/batch scaling handled explicitly.

## Cost

Five GPU instances across the sweep (including one alpha-bug abort, one out-of-credit kill, and one stuck host): ≈6 GPU-hours, ≈$2.50. Cumulative Phase 1 spend ≈9 GPU-hours, ≈$4 — within the 50–100 GPU-hour Phase 1 budget.

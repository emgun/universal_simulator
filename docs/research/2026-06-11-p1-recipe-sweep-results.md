# P1 Recipe Sweep Results: Stability Recipes Did Not Beat Tier B

Date: 2026-06-11

Status: research evidence for north-star roadmap Phase 1 and explore bet E1. Validation-only; no held-out test data was read and no ledger was written. Not claim evidence.

## Protocol

- Data: medium-v1 (`b2://pdebench/medium-v1`), tasks advection1d/burgers1d/darcy2d, 512 train samples, 128 validation samples, test split never fetched.
- Eval: 16-step decoded rollout on validation, pure model prediction (`decoded_persistence_residual_alpha = 1.0` default), no roll-shift or context estimators.
- Fixed capacity: tier_b from the P1.2 capacity sweep, latent dim 64, 64 latent tokens, operator hidden dim 128, operator depths `[2,2,2]`, decoder hidden dim 128.
- Training variants: rollout horizon 8, rollout horizon 16, horizon-weighted rollout loss, stronger semigroup loss, longer training budget, and the combined recipe.
- Hardware: single RTX 4090 Vast.ai instance, valid run relaunched at batch size 4 after the default batch size 16 produced CUDA OOM skipped batches.
- Published artifact: `b2://pdebench/remote-runs/recipe-sweep/recipe_sweep_medium-v1_20260611T185755Z.tar.gz`.
- Verified local tarball SHA256: `521955d442c86c681cd5a19a30ff1ede197e24c08a15bee19b42eb6de86b3c93`.
- Metrics JSON committed at `docs/research/artifacts/p1_recipe_sweep_medium_v1_val.json`.

## Results

Validation `decoded_rollout_nrmse`; lower is better.

| run | overall | advection | burgers | darcy | h16 |
|---|---:|---:|---:|---:|---:|
| persistence_medium_v1_val | 0.3826 | 0.5339 | 0.1509 | 0.2200 | 0.3710 |
| tier_b capacity reference | 0.7449 | 0.8624 | 0.6322 | 0.5587 | 0.7723 |
| r_hpower | 0.7621 | 0.9470 | 0.5587 | 0.4988 | 0.7972 |
| r_rollout16 | 0.7872 | 0.9350 | 0.6439 | 0.5077 | 0.8017 |
| r_rollout8 | 0.8012 | 0.9232 | 0.6942 | 0.5133 | 0.8051 |
| r_semigroup | 0.8024 | 0.9099 | 0.7134 | 0.5213 | 0.8127 |
| r_combo | 0.8114 | 0.9677 | 0.6602 | 0.5007 | 0.8333 |
| r_long | 0.8380 | 0.9291 | 0.7704 | 0.5479 | 0.8955 |

## Findings

1. **The pre-registered kill condition is met.** Best recipe `r_hpower` scored `0.7620583413339258`, worse than tier_b capacity `0.7449043873888164` by `0.017153953945109435` absolute and worse than persistence `0.38260034902058476` by `0.37945799231334104` absolute.
2. **The combined recipe did not rescue rollout stability.** `r_combo` finished at `0.8113905817903249`, worse than the simpler horizon-weighted recipe and worse than the tier_b baseline.
3. **Horizon 16 stayed worse than tier_b.** Best h16 was `r_hpower=0.7972030250918781` versus tier_b `0.772299933386223`; the core failure remains autoregressive rollout compounding.
4. **Task movement is not aligned with the gate.** Some recipes improve Burgers/Darcy relative to tier_b, but they do so while worsening advection enough to miss the aggregate gate and persistence by a large margin.
5. **The protocol stayed clean.** The remote data directory contained only train/val shards, all six summaries report validation split metadata, and the self-summary reports `held_out_test_data_read=false`.

## Decision

- Do not promote any recipe and do not spend held-out test budget.
- Stop this fixed-tier_b rollout-stability recipe line. More horizon weighting, semigroup weight, or longer training at this architecture/scale is not the next high-signal path.
- The next useful Phase 1 exploit step is a data-budget sweep only if it is treated as the remaining scale-axis check, not as another recipe tweak.
- If the data-budget sweep does not materially close the gap to persistence, move the in-house core to explore-track status and prioritize the backbone-transplant or physics-primitive paths from the north-star roadmap.


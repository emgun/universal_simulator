# P1.3 Recipe Sweep Results: Rollout Collapse Is Structural — Invoke the Phase 2 Fallback

Date: 2026-06-11

Status: research evidence for north-star roadmap Phase 1 (P1.3 / explore bet E1). Validation-only; no held-out test data was read and no ledger was written. Not claim evidence.

## Protocol

- Fixed tier_b capacity (latent 64, hidden 128, depths [2,2,2], tokens 64, ~759K operator params), medium-v1 train(512)/val(128), 16-step decoded rollout, pure model prediction, no estimators; runner `scripts/run_remote_recipe_sweep.sh`.
- Six recipes varying decoded/joint `rollout_steps` (8/16), `rollout_loss_horizon_power` (2.0), `training.lambda_semigroup` (0.3 vs default 0.05), a 3x epoch budget, and a combination.
- `r_combo` did not complete: the Vast account ran out of credit during its training. Overall metrics below are from run summaries in the instance logs; h1/h16 are from the W&B run summaries (project `universal-simulator`).

## Results (validation `decoded_rollout_nrmse`)

| run | overall | h1 | h16 |
|---|---|---|---|
| persistence_medium_v1_val (reference) | 0.3826 | 0.5240 | 0.3710 |
| tier_b baseline (P1.2 recipe) | 0.7449 | 0.5021 | 0.7723 |
| r_rollout8 | 0.7532 | 0.4981 | 0.7498 |
| r_rollout16 | 0.7405 | 0.4872 | 0.7520 |
| r_hpower | 0.7343 | 0.4924 | 0.7517 |
| r_semigroup | 0.7449 | 0.5021 | 0.7723 |
| r_long | 0.7806 | 0.5100 | 0.8080 |
| r_combo | incomplete (out of credit) | — | — |

## Findings

1. **No recipe lever moves the rollout collapse.** Best overall improvement is 1.4% (`r_hpower`); h16 improves at most ~3% (0.772 → 0.750) against a 2.1x gap to persistence. Direct multi-step rollout training pressure — the canonical fix for exposure bias — barely registers.
2. **Semigroup consistency at lambda 0.3 changes nothing** (identical metrics to baseline), suggesting the operator already satisfies the consistency it can express.
3. **A 3x training budget makes things worse** (0.7806, h16 0.808): the model overfits one-step dynamics rather than learning stable rollouts.
4. Combined with the P1.2 capacity sweep (33.8K → 12.6M params flat-to-worse), the conclusion is that the h1-competent/h16-collapsing behavior is **structural to the current operator/decoder stack and objective family**, not a capacity or recipe deficiency.

## Decision: invoke the Phase 1 fallback early

The north-star roadmap's Phase 1 fallback reads: "if G1 fails after the capacity and data sweeps, that is strong evidence the current core architecture is the problem — skip directly to Phase 2 (transplant) and treat the in-house core as explore-track only."

We invoke it early, before a data-budget sweep, with this justification: the model loses worst on persistence-friendly, near-static tasks (Burgers persistence 0.151 vs model ~0.63; Darcy 0.220 vs ~0.53), where error comes from self-injected drift, not from under-sampled dynamics. More trajectories do not fix drift; the failure is independent of train-set size in the regime where one-step error is already at the persistence level. Total Phase 1 spend to reach this conclusion: ~10 GPU-hours / ~$9, versus the 50-100 GPU-hour budget.

Next steps:

- **Phase 2 (P2.1)**: Poseidon/DPOT adapter design doc — CPU/paper work, no GPU required. Then frozen-backbone fine-tune against roadmap Gate G2a (validation <= 0.3634).
- **Explore track**: E2 (spectral refiner on decoded outputs) and E3 (physics-primitive core) remain the in-house bets consistent with this evidence; E1 (semigroup/horizon objectives) is now measured and killed at this scale per its pre-registered kill criterion (< 5% after tuning).
- `r_combo` is not worth completing: its components are individually flat-to-negative.

## Cost note

The sweep was interrupted once by a broken-host launch wrapper (instance idled ~14h, ~$4) and terminated by credit exhaustion during `r_combo`. Cumulative Phase 1 spend: ~$9. Vast balance at time of writing: $0 — Phase 2 GPU work (transplant fine-tunes, ~$5-15) requires a top-up; P2.1 design work does not.

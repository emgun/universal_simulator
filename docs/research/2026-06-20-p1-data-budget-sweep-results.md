# P1 Data-Budget Sweep Results: More Samples Did Not Close the Gap

Date: 2026-06-20

Status: research evidence for north-star roadmap Phase 1. Validation-only; no held-out test data was read and no ledger was written. Not claim evidence.

## Protocol

- Data: medium-v1 (`b2://pdebench/medium-v1`), tasks advection1d/burgers1d/darcy2d, train budgets 128/256/512/1024, 128 validation samples, test split never fetched.
- Eval: 16-step decoded rollout on validation, pure model prediction, no roll-shift or context estimators.
- Fixed model: tier_b capacity from the P1.2 capacity sweep, latent dim 64, 64 latent tokens, operator hidden dim 128, operator depths `[2,2,2]`, decoder hidden dim 128.
- Training: stages operator(12) / decoder(6) / operator_decoded(6) / joint_codec_operator(4), patience 3, AdamW lr 1e-3, batch size 4.
- Hardware: single RTX 4090 Vast.ai instance `41875102`, launched from PR #84 branch `codex/p1-data-budget-sweep`.
- Published artifact: `b2://pdebench/remote-runs/data-budget-sweep/data_budget_sweep_medium-v1_20260621T020525Z.tar.gz`.
- Verified local tarball SHA256: `aebcdbf4993170b46aaa67a5421d26555fda7d979077ba00a90c6887d8ffe305`.
- Metrics JSON committed at `docs/research/artifacts/p1_data_budget_sweep_medium_v1_val.json`.

## Results

Validation `decoded_rollout_nrmse`; lower is better.

| run | train samples | overall | h16 | delta vs tier_b capacity | delta vs persistence |
|---|---:|---:|---:|---:|---:|
| persistence_medium_v1_val | - | 0.3826003490 | 0.3710499334 | - | - |
| tier_b capacity reference | 512 | 0.7449043874 | 0.7722999334 | - | +0.3623040383 |
| ups_medium_data_budget_n128 | 128 | 0.7905242943 | 0.7866944203 | +0.0456199069 | +0.4079239453 |
| ups_medium_data_budget_n256 | 256 | 0.8741901416 | 0.9676473012 | +0.1292857542 | +0.4915897926 |
| ups_medium_data_budget_n512 | 512 | 0.7992866927 | 0.8093503543 | +0.0543823053 | +0.4166863436 |
| ups_medium_data_budget_n1024 | 1024 | 0.8170613903 | 0.8385859520 | +0.0721570029 | +0.4344610413 |

## Findings

1. **The data-budget scale axis did not help.** The best data-budget run was `n128`, not the larger budgets. It scored `0.7905242942784613`, worse than tier_b capacity `0.7449043873888164` by `0.0456199068896449` absolute.
2. **Persistence remains far ahead.** The best run was worse than persistence `0.38260034902058476` by `0.4079239452578765` absolute, so the validation gate remains missed by a large margin.
3. **The curve is non-monotonic and unfavorable.** `n256` regressed hardest, `n512` roughly returned near `n128`, and `n1024` remained worse than `n128`; raw train-sample count is not the binding bottleneck for this architecture/recipe.
4. **The protocol stayed clean.** Remote logs showed only train/val hydration for burgers1d, advection1d, and darcy2d. All four budget summaries were present, the sweep summary reports `held_out_test_data_read=false`, and B2 download verification matched the remote tarball SHA.
5. **A runner bug was found after training, not in training.** The remote runner invoked `python scripts/summarize_data_budget_sweep.py` without `PYTHONPATH=.`, causing `ModuleNotFoundError: No module named 'scripts'` after all four budget runs completed. The summary and artifact were salvaged from completed outputs with `PYTHONPATH=.`, and the runner is patched to prevent recurrence.

## Decision

- Do not promote any data-budget run and do not spend held-out test budget.
- Stop the fixed-tier_b in-house scale-axis line: capacity, rollout-stability recipes, and data budget have all failed to beat persistence on medium-v1 validation.
- Move the current in-house core to explore-track status for this north-star path unless a new architecture changes the bottleneck.
- Next best path: prioritize the backbone-transplant or physics-primitive route from the roadmap, with the same validation-only discipline before any held-out test use.

# Causal Transport Phase Estimator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the 2026-06-04 literature landscape into a concrete UPS execution path and implement the first validation-only P1 milestone: a default-off causal data-conditioned transport phase estimator.

**Architecture:** Keep the frozen `light-v1` claim protocol and existing UPS encoder/operator/decoder stack. Add a train-fitted field-feature shift estimator that predicts per-sample periodic shifts from allowed causal field features, consumes the locked coefficients during decoded validation, and records phase-gate evidence before any held-out access. Broader foundation work remains sequenced behind this cheap validation gate.

**Tech Stack:** Python, PyTorch, NumPy, HDF5 PDEBench shards, `scripts/run_light_experiment.py`, `src/ups/eval/pdebench_runner.py`, JSON evidence manifests, pytest, ruff, black.

---

## North Star And Decision Gates

Build the strongest defensible universal-simulation claim in this repository without broadening beyond the measured protocol.

- Frozen protocol: `light-v1`, tasks `advection1d`, `burgers1d`, `darcy2d`, validation/test separation, 32-sample caps where recorded, 16-step decoded rollout, `decoded_rollout_nrmse`, artifact hashes, and held-out ledger discipline.
- Current blocker: no-context model-side advection h16/phase tracking.
- Existing phase gate: overall `decoded_rollout_nrmse < 0.35078329353213156`, advection rollout `< 0.4866576789288726`, and advection h16 `<= 0.44444171136384397`.
- Held-out policy: no new held-out test or pre-test contract unless the validation-only phase gate clears first.

## Comprehensive Goal Ladder

### G1: P1 Data-Conditioned Causal Transport Phase Estimator

Objective: train on `train` only, predict per-sample fractional shift from causal field features, and apply that locked estimator during `val` decoded evaluation.

Success:

- The estimator is default-off and does not alter existing metrics unless configured.
- The fitting script records train and validation predicted-shift distributions.
- Validation candidate is evaluated against the phase gate with no held-out access.
- Positive or negative evidence is packaged and validated.

### G2: P2 Learned Warp Sidecar

Objective: if P1 fails, move from scalar shift to a small learned scalar/per-position warp.

Success:

- Periodic interpolation is unit-tested for identity, integer, and fractional displacements.
- Module is task-scoped/default-off.
- Validation reports shift/warp distributions and h16 movement.

### G3: P3 PDE-Refiner-Style Decoded Corrector

Objective: add a lightweight decoded refiner for long rollout and spectral/high-frequency errors.

Success:

- Refiner is trained only on train split.
- Validation improves h16/spectral metrics without Burgers/Darcy regressions.
- Runtime/cost overhead is recorded.

### G4: P4 Hybrid Local/Global Branch

Objective: test a local stencil branch plus global latent/operator branch with task/family gating.

Success:

- Transport/Burgers improvements do not degrade Darcy materially.
- Gate outputs are logged by task/family/horizon.

### G5: P5 Semigroup / Continuous-Time Objective

Objective: add Poseidon/CFO-inspired time-consistency training after cheap phase/warp/refiner evidence.

Success:

- Direct and composed rollouts are consistent on decoded fields.
- Random temporal skip training improves h16 without h1 regression.

### G6: P6 Foundation Backbone Track

Objective: only after light-v1 evidence improves, pursue UPT/Poseidon/MORPH/DPOT-style backbone work.

Success:

- Separate research branch with transfer/scale gates.
- No claim-language changes without protocol-mapped measurements.

## File Structure

- Create `docs/research/2026-06-04-universal-simulator-literature-and-ecosystem-landscape.md`: committed copy of the June 4 landscape snapshot so this worktree carries the latest research context.
- Create `docs/superpowers/plans/2026-06-07-causal-transport-phase-estimator-plan.md`: this execution plan.
- Modify `src/ups/eval/pdebench_runner.py`: add `evaluation.decoded_data_conditioned_roll_shift_estimator`, field-feature extraction, coefficient-based shift prediction, metric stats, and report metadata.
- Create `scripts/run_data_conditioned_transport_shift_gate.py`: train-only fitting/gating script for field-feature shift prediction.
- Modify `tests/unit/test_pdebench_runner_eval.py`: decoded evaluator tests for configured data-conditioned shift application and default-off behavior.
- Create `tests/unit/test_run_data_conditioned_transport_shift_gate.py`: synthetic HDF5 tests for fitting, validation scoring, and held-out exclusion.
- Create `docs/claim_evidence/ups_advection_data_conditioned_phase_candidate_val_evidence.json`: validation evidence after the candidate run.
- Create `scripts/validate_ups_advection_data_conditioned_phase_candidate_evidence.py`: evidence validator.
- Create `tests/unit/test_validate_ups_advection_data_conditioned_phase_candidate_evidence.py`: validator regression tests.
- Modify `docs/claim_evidence/universal_sota_roadmap.md`: append the P1 candidate result and next decision.

## Task 1: Commit The Latest Research Context And Plan

**Files:**
- Create: `docs/research/2026-06-04-universal-simulator-literature-and-ecosystem-landscape.md`
- Create: `docs/superpowers/plans/2026-06-07-causal-transport-phase-estimator-plan.md`

- [x] **Step 1: Add the June 4 literature landscape snapshot**

Copy the exact research brief from `/Users/emerygunselman/.codex/worktrees/50bf/universal_simulator/docs/research/2026-06-04-universal-simulator-literature-and-ecosystem-landscape.md` into this worktree under `docs/research/`.

- [x] **Step 2: Verify the research file exists**

Run: `test -s docs/research/2026-06-04-universal-simulator-literature-and-ecosystem-landscape.md`

Expected: exit code `0`.

- [x] **Step 3: Verify the plan exists**

Run: `test -s docs/superpowers/plans/2026-06-07-causal-transport-phase-estimator-plan.md`

Expected: exit code `0`.

## Task 2: Add Default-Off Decoded Data-Conditioned Shift Estimator

**Files:**
- Modify: `src/ups/eval/pdebench_runner.py`
- Modify: `tests/unit/test_pdebench_runner_eval.py`

- [x] **Step 1: Add failing evaluator tests**

Add tests that configure `evaluation.decoded_data_conditioned_roll_shift_estimator` with a constant bias coefficient and assert it can roll persistence for `advection1d` while remaining absent by default.

- [x] **Step 2: Run the focused tests and verify failure**

Run: `python -m pytest tests/unit/test_pdebench_runner_eval.py -q`

Expected before implementation: tests fail because the config key is not implemented.

- [x] **Step 3: Implement estimator config and field-feature helpers**

Add helpers in `pdebench_runner.py`:

- `_data_conditioned_roll_shift_estimator_config(raw)`
- `_data_conditioned_shift_features(field, horizon, rollout_steps)`
- `_estimate_data_conditioned_roll_shift(field, cfg, horizon, rollout_steps)`

Required feature names: `bias`, `horizon_norm`, `mean`, `std`, `rms`, `abs_mean`, `max`, `min`.

- [x] **Step 4: Integrate estimator into decoded rollout loop**

When configured and applicable, predict `roll_shift`, append metrics under `decoded_data_conditioned_roll_shift_*`, and support `mode: roll_persistence` by rolling the persistence field.

- [x] **Step 5: Run focused tests**

Run: `python -m pytest tests/unit/test_pdebench_runner_eval.py -q`

Expected: pass.

## Task 3: Add Train-Only Data-Conditioned Shift Gate Script

**Files:**
- Create: `scripts/run_data_conditioned_transport_shift_gate.py`
- Create: `tests/unit/test_run_data_conditioned_transport_shift_gate.py`

- [x] **Step 1: Add synthetic gate tests**

Create tests that write train/val HDF5 files, fit coefficients on train, validate on val, and assert no test split is read when `--test-split` is omitted.

- [x] **Step 2: Implement HDF5 loading and feature extraction**

The script must load `data`, normalize shapes to `(samples, steps, width)`, compute candidate best shifts on train transitions, and compute feature rows from the previous field plus horizon.

- [x] **Step 3: Implement ridge fitting and validation scoring**

Fit linear coefficients by least squares with small L2 regularization. Score validation by applying predicted shifts and computing NRMSE.

- [x] **Step 4: Export evaluator config**

The JSON output must include:

```json
{
  "selected_override": {
    "evaluation.decoded_data_conditioned_roll_shift_estimator": {
      "mode": "roll_persistence",
      "tasks": ["advection1d"],
      "min_horizon": 1,
      "feature_names": ["bias", "horizon_norm", "mean", "std", "rms", "abs_mean", "max", "min"],
      "coefficients": {"bias": 0.0}
    }
  }
}
```

- [x] **Step 5: Run focused script tests**

Run: `python -m pytest tests/unit/test_run_data_conditioned_transport_shift_gate.py -q`

Expected: pass.

## Task 4: Run Validation-Only Candidate And Package Evidence

**Files:**
- Create: `docs/claim_evidence/ups_advection_data_conditioned_phase_candidate_val_evidence.json`
- Create: `docs/claim_evidence/artifacts/ups_advection_data_conditioned_phase_candidate_val.tar.gz`
- Create: `scripts/validate_ups_advection_data_conditioned_phase_candidate_evidence.py`
- Create: `tests/unit/test_validate_ups_advection_data_conditioned_phase_candidate_evidence.py`
- Modify: `docs/claim_evidence/universal_sota_roadmap.md`

- [x] **Step 1: Fit train-only estimator**

Run `scripts/run_data_conditioned_transport_shift_gate.py` on `advection1d_train.h5` and `advection1d_val.h5`, 32-sample cap, 16-step rollout, no held-out test split.

- [x] **Step 2: Run decoded validation-only UPS candidate**

Run `scripts/run_light_experiment.py` with `--skip-training`, the existing checkpoint source, decoded 16-step evaluation, residual alpha `transport: 0.21`, and the fitted data-conditioned estimator override.

- [x] **Step 3: Evaluate against phase gate**

Use `scripts.validate_ups_advection_phase_tracking_gate_contract.evaluate_candidate_summary` on the validation summary.

- [x] **Step 4: Package evidence**

Tar only `summary.json`, `resolved_train.yaml`, `resolved_eval.yaml`, and the train-fit JSON. Record SHA256 and byte sizes.

- [x] **Step 5: Add evidence validator and tests**

Validator must prove held-out flags are false, artifact hashes match, metrics match the artifact summary, train-fit metadata matches the tar member, and phase-gate errors recompute.

## Task 5: Verification, PR, CI, Merge

**Files:**
- All changed files.

- [x] **Step 1: Run targeted checks**

Run:

```bash
python scripts/validate_ups_advection_data_conditioned_phase_candidate_evidence.py
python -m pytest tests/unit/test_pdebench_runner_eval.py tests/unit/test_run_data_conditioned_transport_shift_gate.py tests/unit/test_validate_ups_advection_data_conditioned_phase_candidate_evidence.py -q
python -m black --check src/ups/eval/pdebench_runner.py scripts/run_data_conditioned_transport_shift_gate.py scripts/validate_ups_advection_data_conditioned_phase_candidate_evidence.py tests/unit/test_pdebench_runner_eval.py tests/unit/test_run_data_conditioned_transport_shift_gate.py tests/unit/test_validate_ups_advection_data_conditioned_phase_candidate_evidence.py
python -m ruff check src/ups/eval/pdebench_runner.py scripts/run_data_conditioned_transport_shift_gate.py scripts/validate_ups_advection_data_conditioned_phase_candidate_evidence.py tests/unit/test_pdebench_runner_eval.py tests/unit/test_run_data_conditioned_transport_shift_gate.py tests/unit/test_validate_ups_advection_data_conditioned_phase_candidate_evidence.py
git diff --check
```

- [x] **Step 2: Run full tests**

Run: `python -m pytest -q`

Expected: pass.

- [ ] **Step 3: Commit, push, PR, CI, merge**

Commit the scoped change, push the branch, open a draft PR, wait for CI, fix failures, mark ready, and merge if CI passes.

## Stop Conditions

- Stop before held-out access unless the validation phase gate clears.
- Stop and record negative evidence if the estimator worsens h16 or causes Burgers/Darcy regression.
- Stop before foundation-backbone refactors until P1/P2/P3 have validated or failed under the claim protocol.

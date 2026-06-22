# Public Results Benchmarks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a zero-GPU public results packet with figures, benchmark tables, and external benchmark readiness notes generated from committed evidence.

**Architecture:** Add one deterministic generator that reads committed JSON evidence, normalizes benchmark rows, writes `docs/results/` tables, and renders a small set of README-grade PNG figures. Keep generated numbers source-of-truth driven and label external results as measured under the repo's `light-v1` protocol, not as published-paper leaderboard claims.

**Tech Stack:** Python 3.10, standard-library JSON/CSV/pathlib, matplotlib for PNG output, existing `docs/claim_evidence/` manifests, pytest unit tests, Markdown public docs.

---

### Task 1: Add the Public Asset Generator

**Files:**
- Create: `scripts/build_public_assets.py`
- Test: `tests/unit/test_build_public_assets.py`

- [ ] **Step 1: Write fixture tests for benchmark-row extraction**

Create `tests/unit/test_build_public_assets.py` with fixture JSON objects covering a primary UPS row, persistence row, local strong baseline row, one scoped UPS variant, and one measured external FNO row. Test that `build_benchmark_rows` returns sorted rows with `label`, `category`, `metric_value`, `improvement_vs_primary`, `claim_comparable`, and `published_numbers_directly_comparable`.

- [ ] **Step 2: Write fixture tests for per-task extraction**

In the same test file, create a fixture with `task_advection1d_decoded_rollout_nrmse`, `task_burgers1d_decoded_rollout_nrmse`, and `task_darcy2d_decoded_rollout_nrmse`. Test that `build_task_rows` emits one row per task and preserves the task labels.

- [ ] **Step 3: Implement the generator functions**

Implement `load_json`, `write_json`, `write_tsv`, `build_benchmark_rows`, `build_task_rows`, and `build_external_matrix_rows` in `scripts/build_public_assets.py`. Inputs must default to:

```text
docs/claim_evidence/universal_sota_claim_evidence.json
docs/claim_evidence/external_baseline_mapping.json
docs/claim_evidence/artifacts/light_v1_demo_scorecard.json
```

The output directory must default to `docs/results/generated`.

- [ ] **Step 4: Implement PNG rendering**

Add matplotlib-backed renderers:

```text
claim_scorecard.png
per_task_breakdown.png
external_benchmarks.png
```

Use a restrained white-background style with clear labels and "lower is better" in the y-axis label.

- [ ] **Step 5: Implement CLI entrypoint**

Add an argparse CLI so `python scripts/build_public_assets.py` creates all tables and figures in one command. The command must print every output path it writes.

- [ ] **Step 6: Run targeted tests**

Run:

```bash
python -m pytest tests/unit/test_build_public_assets.py -q
```

Expected: all new tests pass.

### Task 2: Add Public Results Docs

**Files:**
- Create: `docs/results/README.md`
- Create: `docs/results/external_benchmarks.md`
- Modify: `README.md`
- Modify: `docs/public/README.md`

- [ ] **Step 1: Generate public assets**

Run:

```bash
python scripts/build_public_assets.py
```

Expected generated files:

```text
docs/results/generated/benchmark_summary.json
docs/results/generated/benchmark_summary.tsv
docs/results/generated/per_task_summary.tsv
docs/results/generated/external_benchmark_matrix.tsv
docs/results/generated/claim_scorecard.png
docs/results/generated/per_task_breakdown.png
docs/results/generated/external_benchmarks.png
```

- [ ] **Step 2: Write `docs/results/README.md`**

Document what the figures show, how to regenerate them, and the claim boundary:

```bash
python scripts/build_public_assets.py
```

State that `light-v1` matched-protocol numbers are claim-comparable within this repo, while published-paper table values are not directly comparable unless rerun or mapped.

- [ ] **Step 3: Write `docs/results/external_benchmarks.md`**

Create a matrix for PDEBench, NeuralOperator FNO/UNO, PDEBench U-Net, CNO/Poseidon, PDEArena, PhysicsNeMo, and RealPDEBench with columns:

```text
Surface | Status | What it proves | Next step | Claim boundary
```

- [ ] **Step 4: Link generated results from public docs**

Add a short results section to `README.md` and `docs/public/README.md` linking to `docs/results/README.md`, the generated scorecard figure, and the external benchmark matrix.

### Task 3: Verify and Commit

**Files:**
- Modify as produced by Tasks 1-2.

- [ ] **Step 1: Run focused validation**

Run:

```bash
python -m pytest tests/unit/test_build_public_assets.py tests/unit/test_demo_scorecard.py tests/unit/test_validate_external_baseline_mapping.py -q
python scripts/build_public_assets.py
python scripts/validate_external_baseline_mapping.py
git diff --check
```

Expected: all commands exit with status 0.

- [ ] **Step 2: Inspect generated image files**

Verify that `docs/results/generated/*.png` exist and are non-empty.

- [ ] **Step 3: Commit the public results packet**

Run:

```bash
git add README.md docs/public/README.md docs/results scripts/build_public_assets.py tests/unit/test_build_public_assets.py <this plan file>
git commit -m "Add public benchmark assets"
```

Expected: a single commit on `codex/public-results-benchmarks`.

### Task 4: Extend With Secondary Metrics And Repeatability

**Files:**
- Modify: `scripts/build_public_assets.py`
- Modify: `tests/unit/test_build_public_assets.py`
- Create: `docs/results/metrics_beyond_nrmse.md`
- Create: `docs/results/research_diagnostics.md`
- Modify generated assets under `docs/results/generated/`
- Modify public docs that link to public asset reproducibility.

- [x] **Step 1: Add artifact-backed metric-suite extraction**

Read the primary UPS artifact tarball referenced by the claim evidence when it
is available, merge its `summary_test.json` metrics into the public results
rows, and compare them against the durable persistence scorecard.

- [x] **Step 2: Add secondary metric and horizon figures**

Generate:

```text
docs/results/generated/metric_suite_summary.tsv
docs/results/generated/horizon_summary.tsv
docs/results/generated/primary_metric_suite.png
docs/results/generated/horizon_profile.png
```

Keep decoded rollout NRMSE as the primary claim metric. Treat MAE, MSE,
spectral energy error, step-1 NRMSE, H4 NRMSE, and H16 NRMSE as diagnostics.

- [x] **Step 3: Make public asset generation repeatable**

Add:

```text
docs/results/generated/asset_manifest.json
python scripts/build_public_assets.py --check
```

The check command regenerates all public assets into a temporary directory
and compares them byte-for-byte with committed outputs.

- [x] **Step 4: Document secondary metrics and future gates**

Explain why NRMSE remains primary, what the secondary metrics currently show,
and which future metrics need evaluator support before they can become public
evidence gates.

- [x] **Step 5: Add repeatable research diagnostics**

Generate validation-only transport ablation and train/validation transfer
figures from tracked evidence:

```text
docs/results/generated/transport_ablation_summary.tsv
docs/results/generated/transfer_validation_summary.tsv
docs/results/generated/transport_ablation.png
docs/results/generated/transfer_validation.png
```

Document them as roadmap diagnostics, not held-out claim-comparable benchmark
results.

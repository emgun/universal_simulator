# Universal SOTA Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the current UPS light-v1 result into progressively stronger, evidence-backed universal SOTA claims without overclaiming beyond the measured protocol.

**Architecture:** Treat `docs/claim_evidence/` as the source of truth for claim surfaces, provenance, hashes, validation gates, and held-out-test decisions. Work proceeds in small PRs that add one evidence surface at a time, update this append-only roadmap, and run targeted validation before any broader claim language changes.

**Tech Stack:** Python, PyTorch, HDF5 PDEBench-shaped data, pytest, JSON evidence manifests, Markdown evidence notes, optional external baselines through NeuralOperator, PDEBench, ConvolutionalNeuralOperator, Poseidon ScOT, and Hugging Face Hub.

---

## North Star

Build the strongest defensible universal simulation claim in this repository by improving and comparing `decoded_rollout_nrmse` under a frozen, auditable protocol.

The current claim protocol is `light-v1`: the PDEBench-shaped multitask set, train/validation/test split boundaries, 32-sample caps where recorded, 16-step decoded rollout horizon, `decoded_rollout_nrmse`, per-task/family breakdowns, artifact bundles, and held-out ledger discipline.

Published paper-table values are not interchangeable with this protocol unless the split, sample budget, rollout horizon, metric, and task bundle are mapped or remeasured. A fair claim-protocol baseline answers whether another model wins when measured under this repository's contract. An external-paper reproduction is a stronger but separate artifact: official or faithful outside code plus explicit mapping back to the claim protocol.

## Current Evidence Snapshot

Claim candidate:

- UPS `ups_light_shared_context_transport_guarded` held-out test `decoded_rollout_nrmse = 0.4165820594268877`.
- UPS held-out per-task test NRMSE: advection `0.5765863333379032`, Burgers `0.17446857017795178`, Darcy `0.20909553062258152`.
- UPS validation metric recorded with the held-out measurement: `decoded_rollout_nrmse = 0.2723239543019452`.

Measured fair and external baselines under the claim protocol:

- Repo-local physical Fourier neural baseline held-out test: `0.5636730976415197`.
- NeuralOperator FNO validation/test: `0.46753981278379725` / `0.6391747076887233`.
- NeuralOperator UNO validation/test: `0.363424243629033` / `0.5560551396226746`.
- PDEBench U-Net validation/test: `0.5394672411385386` / `0.6095843876848097`.
- Official simplified CNO1d validation/test: `0.40815509179677445` / `0.5918753212407414`.

Foundation-transfer track:

- `foundation_transfer_readiness_light_v1` recorded train/validation-only blockers for Poseidon and CNO-FM transfer.
- `poseidon_transfer_adapter_manifest_light_v1` recorded square-pixel adapter round-trip distortion `adapter_roundtrip_nrmse = 0.0023447850529950184`.
- `poseidon_scot_val_light_v1` loaded `camlab-ethz/Poseidon-T`, verified `model.safetensors` SHA256 `e97428c93a16cbb52a41bc4794eb71be3aed436fb9cc547d9eeebb20f3940fb2`, and measured validation `decoded_rollout_nrmse = 0.9999999950370435`.
- The Poseidon zero-shot result is real evidence, but not claim-comparable and not useful as a transfer result because scalar light-v1 replaces the official 4-channel input embedding and recovery layers.

Historical isolated validation numbers, including old very low per-task values, must not be used in claim language unless the exact command, split, artifact, provenance, and mapping to the current claim protocol are recovered and recorded here.

## Hard Constraints

- Do not run or record a new held-out test measurement unless the relevant validation gate below is cleared first.
- Any held-out test runner must fail closed unless an explicit `--allow-held-out-test-eval` style flag is present and a ledger key prevents accidental duplicate test-budget use.
- Every claim-supporting result needs a command, config, split, metric, task list, source provenance, dependency/version details, artifact hash, and reason if it is not published-table comparable.
- Keep validation and test evidence separate. Validation can guide architecture choices; held-out test can only confirm a validation-selected candidate.
- Do not broaden from "measured under the claim protocol" to "published SOTA" unless the published benchmark protocol has been directly reproduced or defensibly mapped.

## Decision Gates

Gate 1: Poseidon scalar adapter finetune on train/validation only.

- Clear for held-out consideration only if validation `decoded_rollout_nrmse <= 0.363424243629033`, the current best measured external validation baseline, with no single task collapsing near `1.0`.
- Continue train/validation-only Poseidon work if validation is between `0.363424243629033` and `0.5`, because it may justify controlled unfreezing or low-rank adaptation.
- Stop the current scalar-transfer path if validation remains above `0.5` after a clean frozen-adapter finetune, because it is not close enough to spend held-out budget.

Gate 2: Controlled Poseidon unfreeze or low-rank adaptation.

- Enter only after Gate 1 proves the adapter can learn nontrivial light-v1 dynamics.
- Keep the same train/validation-only split discipline and evidence schema.
- Compare against the Gate 1 validation artifact, not against held-out test.

Gate 3: Held-out Poseidon transfer measurement.

- Enter only if Gate 1 or Gate 2 clears the validation threshold and all provenance/hashes are recorded.
- Use a ledger key specific to model family, checkpoint, train split, validation-selected hyperparameters, test split, rollout steps, task list, and adapter mode.
- Update `docs/claim_evidence/external_baseline_mapping.json` and `docs/claim_evidence/external_baseline_mapping.md` in the same PR as the evidence artifact.

Gate 4: UPS-side improvement track.

- Pursue only when a candidate can be selected on validation without touching held-out test.
- Highest-signal weakness is the current UPS held-out advection error relative to Burgers and Darcy.
- Any old advection or Burgers validation result must be revalidated through the current harness before it influences the claim.

Gate 5: Broader foundation-model alternatives.

- Keep CNO-FM and other channel-rich or 2D foundation paths separate from scalar light-v1 Poseidon.
- Prefer train/validation adapter-readiness evidence before model-score evidence.
- Do not collapse CNO2d, CNO-FM, Poseidon, and scalar light-v1 into one comparison without an explicit adapter and metric contract.

## File Responsibilities

- `docs/claim_evidence/universal_sota_roadmap.md`: append-only roadmap, worklog, gates, and current active plan.
- `docs/claim_evidence/external_baseline_mapping.md`: public explanation of what each measured baseline does and does not claim.
- `docs/claim_evidence/external_baseline_mapping.json`: machine-readable baseline/evidence registry.
- `docs/claim_evidence/*_evidence.json`: immutable evidence manifests for committed results.
- `docs/claim_evidence/artifacts/*.tar.gz`: compact evidence bundles with hashes recorded in manifests.
- `reports/research/sota_loop/external_baselines/test_ledger.json`: held-out external-baseline test budget guard.
- `scripts/run_external_poseidon_scot_validation.py`: current zero-shot Poseidon validation runner.
- `scripts/run_external_poseidon_transfer_adapter.py`: current Poseidon square-pixel adapter manifest runner.
- `tests/unit/test_external_poseidon_scot_validation.py`: guard tests for the zero-shot Poseidon validation runner.
- `tests/unit/test_external_poseidon_transfer_adapter.py`: guard tests for the Poseidon adapter manifest runner.

## Active Implementation Plan: Poseidon Frozen Scalar Adapter Finetune

### Task 1: Add the finetune runner contract

**Files:**

- Create: `scripts/run_external_poseidon_scot_finetune.py`
- Create: `tests/unit/test_external_poseidon_scot_finetune.py`

- [x] **Step 1: Write a failing unit test for held-out split blocking**

```bash
python -m pytest tests/unit/test_external_poseidon_scot_finetune.py::test_poseidon_finetune_cli_blocks_test_split_before_loading_data -q
```

Expected before implementation: pytest cannot import `scripts.run_external_poseidon_scot_finetune`.

- [x] **Step 2: Write a failing unit test for trainable parameter selection**

```bash
python -m pytest tests/unit/test_external_poseidon_scot_finetune.py::test_configure_trainable_poseidon_parameters_keeps_backbone_frozen -q
```

Expected before implementation: pytest cannot import `configure_trainable_poseidon_parameters`.

- [x] **Step 3: Implement `configure_trainable_poseidon_parameters`**

The function must freeze every parameter by default and unfreeze only scalar input/output adapter parameters for the first gate. The test should use a small fake module with names that mirror Poseidon embedding/recovery layers so it does not require downloading the checkpoint.

- [x] **Step 4: Implement CLI split guards**

The CLI must reject `--eval-split test` unless `--allow-held-out-test-eval` is supplied. It must perform this check before any data load, model load, or Hugging Face checkpoint resolution.

- [x] **Step 5: Run the targeted tests**

```bash
python -m pytest tests/unit/test_external_poseidon_scot_finetune.py -q
```

Expected after implementation: all tests pass.

### Task 2: Add train/validation-only finetuning

**Files:**

- Modify: `scripts/run_external_poseidon_scot_finetune.py`
- Modify: `tests/unit/test_external_poseidon_scot_finetune.py`

- [x] **Step 1: Reuse the existing adapter and evaluation path**

The runner must reuse `light_step_to_poseidon_pixels`, `poseidon_pixels_to_repo_flat`, `load_poseidon_scot_model`, and `evaluate_poseidon_scot_validation` instead of creating a second adapter implementation.

- [x] **Step 2: Add a small supervised train loop**

The loop trains teacher-forced one-step pairs from `train` using the same task list, sample cap, image size, and time value as validation. The default first gate trains only scalar embedding/recovery parameters.

- [x] **Step 3: Emit a validation evidence summary**

The summary must include `measurement_type = "poseidon_scot_finetune_validation_measurement"`, `held_out_test_used = false`, `claim_comparable = false`, checkpoint SHA256, train split, validation split, adapter mode, trainable parameter count, total parameter count, optimizer settings, duration, task metrics, and family metrics.

- [x] **Step 4: Run a dry or fixture-backed unit test**

```bash
python -m pytest tests/unit/test_external_poseidon_scot_finetune.py -q
```

Expected: the summary validator rejects missing checkpoint hash and rejects summaries that mark held-out test use as true.

### Task 3: Run the validation-only Poseidon finetune gate

**Files:**

- Create: `docs/claim_evidence/poseidon_scot_scalar_ft_val_light_v1_e3_lr1e4_clip1_evidence.json`
- Create: `docs/claim_evidence/artifacts/poseidon_scot_scalar_ft_val_light_v1_e3_lr1e4_clip1.tar.gz`
- Modify: `docs/claim_evidence/external_baseline_mapping.json`
- Modify: `docs/claim_evidence/external_baseline_mapping.md`
- Modify: `docs/claim_evidence/universal_sota_roadmap.md`

- [x] **Step 1: Run the train/validation-only command**

```bash
python scripts/run_external_poseidon_scot_finetune.py --config configs/train_multitask_heterogeneous_light_best.yaml --name poseidon_scot_scalar_ft_val_light_v1_e3_lr1e4_clip1 --output-root reports/research/sota_loop/external_baselines --train-split train --eval-split val --max-train-samples 32 --max-eval-samples 32 --rollout-steps 16 --poseidon-model-size T --checkpoint-file model.safetensors --device cpu --time-value 1.0 --data-root data/pdebench --poseidon-repo /tmp/poseidon-official --expected-checkpoint-sha256 e97428c93a16cbb52a41bc4794eb71be3aed436fb9cc547d9eeebb20f3940fb2 --tasks advection1d burgers1d darcy2d --epochs 3 --learning-rate 0.0001 --weight-decay 0.0001 --batch-size 32 --grad-clip-norm 1.0 --adapter-mode scalar_layers
```

Expected: writes a validation summary under `reports/research/sota_loop/external_baselines/poseidon_scot_scalar_ft_val_light_v1_e3_lr1e4_clip1/summary.json` and does not read the held-out test split.

- [x] **Step 2: Package and record the evidence**

```bash
tar -czf docs/claim_evidence/artifacts/poseidon_scot_scalar_ft_val_light_v1_e3_lr1e4_clip1.tar.gz -C reports/research/sota_loop/external_baselines poseidon_scot_scalar_ft_val_light_v1_e3_lr1e4_clip1/summary.json
```

Expected: the artifact is small enough to commit, and `shasum -a 256 docs/claim_evidence/artifacts/poseidon_scot_scalar_ft_val_light_v1_e3_lr1e4_clip1.tar.gz` produces the hash recorded in the evidence manifest. Update the mapping and evidence files manually with exact command, metric, artifact path, SHA256, and non-comparability reason.

- [x] **Step 3: Apply the decision gate**

If validation `decoded_rollout_nrmse <= 0.363424243629033`, mark Gate 1 cleared for held-out consideration. If it is between `0.363424243629033` and `0.5`, mark Gate 1 as partial and plan controlled unfreeze. If it is above `0.5`, mark the scalar-transfer path stopped.

- [x] **Step 4: Run repository checks**

```bash
python -m pytest tests/unit/test_external_poseidon_scot_finetune.py tests/unit/test_external_poseidon_scot_validation.py tests/unit/test_external_poseidon_transfer_adapter.py -q
```

Expected: all targeted Poseidon tests pass.

```bash
python scripts/validate_external_baseline_mapping.py
```

Expected: baseline mapping validation passes.

## Append-Only Worklog

### 2026-05-31

Status:

- Created this roadmap to make the north-star path explicit and appendable.
- Added `scripts/run_external_poseidon_scot_finetune.py` and `tests/unit/test_external_poseidon_scot_finetune.py`.
- Verified the finetune runner contract with unit tests, existing Poseidon tests, baseline-mapping validation, and a tiny real-checkpoint train/validation smoke.
- Ran the original full Gate 1 scalar-layer command at `learning_rate = 0.001`; it failed during validation with `RuntimeError: Non-finite Poseidon prediction for task=advection1d`.
- Added finite-prediction/loss/gradient/parameter guards plus `--grad-clip-norm` so unstable runs fail at the training boundary with a clear diagnostic.
- Ran the safer full Gate 1 scalar-layer command as `poseidon_scot_scalar_ft_val_light_v1_e3_lr1e4_clip1`: 32 train samples, 32 validation samples, all three tasks, 16 teacher-forced steps, 3 epochs, `learning_rate = 0.0001`, `grad_clip_norm = 1.0`.
- Packaged evidence at `docs/claim_evidence/poseidon_scot_scalar_ft_val_light_v1_e3_lr1e4_clip1_evidence.json` and artifact SHA256 `c0aef6f4c42a2e499a37df23341c462866fb7d4a86f4133375b790ddebfae17d`.
- Current best UPS claim remains held-out test `decoded_rollout_nrmse = 0.4165820594268877`.
- Current best external validation baseline is NeuralOperator UNO at `0.363424243629033`.
- Current best measured external held-out test baseline is NeuralOperator UNO at `0.5560551396226746`.
- Poseidon zero-shot validation is `0.9999999950370435`, which is not competitive and proves that direct scalar transfer is not enough.
- Poseidon scalar-layer finetune validation is `0.5453508470039229`; per-task validation NRMSE is advection `0.6030753349043854`, Burgers `0.49033314173084885`, Darcy `0.47892385326272763`.

Decision:

- The Poseidon frozen scalar adapter finetune path is now measured and stopped on validation.
- No Poseidon held-out test should run from the scalar-only adapter path.
- UPS-side improvements should focus on validation-selected advection robustness unless a controlled Poseidon unfreeze/LoRA gate is intentionally opened on train/validation only.
- The tiny smoke result, validation `decoded_rollout_nrmse = 0.9988580194089105`, is only a runner check. It does not count as claim evidence because it used one advection sample, one rollout step, and one epoch.
- The full scalar-layer finetune result improves over zero-shot but is above the `0.5` stop threshold and above the `0.363424243629033` held-out consideration threshold.
- Decision: stop scalar-only Poseidon transfer and do not spend held-out Poseidon test budget from this path.

Next checkpoint:

- Finish repository checks for the committed evidence.
- Next technical path: either controlled Poseidon unfreeze/LoRA on train/validation only, or pivot to UPS-side advection robustness because advection is still the weakest claim task.

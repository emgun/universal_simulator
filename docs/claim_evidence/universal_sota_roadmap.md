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

## Active Implementation Plan: UPS Advection Context-Delay Validation Gate

### Task 1: Revalidate the current UPS advection context path

**Files:**

- Create: `docs/claim_evidence/ups_advection_context_delay_val_gate_evidence.json`
- Create: `docs/claim_evidence/artifacts/ups_advection_context_delay_val_gate.tar.gz`
- Modify: `docs/claim_evidence/universal_sota_roadmap.md`

- [x] **Step 1: Reproduce the current claim-style validation baseline**

```bash
python scripts/run_light_experiment.py --config configs/train_multitask_heterogeneous_light_best.yaml --name ups_light_advection_context_current_val --output-root reports/research/sota_loop/advection_robustness_gate --skip-training --checkpoint-source reports/research/sota_loop/learned_capacity_gate/ups_light_local_joint_rollout4_residual_ft_val --decoded --decoded-rollout-steps 16 --device cpu --override data.root=data/pdebench --override data.split=val --override data.max_samples=32 --override 'operator.conditioning.sources={"task_id":3,"equation_signature":15}' --eval-override evaluation.decoded_persistence_residual_alpha=0.0 --eval-override 'evaluation.decoded_context_roll_shift_estimator={candidate_shifts: [-4, -2, -1, 0, 1, 2, 4, 8, 16, 24, 32, 40, 48, 64], context_transitions: 8, coefficients: {slope: 0.9974352988185539, intercept: 0.0}, families: [transport, conservation], mode: roll_persistence, calibration_scope: shared_1d_transport}'
```

Expected: validation-only output matching current claim validation `decoded_rollout_nrmse = 0.2723239543019452`; no `--extra-eval-split test` and no held-out ledger use.

- [x] **Step 2: Run transport-only context-delay ablations**

Run `context_transitions = 1`, `2`, `4`, and `8`, with `families: [transport]`, `mode: roll_persistence`, `decoded_persistence_residual_alpha = 0.0`, and the same checkpoint, task bundle, sample cap, split, and rollout horizon.

Expected: select only on validation and compare against the reproduced current validation baseline.

- [x] **Step 3: Package validation evidence**

```bash
tar -czf docs/claim_evidence/artifacts/ups_advection_context_delay_val_gate.tar.gz -C reports/research/sota_loop/advection_robustness_gate ups_light_advection_context_current_val/summary.json ups_light_advection_context_transport_only_ct1_val/summary.json ups_light_advection_context_transport_only_ct2_val/summary.json ups_light_advection_context_transport_only_ct4_val/summary.json ups_light_advection_context_transport_only_ct8_val/summary.json
```

Expected: record artifact SHA256 in `docs/claim_evidence/ups_advection_context_delay_val_gate_evidence.json`.

### Task 2: Decide whether the validation candidate can spend held-out budget

- [x] **Step 1: Protocol review**

Decide whether `context_transitions = 1`, `min_horizon = 2`, and `families: [transport]` are an admissible light-v1 claim variant. This setting uses less context to estimate the shift than the current CT8 claim config, but it applies the roll-persistence correction earlier in the teacher-forced decoded evaluator.

- [x] **Step 2: Wire audit/evidence support before test**

If the protocol review accepts CT1, add or update the claim evidence/audit surface so the selected validation config, summary, artifact SHA256, and intended held-out ledger key are explicit before the held-out command is run.

- [x] **Step 3: Run exactly one ledger-protected held-out confirmation**

Run a held-out test only after Steps 1 and 2. The command must include the explicit held-out ledger guard and must not be repeated under the same key.

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
- Set the next roadmap goal to Gate 4, UPS-side advection robustness, because the scalar Poseidon path was stopped and current UPS held-out advection is the weakest task.
- Reproduced the current validation baseline under the live harness without touching held-out test: `ups_light_advection_context_current_val decoded_rollout_nrmse = 0.2723239543019452`, advection `0.36362945500857824`, Burgers `0.14737692709626082`, Darcy `0.188979512124482`.
- Ran validation-only transport context-delay ablations using the same checkpoint, split, sample cap, task bundle, rollout horizon, and metric:
- CT1 transport-only: overall `0.1419775490176828`, advection `0.12911778915203231`, Burgers `0.14738121412908425`, Darcy `0.188979512124482`.
- CT2 transport-only: overall `0.16694739388393856`, advection `0.18214615629606878`, Burgers `0.14738121412908425`, Darcy `0.188979512124482`.
- CT4 transport-only: overall `0.20808368063656887`, advection `0.2572710005249541`, Burgers `0.14738121412908425`, Darcy `0.188979512124482`.
- CT8 transport-only: overall `0.27230919020248034`, advection `0.36360402666634595`, Burgers `0.14738121412908425`, Darcy `0.188979512124482`.
- Packaged validation evidence at `docs/claim_evidence/ups_advection_context_delay_val_gate_evidence.json` and artifact SHA256 `029495fdaa60fbcb0c341c14980bb1aa40bbbad8bb06615629ad7c8ef7ab07a3`.
- Added `scripts/validate_ups_advection_context_gate_evidence.py` and `tests/unit/test_validate_ups_advection_context_gate_evidence.py`.
- The validator enforces that the CT1 validation gate used no held-out test path, that all commands pin validation split, that artifact bytes match the recorded SHA256, that artifact summary metrics match the manifest, that CT1 is the best validation run, and that CT1/CT2/CT4/CT8 are monotonic as context delay increases.
- Repacked `docs/claim_evidence/artifacts/ups_advection_context_delay_val_gate.tar.gz` with `COPYFILE_DISABLE=1` after the validator caught an AppleDouble `._summary.json` member from the first macOS tarball.
- Added the CT1 pre-test protocol contract at `docs/claim_evidence/ups_advection_ct1_pretest_contract.json`.
- Added `scripts/validate_ups_advection_ct1_pretest_contract.py` and `tests/unit/test_validate_ups_advection_ct1_pretest_contract.py`; the validator reuses the CT1 evidence validator and recomputes the intended held-out measurement key from the exact pre-registered command.
- Pre-registered intended held-out measurement key `11d176a6466fe04af43ccb47645f9a9ae8efdf68520493fbdc34741d86abd716` with ledger path `reports/research/sota_loop/advection_robustness_gate/test_ledger.json`, but did not run the held-out command.
- Ran the pre-registered CT1 held-out command exactly once. Validation remained `decoded_rollout_nrmse = 0.1419775490176828`; held-out test produced `decoded_rollout_nrmse = 0.20177292896682064`, advection `0.22508631227914033`, Burgers `0.17446879896821743`, Darcy `0.20909553062258152`.
- Recorded the held-out ledger entry with measurement key `11d176a6466fe04af43ccb47645f9a9ae8efdf68520493fbdc34741d86abd716` and ledger SHA256 `56cf65f9fe17b0dc03d7a3a5d77070e54c0b4aab7def6554eff7d2b616be45fd`.
- Packaged scoped CT1 held-out evidence at `docs/claim_evidence/ups_advection_ct1_heldout_light_v1_evidence.json` and artifact SHA256 `b3b0809afc58085433ba0bbe1efbfa87deb1c227c18b8e6154b7d343e372834d`.

Decision:

- The Poseidon frozen scalar adapter finetune path is now measured and stopped on validation.
- No Poseidon held-out test should run from the scalar-only adapter path.
- UPS-side improvements should focus on validation-selected advection robustness unless a controlled Poseidon unfreeze/LoRA gate is intentionally opened on train/validation only.
- The tiny smoke result, validation `decoded_rollout_nrmse = 0.9988580194089105`, is only a runner check. It does not count as claim evidence because it used one advection sample, one rollout step, and one epoch.
- The full scalar-layer finetune result improves over zero-shot but is above the `0.5` stop threshold and above the `0.363424243629033` held-out consideration threshold.
- Decision: stop scalar-only Poseidon transfer and do not spend held-out Poseidon test budget from this path.
- The UPS advection context-delay validation gate cleared strongly: CT1 improves validation overall by `0.13034640528426242` absolute and `0.47864465547433244` relative versus the reproduced current validation baseline.
- This is not yet a replacement held-out claim, because CT1 changes the current claim evaluation config from `context_transitions = 8`, `families: [transport, conservation]`, `slope = 0.9974352988185539` to `context_transitions = 1`, `families: [transport]`, `slope = 1.0`.
- The validation evidence is now machine-checkable, but held-out execution is still blocked until the CT1 claim-contract decision and intended ledger key are written before the test command.
- Protocol decision: CT1 is accepted for exactly one ledger-protected held-out confirmation only as a scoped `light-v1 CT1 online transport-context UPS variant`. It must not be described as an autonomous rollout claim, a published-paper SOTA result, or the same exact inference contract as the CT8 claim.
- CT1 held-out confirmation beats the current CT8 held-out claim overall by `0.21480913046006705` absolute / `0.5156466189532753` relative and advection by `0.3515000210587629` absolute / `0.6096225330626585` relative, but it remains a different scoped inference contract.

Next checkpoint:

- Run repository checks for the new evidence and roadmap files.
- Next technical path: protocol-review the CT1 context-delay variant, then either wire it into the claim audit and run one ledger-protected held-out confirmation, or reject it as protocol-shift evidence and open a model-side advection objective instead.
- If CT1 is accepted, add pre-test claim audit wiring that names the validation evidence manifest, selected CT1 command, intended held-out command, and ledger key; only then run the held-out confirmation.
- The pre-test contract is now in place. The next irreversible step is to spend exactly one held-out test measurement using the pre-registered command, then package/update claim evidence only if the held-out result beats the current CT8 held-out claim and is documented with the scoped CT1 language.
- Next technical path: add a scoped CT1 claim row/audit path so downstream status scripts can distinguish the CT1 online-context result from the older CT8 claim and from external-paper SOTA claims.

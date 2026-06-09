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

Scoped UPS variant:

- CT1 online transport-context variant `ups_light_advection_context_transport_only_ct1_guarded` held-out test `decoded_rollout_nrmse = 0.20177292896682064`.
- CT1 held-out per-task test NRMSE: advection `0.22508631227914033`, Burgers `0.17446879896821743`, Darcy `0.20909553062258152`.
- CT1 validation-selected metric before the one ledger-protected held-out confirmation: `decoded_rollout_nrmse = 0.1419775490176828`.
- CT1 improves over CT8 overall by `0.21480913046006705` absolute / `0.5156466189532753` relative and on advection by `0.3515000210587629` absolute / `0.6096225330626585` relative.
- CT1 is not the same exact inference contract as the CT8 primary claim, not an autonomous rollout claim, not an external-paper reproduction, and not directly comparable to published table values. It is reportable only as a separate scoped `light-v1 CT1 online transport-context UPS variant`.

Model-side validation gate without online context roll-shift:

- Added a training-time `task_loss_weights` lever for decoded operator/joint fine-tuning and used it only on `train`.
- Best selected validation-only candidate: `ups_light_advection_weighted_operator_ft_val_w15_lr1e4_e8_alpha19`.
- Validation `decoded_rollout_nrmse = 0.3535522468895649`, advection `0.4909265135126871`, Burgers `0.14738121412908425`, Darcy `0.188979512124482`.
- Improvement over the previous best no-context validation baseline `ups_light_local_joint_rollout4_residual_ft_val_transport_alpha18`: overall `0.00021343708549476093` absolute / `0.0006033289693236842` relative and advection `0.00032889772207145285` absolute / `0.0006695045276850516` relative.
- Evidence: `docs/claim_evidence/ups_advection_model_gate_val_evidence.json`; artifact SHA256 `90951476e2810608724cbf479ba10cfd91190fb4e29854dd33d44e9f9a6e414b`.
- Broader sweep update: selected validation-only candidate `ups_light_advection_weighted_operator_sweep_w15_lr1e4_e8_r8_alpha21`.
- Broader sweep validation `decoded_rollout_nrmse = 0.3514883905111875`, advection `0.4877450650030357`, Burgers `0.14738121412908425`, Darcy `0.188979512124482`.
- Broader sweep improvement over the previous best no-context validation baseline `ups_light_local_joint_rollout4_residual_ft_val_transport_alpha18`: overall `0.002277293463872121` absolute / `0.006437293290529189` relative and advection `0.0035103462317228606` absolute / `0.00714566425415995` relative.
- Broader sweep evidence: `docs/claim_evidence/ups_advection_model_sweep_val_evidence.json`; artifact SHA256 `f8f43e475812cd32e5e8cfb15a7c191e4dfd176c84ed1a4ebabb50927cb7e4c1`.
- Stability update: seed-23 replicate `ups_light_advection_weighted_operator_stability_seed23_w15_lr1e4_e8_r8_alpha21` validated at `decoded_rollout_nrmse = 0.35078329353213156`, advection `0.4866576789288726`, Burgers `0.14738121412908425`, Darcy `0.188979512124482`.
- Stability evidence: `docs/claim_evidence/ups_advection_model_stability_val_evidence.json`; artifact SHA256 `92e4c2ff63b9949bc5154ca53d8f5eedd8cfa20e8d436ac5b6b0363b01f11dd8`.
- Pre-test primary-contract registration: `docs/claim_evidence/ups_advection_model_primary_pretest_contract.json` records intended held-out measurement key `8afdb38b42feb138752101d74e11b95a0077eb2e8ba8cfb0fb5dffa6b67a5128` and the exact guarded command.
- Held-out primary-candidate result: `ups_light_advection_weighted_operator_stability_seed23_primary_guarded` measured test `decoded_rollout_nrmse = 0.5226095521324494`, advection `0.7373638522454458`, Burgers `0.17446879896821743`, Darcy `0.20909553062258152`.
- Held-out primary-candidate evidence: `docs/claim_evidence/ups_advection_model_primary_heldout_light_v1_evidence.json`; artifact SHA256 `d1f450e3487b9b208d52e45ee5654d5f946ebd2fd7875dda00079433a1a113d6`.
- The no-context model-side candidate failed to beat the current CT8 primary held-out claim (`0.5226095521324494` vs `0.4165820594268877`), so it must not be promoted.
- No-heldout-rerun gap analysis: `docs/claim_evidence/ups_advection_model_primary_gap_analysis.json` recomputes the validation/test gap from existing summaries only. It identifies long-horizon advection as the dominant failed-transfer signal: candidate minus CT8 held-out advection h16 is `0.648045534125835`, while Burgers and Darcy rollout errors are effectively unchanged.
- Phase-tracking validation gate contract: `docs/claim_evidence/ups_advection_phase_tracking_validation_gate_contract.json` now requires any future no-context primary candidate to clear validation-only thresholds on overall rollout, advection rollout, and advection h16 before a new held-out pre-test contract can be written.
- Phase-gate alpha diagnostic: `docs/claim_evidence/ups_advection_phase_alpha_diagnostic_val_evidence.json` swept transport residual alpha `0.0`, `0.1`, `0.21`, `0.3`, and `0.4` on validation only. No setting cleared the phase gate; alpha `0.21` remained best overall at `0.35078329353213156` but still failed h16 (`0.4938241237376044` vs required `<= 0.44444171136384397`).
- H16 training candidate: `docs/claim_evidence/ups_advection_h16_candidate_val_evidence.json` tried operator-decoded fine-tuning with `training_rollout_steps = 16` on validation only. It did not clear the phase gate: overall `0.3516245417982602`, advection rollout `0.487955006724297`, advection h16 `0.4961000768880384`.
- Horizon-weighted training candidate: `docs/claim_evidence/ups_advection_horizon_weighted_candidate_val_evidence.json` added opt-in `rollout_loss_horizon_power` and ran one validation-only candidate with power `2.0`. It improved slightly over the rollout-16 candidate but did not clear the phase gate: overall `0.35108013463927584`, advection rollout `0.48711549085823747`, advection h16 `0.4952380546421234`.

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
- Added `scoped_claim_variants` to `docs/claim_evidence/universal_sota_claim_evidence.json` and updated `scripts/audit_universal_sota_status.py` so the default audit reports CT1 as a validated scoped variant while keeping CT8 as the primary claim contract.

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
- Audit decision: CT1 is now machine-visible as `scoped_claim_variants.best_valid_variant`, not as `light_v1.best_run_name`; this prevents silently broadening the primary CT8 claim.

### 2026-06-01

Status:

- Promoted the CT8-vs-CT1 distinction from audit-only output into user-facing claim documentation.
- Added a machine-readable CT1 row to `docs/claim_evidence/external_baseline_mapping.json` under `scoped_claim_variants`, mirroring `docs/claim_evidence/universal_sota_claim_evidence.json`.
- Added validator coverage so the external-baseline mapping rejects CT1 metric drift and rejects overclaim flags that would mark CT1 as the same exact inference contract, an autonomous rollout claim, a published-number-comparable result, or an external-paper reproduction.
- Updated `docs/claim_evidence/external_baseline_mapping.md` with a side-by-side CT8 primary claim / CT1 scoped variant table.
- Updated the current evidence snapshot above so readers do not need to reconstruct CT1 status from the append-only worklog.

Decision:

- Keep CT8 as the primary frozen `light-v1` claim contract for broad claim-protocol comparisons against fair and external baselines.
- Report CT1 only as the scoped `light-v1 CT1 online transport-context UPS variant`, even though its held-out metric is much better, because the inference contract changed.
- Next model-side progress should target advection robustness without relying on the online roll-persistence correction, so a future candidate can improve the primary contract rather than only a scoped evaluation variant.

### 2026-06-01 Model-Side Advection Gate

Status:

- Added `task_loss_weights` support to decoded operator and joint codec/operator training, with unit coverage in `tests/unit/test_losses.py`.
- Ran the first validation-only model-side advection gate from `reports/research/sota_loop/learned_capacity_gate/ups_light_local_joint_rollout4_residual_ft_val`, using `train` only for fine-tuning and `val` only for selection.
- Rejected the joint codec/operator `advection1d:3.0` run because validation worsened to `decoded_rollout_nrmse = 0.3566482531298049`.
- Rejected the operator-only `advection1d:3.0` run because it missed the incumbent no-context gate at `0.3539187529949886`.
- Accepted the operator-only `advection1d:2.0` and `advection1d:1.5` runs as validation improvements; `advection1d:1.5` was stronger before alpha sweep at `0.3535584718194382`.
- Selected alpha `0.19` for the `advection1d:1.5` operator-only run on validation, producing `decoded_rollout_nrmse = 0.3535522468895649` and advection `0.4909265135126871`.
- Packaged validation-only evidence at `docs/claim_evidence/ups_advection_model_gate_val_evidence.json` and artifact SHA256 `90951476e2810608724cbf479ba10cfd91190fb4e29854dd33d44e9f9a6e414b`.
- Added `scripts/validate_ups_advection_model_gate_evidence.py` and `tests/unit/test_validate_ups_advection_model_gate_evidence.py` to guard the validation-only boundary, selected alpha-sweep best, positive improvement, empty context/observed/prediction shift estimators, and artifact SHA/size.

Decision:

- This is real model-side movement in the right direction because it changes train-time loss weighting and improves the no-context validation metric without CT1 online roll-persistence correction.
- The margin is too small to justify held-out primary-contract spend. Continue validation-only sweeps around advection weights `1.25-1.75`, lower learning rates, and `rollout_steps` `4-8` before pre-registering any held-out test.

Next checkpoint:

- Run repository checks for the new evidence and roadmap files.
- Next technical path: protocol-review the CT1 context-delay variant, then either wire it into the claim audit and run one ledger-protected held-out confirmation, or reject it as protocol-shift evidence and open a model-side advection objective instead.
- If CT1 is accepted, add pre-test claim audit wiring that names the validation evidence manifest, selected CT1 command, intended held-out command, and ledger key; only then run the held-out confirmation.
- The pre-test contract is now in place. The next irreversible step is to spend exactly one held-out test measurement using the pre-registered command, then package/update claim evidence only if the held-out result beats the current CT8 held-out claim and is documented with the scoped CT1 language.
- Next technical path: add a scoped CT1 claim row/audit path so downstream status scripts can distinguish the CT1 online-context result from the older CT8 claim and from external-paper SOTA claims.
- Next technical path: update user-facing claim documentation to describe the CT8 primary claim and CT1 online-context variant side by side, then decide whether to pursue a model-side advection objective that does not depend on online roll-persistence correction.

### 2026-06-01 Broader Model-Side Advection Sweep

Status:

- Ran a validation-only decoded-operator sweep around the small model-side signal, still from `reports/research/sota_loop/learned_capacity_gate/ups_light_local_joint_rollout4_residual_ft_val`, using `train` only for fine-tuning and `val` only for selection.
- Rejected `advection1d:1.75`, `learning_rate = 0.0001`, `rollout_steps = 4`, `alpha = 0.19` because validation worsened to `decoded_rollout_nrmse = 0.3535970281914783`.
- Rejected `advection1d:1.5`, `learning_rate = 0.00005`, `epochs = 12`, `rollout_steps = 4`, `alpha = 0.19` because validation worsened to `decoded_rollout_nrmse = 0.3546448305722839`.
- Accepted `advection1d:1.25`, `learning_rate = 0.0001`, `epochs = 8`, `rollout_steps = 4`, `alpha = 0.19` as a small validation improvement at `decoded_rollout_nrmse = 0.35352167517959965`.
- Accepted `advection1d:1.5`, `learning_rate = 0.0001`, `epochs = 8`, `rollout_steps = 8`, `alpha = 0.19` as the first materially better model-side candidate at `decoded_rollout_nrmse = 0.35165618765263623`.
- Selected alpha `0.21` for the `rollout_steps = 8` candidate on validation, producing `decoded_rollout_nrmse = 0.3514883905111875` and advection `0.4877450650030357`.
- Packaged validation-only sweep evidence at `docs/claim_evidence/ups_advection_model_sweep_val_evidence.json` and artifact SHA256 `f8f43e475812cd32e5e8cfb15a7c191e4dfd176c84ed1a4ebabb50927cb7e4c1`.

Decision:

- This is a stronger model-side result than the previous gate because the validation margin expanded from `0.00021343708549476093` to `0.002277293463872121` overall and from `0.00032889772207145285` to `0.0035103462317228606` on advection versus the same no-context baseline.
- This still is not a held-out primary-contract result. The sweep used validation for model and alpha selection, did not read `test`, and kept online context, observed, and prediction roll-shift estimators empty.
- The result is now large enough to justify either one stability-focused validation replicate around the selected setting or a pre-test protocol contract review, but not enough to silently spend held-out budget without that checkpoint.

Next checkpoint:

- Run the evidence validator, artifact hash/contents checks, targeted unit tests, lint, formatting checks, and the full pytest suite.
- If checks pass, open a PR for the sweep evidence and roadmap update.
- Next technical path: run a stability-only validation replicate near `advection1d:1.5`, `learning_rate = 0.0001`, `epochs = 8`, `rollout_steps = 8`, `alpha = 0.21`, or write the pre-test primary-contract confirmation contract before any held-out command.

### 2026-06-01 Model-Side Advection Stability Gate

Status:

- Ran a seed-23 validation-only stability replicate for the selected no-context setting: `advection1d:1.5`, `learning_rate = 0.0001`, `epochs = 8`, `rollout_steps = 8`, validation transport alpha `0.21`.
- Kept the same protocol boundary as the broader sweep: `train` only for fine-tuning, `val` only for selection/evidence, `max_samples = 32`, 16-step decoded rollout, and empty context/observed/prediction roll-shift estimators.
- The seed-23 replicate passed the validation promotion rule against the no-context baseline and produced `decoded_rollout_nrmse = 0.35078329353213156`, advection `0.4866576789288726`, Burgers `0.14738121412908425`, Darcy `0.188979512124482`.
- The seed-23 result improves over the seed-17 selected sweep by `0.0007050969790559636` overall and `0.0010873860741631436` on advection.
- Packaged validation-only stability evidence at `docs/claim_evidence/ups_advection_model_stability_val_evidence.json` and artifact SHA256 `92e4c2ff63b9949bc5154ca53d8f5eedd8cfa20e8d436ac5b6b0363b01f11dd8`.
- Extended `scripts/validate_ups_advection_model_gate_evidence.py` so stability evidence must include at least two distinct seeds, every replicate must beat the baseline, the selected candidate must be the best replicate, and the artifact SHA/size/selected summary must match.

Decision:

- This clears the stability-only validation gate: the material improvement survived a distinct seed and improved further under the same no-context protocol.
- This still cannot be reported as a held-out primary-contract claim. It is the validation-selected candidate family that can justify the next irreversible step only after a pre-test contract records the exact intended held-out command and ledger key.

Next checkpoint:

- Run the evidence validator, targeted tests, lint/formatting, artifact checks, and full pytest.
- If checks pass, open a PR for the stability evidence.
- Next technical path after merge: write the pre-test primary-contract confirmation contract for this no-context model-side candidate before any held-out command.

### 2026-06-01 Model-Side Primary Pre-Test Contract

Status:

- Wrote `docs/claim_evidence/ups_advection_model_primary_pretest_contract.json` for the seed-23 no-context model-side candidate, without reading or running held-out test.
- Registered validation evidence SHA256 `fecf10e6936e511fab091e6fc1936d41736fa52b02ea7ad4d1ed326d556ef306` for `docs/claim_evidence/ups_advection_model_stability_val_evidence.json`.
- Registered intended held-out measurement key `8afdb38b42feb138752101d74e11b95a0077eb2e8ba8cfb0fb5dffa6b67a5128`.
- Registered intended ledger path `reports/research/sota_loop/model_advection_primary_contract/test_ledger.json`.
- Added `scripts/validate_ups_advection_model_primary_pretest_contract.py` and unit tests so the command must use the seed-23 checkpoint source, must skip training, must include `--extra-eval-split test`, must use the ledger guard, must reject repeat-test bypass, must recompute the measurement key, and must not include online context/observed/prediction roll-shift estimators.

Decision:

- The next held-out spend is now auditable but still not executed. A future worker must run the validator immediately before any held-out command and should only run the exact registered command once.
- The no-context candidate is distinct from CT1 online-context evidence. If held-out later succeeds, claim audit and language still need an explicit update before replacing or superseding the current primary claim.

Next checkpoint:

- Run the new pre-test contract validator, targeted tests, lint/formatting, and full pytest.
- If checks pass, open a PR for the pre-test contract.
- Next technical path after merge: decide whether to spend exactly one held-out primary-contract confirmation using the registered command, or require an additional validation seed first.

### 2026-06-01 Model-Side Primary Held-Out Result

Status:

- Re-ran `scripts/validate_ups_advection_model_primary_pretest_contract.py` before any held-out access; it passed.
- The first execution attempt stopped before `test` because `--skip-training` without a checkpoint-preference stage selected `operator_joint.pt` ahead of the validation-selected `operator_decoded.pt`, causing validation promotion to fail at `0.35413255274913563`.
- Patched `scripts/run_light_experiment.py` so `--stage operator_decoded --skip-training` remains no-train but controls checkpoint preference, and added unit coverage for that behavior.
- Updated the pre-test contract command and measurement key to `8afdb38b42feb138752101d74e11b95a0077eb2e8ba8cfb0fb5dffa6b67a5128`.
- Re-ran the validation-only path with `--stage operator_decoded --skip-training`; it selected `operator_decoded.pt`, reproduced validation `decoded_rollout_nrmse = 0.35078329353213156`, and passed promotion.
- Ran the registered held-out command exactly once. The ledger recorded measurement key `8afdb38b42feb138752101d74e11b95a0077eb2e8ba8cfb0fb5dffa6b67a5128`.
- Held-out test result was `decoded_rollout_nrmse = 0.5226095521324494`, advection `0.7373638522454458`, Burgers `0.17446879896821743`, Darcy `0.20909553062258152`.
- Packaged held-out negative evidence at `docs/claim_evidence/ups_advection_model_primary_heldout_light_v1_evidence.json` and artifact SHA256 `d1f450e3487b9b208d52e45ee5654d5f946ebd2fd7875dda00079433a1a113d6`.

Decision:

- Do not promote the no-context model-side candidate. It failed the held-out primary comparison by `0.10602749270556172` overall and `0.16077751890754255` on advection versus the current CT8 primary claim.
- Do not repeat this held-out key. The result is negative evidence that the validation-only no-context advection gain did not transfer to held-out test.

Next checkpoint:

- Validate the held-out evidence manifest, artifact, runner checkpoint-preference patch, and full test suite.
- If checks pass, open a PR for the negative held-out evidence.
- Next technical path after merge: analyze the validation/test distribution gap and return to validation-only model-side work, not held-out reruns.

### 2026-06-01 Model-Side Primary Gap Analysis

Status:

- Added `docs/claim_evidence/ups_advection_model_primary_gap_analysis.json`, generated from the already-committed held-out evidence artifact and the current CT8 primary validation/test summaries.
- Added `scripts/build_ups_advection_model_primary_gap_analysis.py` so the analysis is reproducible without running a new held-out command or rereading held-out data.
- Added `scripts/validate_ups_advection_model_primary_gap_analysis.py` and unit coverage so future edits must preserve the no-rerun boundary, negative promotion decision, source-file hashes, and recomputed metric deltas.
- The failed candidate's held-out test is worse than CT8 by `0.10602749270556172` overall and `0.16077751890754255` on advection rollout.
- The largest diagnostic delta is held-out advection h16: candidate `0.7523448239495274` versus CT8 `0.10429928982369245`, a candidate-minus-CT8 regression of `0.648045534125835`.
- The candidate is slightly better than CT8 at held-out advection step1 and h4, but the h16 collapse dominates the rollout metric.
- Burgers and Darcy held-out rollout errors are effectively unchanged versus CT8, so the next useful model-side target is transport phase tracking under validation-only gates.

Decision:

- Keep the failed no-context model-side candidate as negative evidence only.
- Do not rerun the held-out key or spend another primary-contract held-out test on this candidate family.
- The next validation-only gate should require improvement on overall, advection rollout, and long-horizon advection h16 before any future held-out pre-test contract is written.

Next checkpoint:

- Run the gap-analysis validator, targeted unit tests, lint/formatting, and full pytest.
- If checks pass, open a PR for the gap-analysis evidence.
- Next technical path after merge: design a validation-only robustness gate for transport phase tracking rather than another advection loss-weight sweep that can overfit the validation rollout average.

### 2026-06-01 Phase-Tracking Validation Gate Contract

Status:

- Added `docs/claim_evidence/ups_advection_phase_tracking_validation_gate_contract.json` as the next no-context model-side pre-held-out gate.
- Added `scripts/validate_ups_advection_phase_tracking_gate_contract.py` and unit coverage so the contract can validate itself and evaluate future validation summaries.
- The contract explicitly does not authorize held-out test access. Passing this gate only permits writing a separate pre-test held-out contract with a new ledger key.
- Required candidate protocol stays on `val`, with the light-v1 task bundle, 32-sample cap, 16-step decoded rollout, `operator_decoded` checkpoint preference, and empty context/observed/prediction roll-shift estimators.
- Future candidates must beat the failed candidate's validation overall `decoded_rollout_nrmse = 0.35078329353213156`, beat validation advection rollout `0.4866576789288726`, and improve validation advection h16 by at least 10% to `<= 0.44444171136384397`.
- The previously failed candidate does not clear this gate, which is intentional: it reproduced the validation-selected score but then failed held-out on long-horizon advection.

Decision:

- Use this gate before any future no-context primary held-out pre-test contract.
- The gate is stricter than the previous validation average because the held-out miss showed that average validation improvement was not enough to protect transport phase tracking.

Next checkpoint:

- Run the phase-gate validator, targeted tests, lint/formatting, and full pytest.
- If checks pass, open a PR for the contract.
- Next technical path after merge: run validation-only model-side candidates against this gate, prioritizing changes that directly reduce advection h16 without reintroducing online context correction.

### 2026-06-01 Phase-Gate Alpha Diagnostic

Status:

- Ran validation-only skip-training diagnostics from the seed-23 no-context checkpoint with transport residual alpha values `0.0`, `0.1`, `0.21`, `0.3`, and `0.4`.
- Kept the run on `val` only with no held-out command, no ledger, the light-v1 32-sample cap, 16-step decoded rollout, and `operator_decoded` checkpoint preference.
- Packaged summaries in `docs/claim_evidence/artifacts/ups_advection_phase_alpha_diagnostic_val.tar.gz` with SHA256 `4b77e2f88882a3567b75b1a3c4c2db1c47a8d989e8c97c4e0ab7197f90545ab5`.
- Added `docs/claim_evidence/ups_advection_phase_alpha_diagnostic_val_evidence.json` plus validator/test coverage.
- Results: alpha `0.0` overall `0.3685752310100123`, advection rollout `0.5140255043059492`, advection h16 `0.5139634622080358`.
- Results: alpha `0.1` overall `0.3561013566786569`, advection rollout `0.49485317390239136`, advection h16 `0.49734736642642574`.
- Results: alpha `0.21` overall `0.35078329353213156`, advection rollout `0.4866576789288726`, advection h16 `0.4938241237376044`.
- Results: alpha `0.3` overall `0.353259160237994`, advection rollout `0.49047484357229637`, advection h16 `0.5026802065601567`.
- Results: alpha `0.4` overall `0.3630639287116581`, advection rollout `0.5055630865381556`, advection h16 `0.5242369940678469`.

Decision:

- No transport residual alpha setting on the existing seed-23 checkpoint clears the phase-tracking validation gate.
- Do not write a held-out pre-test contract from this diagnostic.
- The next useful model-side work should train or alter the model to reduce validation advection h16 directly; post-hoc alpha tuning around this checkpoint is not enough.

Next checkpoint:

- Run the diagnostic evidence validator, targeted tests, lint/formatting, and full pytest.
- If checks pass, open a PR for the diagnostic evidence.
- Next technical path after merge: add a training-side long-horizon advection objective or sampling change, then evaluate that new validation summary against the phase gate.

### 2026-06-01 H16 Training Candidate

Status:

- Ran one bounded train/validation-only no-context candidate from `reports/research/sota_loop/learned_capacity_gate/ups_light_local_joint_rollout4_residual_ft_val`.
- Candidate: `ups_light_advection_h16_operator_w15_lr1e4_e8_r16_alpha21`.
- Training changed the decoded-operator rollout length from the prior successful `8` to `16`, kept `advection1d:1.5` task loss weighting, `epochs = 8`, `learning_rate = 0.0001`, seed `31`, and used `operator_decoded` checkpoint preference.
- Evaluation stayed on `val`, used 32 samples, 16-step decoded rollout, no online context/observed/prediction roll-shift estimator, and transport residual alpha `0.21`.
- Result: overall validation `decoded_rollout_nrmse = 0.3516245417982602`, advection rollout `0.487955006724297`, advection h16 `0.4961000768880384`, Burgers `0.14738121412908425`, Darcy `0.188979512124482`.
- Packaged evidence at `docs/claim_evidence/ups_advection_h16_candidate_val_evidence.json` and artifact SHA256 `12e260ca702d00b3f70f8b72f3005c14e34c0b39826ab4302b2473b8d3c475b3`.

Decision:

- The rollout-16 decoded-operator candidate does not clear the phase gate and should not advance to any held-out pre-test contract.
- Simply increasing training rollout length from 8 to 16 on the same operator-only fine-tuning path worsened the h16 signal; the next candidate needs a different objective or data sampling change.

Next checkpoint:

- Run the h16-candidate evidence validator, targeted tests, lint/formatting, and full pytest.
- If checks pass, open a PR for the negative candidate evidence.
- Next technical path after merge: introduce an explicit horizon-weighted training loss or advection temporal-window sampling change, then validate against the same phase gate.

### 2026-06-01 Horizon-Weighted Training Candidate

Status:

- Added an opt-in decoded rollout training knob: `stages.<stage>.rollout_loss_horizon_power`.
- Default behavior is unchanged: `rollout_loss_horizon_power = 0.0` preserves uniform averaging over rollout losses.
- A positive horizon power weights later decoded rollout losses more heavily while keeping the loss scale normalized by total weight.
- Ran one bounded train/validation-only candidate: `ups_light_advection_horizon_weighted_operator_w15_lr1e4_e8_r16_p2_alpha21`.
- Candidate settings: `operator_decoded`, seed `37`, `epochs = 8`, `learning_rate = 0.0001`, `training_rollout_steps = 16`, `rollout_loss_horizon_power = 2.0`, task weights `advection1d:1.5`, `burgers1d:1.0`, `darcy2d:1.0`.
- Evaluation stayed on `val`, used 32 samples, 16-step decoded rollout, no online context/observed/prediction roll-shift estimator, and transport residual alpha `0.21`.
- Result: overall validation `decoded_rollout_nrmse = 0.35108013463927584`, advection rollout `0.48711549085823747`, advection h16 `0.4952380546421234`, Burgers `0.14738121412908425`, Darcy `0.188979512124482`.
- Compared with the prior rollout-16 candidate, the horizon-weighted candidate improved overall by `0.0005444071589843502`, advection rollout by `0.0008395158660595148`, and advection h16 by `0.0008620222459150284`.
- Packaged evidence at `docs/claim_evidence/ups_advection_horizon_weighted_candidate_val_evidence.json` and artifact SHA256 `1e78c67449aa12226312c99cc93ff5e8704282d645238e633dd24eeffc350a54`.

Decision:

- The opt-in horizon-weighted loss is a useful measured lever because it improves the prior rollout-16 candidate without changing default behavior.
- It still does not clear the phase gate and does not authorize any held-out pre-test contract.
- The remaining h16 gap is too large for horizon weighting alone; the next candidate should add a more direct transport phase signal or temporal-window sampling change.

Next checkpoint:

- Run the horizon-weighted evidence validator, loss tests, lint/formatting, and full pytest.
- If checks pass, open a PR for the code lever and negative candidate evidence.
- Next technical path after merge: combine `rollout_loss_horizon_power` with a transport-specific phase/shift consistency objective or train-split temporal-window sampling that targets advection h16 more directly.

### 2026-06-02 Temporal-Window Training Candidate

Status:

- Added an opt-in decoded rollout training-window knob: `stages.<stage>.rollout_start_strategy`.
- Default behavior is unchanged: `rollout_start_strategy = zero` preserves frame-0 decoded rollout supervision.
- The `latest` strategy trains the requested decoded rollout length from the latest available window in each sample and passes the absolute window start into structured conditioning.
- The implementation applies consistently to `operator_decoded` and `joint_codec_operator` decoded rollout supervision.
- Ran one bounded train/validation-only candidate: `ups_light_advection_temporal_latest_operator_w15_lr1e4_e8_r16_p2_alpha21`.
- Candidate settings: `operator_decoded`, seed `41`, `epochs = 8`, `learning_rate = 0.0001`, `training_rollout_steps = 16`, `rollout_start_strategy = latest`, `rollout_loss_horizon_power = 2.0`, task weights `advection1d:1.5`, `burgers1d:1.0`, `darcy2d:1.0`.
- Evaluation stayed on `val`, used 32 samples, 16-step decoded rollout, no online context/observed/prediction roll-shift estimator, and transport residual alpha `0.21`.
- Result: overall validation `decoded_rollout_nrmse = 0.35219359968828834`, advection rollout `0.48883238144252295`, advection h16 `0.4963078395394647`, Burgers `0.14738121412908425`, Darcy `0.188979512124482`.
- Compared with the horizon-weighted candidate, latest-window sampling worsened overall by `0.0011134650490124964`, advection rollout by `0.0017168905842854798`, and advection h16 by `0.0010697848973413304`.
- Packaged evidence at `docs/claim_evidence/ups_advection_temporal_window_candidate_val_evidence.json` and artifact SHA256 `567ba875f61e4bca66c18173809114f3c28ae4b59248620fedfe8a0aaf764a1b`.

Decision:

- The opt-in temporal-window lever is useful infrastructure because it is default-off, unit-tested, and exposes a real training-data axis.
- This specific latest-window candidate does not clear the phase gate, worsens the prior horizon-weighted candidate, and does not authorize any held-out pre-test contract.
- Temporal-window-only loss sweeps should stop unless paired with a direct phase target; the validation evidence points back to transport phase tracking rather than generic later-window supervision.

Next checkpoint:

- Run the temporal-window evidence validator, loss tests, lint/formatting, and full pytest.
- If checks pass, open a PR for the code lever and negative candidate evidence.
- Next technical path after merge: implement an explicit train/validation-only transport phase or shift-consistency objective, then validate against the same phase gate before any held-out access.

### 2026-06-02 Transport Shift-Consistency Training Candidate

Status:

- Added an opt-in decoded rollout training objective: `stages.<stage>.transport_shift_consistency_lambda`.
- Default behavior is unchanged: `transport_shift_consistency_lambda = 0.0` disables the phase/shift regularizer.
- The companion mapping `transport_shift_consistency_by_task` selects task-specific periodic shifts and currently applies only where configured.
- Ran a train/validation-only shift-fit diagnostic on `advection1d` with 32 train and 32 validation samples, 16-step rollout, and candidate shifts `[-4, -2, -1, 0, 1, 2, 4, 8, 16, 24, 32, 40, 48, 64]`.
- Shift diagnostic result: train-selected shift `1` with train NRMSE `0.00032598856823269374`; validation oracle shift `40` with validation NRMSE `0.012850484431904393`; locked train shift `1` measured validation NRMSE `0.5027994425730189`.
- Ran one bounded train/validation-only candidate using the train-selected shift, not the validation oracle: `ups_light_advection_transport_shift1_operator_w15_lr1e4_e8_r16_p2_scl1_alpha21`.
- Candidate settings: `operator_decoded`, seed `43`, `epochs = 8`, `learning_rate = 0.0001`, `training_rollout_steps = 16`, `rollout_loss_horizon_power = 2.0`, `transport_shift_consistency_lambda = 1.0`, `transport_shift_consistency_by_task = {advection1d: 1}`, task weights `advection1d:1.5`, `burgers1d:1.0`, `darcy2d:1.0`.
- Evaluation stayed on `val`, used 32 samples, 16-step decoded rollout, no online context/observed/prediction roll-shift estimator, and transport residual alpha `0.21`.
- Result: overall validation `decoded_rollout_nrmse = 0.35199049383849107`, advection rollout `0.4885192504506526`, advection h16 `0.4968563765605278`, Burgers `0.14738121412908425`, Darcy `0.188979512124482`.
- Compared with the horizon-weighted candidate, train-fitted shift-consistency worsened overall by `0.0009103591992152314`, advection rollout by `0.001403759592415104`, and advection h16 by `0.0016183219184043799`.
- Packaged evidence at `docs/claim_evidence/ups_advection_transport_shift_consistency_candidate_val_evidence.json` and artifact SHA256 `1591328abd0f7a032e17ba10fbd8f8622649b56be68b22ebb16f392fa8ecedb9`.

Decision:

- The opt-in shift-consistency loss is useful infrastructure because it is default-off and captures a concrete transport phase prior without changing inference.
- This specific train-fitted fixed-shift candidate does not clear the phase gate, worsens the prior horizon-weighted candidate, and does not authorize any held-out pre-test contract.
- The train/validation shift mismatch shows why fixed train-fitted phase regularization is insufficient for this local light-v1 split.

Next checkpoint:

- Run the transport shift-consistency evidence validator, loss tests, lint/formatting, and full pytest.
- If checks pass, open a PR for the code lever and negative candidate evidence.
- Next technical path after merge: move to a data-conditioned phase estimator or source/parameter-aware transport objective that can explain the train/validation shift mismatch without using validation-oracle shifts for training.

### 2026-06-07 Data-Conditioned Context-Phase Validation Candidate

Status:

- Imported the June 4 literature/ecosystem landscape into this worktree and wrote the execution plan at `docs/superpowers/plans/2026-06-07-causal-transport-phase-estimator-plan.md`.
- Added a default-off decoded evaluator key: `evaluation.decoded_data_conditioned_roll_shift_estimator`.
- The estimator supports field-stat features and an explicit causal `context_shift` feature inferred from early observed context transitions, then applies a train-fitted linear coefficient during validation.
- Static field-feature diagnostic failed as expected: train first-32 rows are a `+1` transport regime, validation first-32 rows are a `+41` regime, and field moments alone produced validation NRMSE `0.5027994693398018`, worse than the `0.4866576789288726` advection reference.
- Context-feature train-fit gate cleared validation without held-out reads: `context_shift` coefficient `0.9999999979166667`, train context shift mean `1.0`, validation context shift mean `41.0`, validation direct transport NRMSE `0.0005432125951258969`.
- Ran validation-only decoded candidate `ups_light_advection_data_conditioned_context_phase_val` with the frozen checkpoint, split `val`, 32 samples, 16 decoded steps, `transport` residual alpha `0.21`, and no held-out/test split.
- Result: overall validation `decoded_rollout_nrmse = 0.1379312547168074`, advection rollout `0.11936459958552438`, advection h1 `0.4774484941309337`, advection h16 `0.0005560538043379871`, Burgers `0.14738121412908425`, Darcy `0.188979512124482`.
- The existing advection phase-tracking validator passes this candidate with no errors.
- Packaged evidence at `docs/claim_evidence/ups_advection_data_conditioned_phase_candidate_val_evidence.json` and artifact SHA256 `f2250d9c7e1db1f070fe3bceab5b98fe5a783d248bd2bcf3d9a972a9a96ff4ab`.

Decision:

- P1 is validated on the current validation protocol: a train-fitted data-conditioned `context_shift` estimator explains the train/validation phase mismatch that fixed train-fitted shifts could not.
- This does not authorize held-out access by itself. It clears the validation phase gate and permits writing a separate held-out pre-test contract with a new measurement key.
- The tradeoff is explicit: the candidate uses one observed transition to infer phase and applies roll-persistence from horizon 2 onward, so it is a scoped data-conditioned phase-estimator protocol variant, not a no-context primary model-capacity claim or external-paper reproduction.

Next checkpoint:

- Run the data-conditioned evidence validator, focused unit tests, lint/formatting, and full pytest.
- If checks pass, open a PR for the plan, default-off estimator, train-fit gate, validation evidence, and validator.
- Next technical path after merge: write a pre-test held-out contract for this exact data-conditioned phase-estimator variant before any held-out command, or proceed to P2 learned warp sidecar if the protocol review rejects one-transition context inference as a claim variant.

### 2026-06-08 Data-Conditioned Context-Phase Pre-Test Contract

Status:

- Added `docs/claim_evidence/ups_advection_data_conditioned_pretest_contract.json`.
- Added `scripts/validate_ups_advection_data_conditioned_pretest_contract.py` and tests for the intended held-out command, measurement-key recomputation, ledger-repeat rejection, and protocol-disclosure fields.
- Pre-registered intended held-out measurement key `361686cf61ebc532c34bd43be7920515c8864db05aa61226ca88a670dacdf88a`.
- Ledger path: `reports/research/sota_loop/data_conditioned_transport_phase/test_ledger.json`.
- Intended command remains `pre_registered_not_run`; no held-out test split was accessed by this contract work.
- The intended command carries the exact `evaluation.decoded_data_conditioned_roll_shift_estimator` selected by the validation evidence, all three phase-gate promotion rules, `--extra-eval-split test`, and the held-out ledger guard.

Decision:

- The data-conditioned context-phase candidate is accepted for exactly one ledger-protected held-out confirmation as a scoped `light-v1 data-conditioned context-phase UPS variant`.
- The claim-language boundary remains explicit: this is not a no-context autonomous rollout claim, not the same inference contract as no-context primary candidates, and not an external-paper reproduction.
- Repeated held-out measurements under the same key are forbidden unless a future debugging-only path explicitly opts into repeat testing and is not used as new claim evidence.

Next checkpoint:

- Run `scripts/validate_ups_advection_data_conditioned_pretest_contract.py`, targeted tests, lint/formatting, and the full suite.
- If checks pass, open a PR for the pre-test contract and validator.
- After merge, run the intended held-out command exactly once if the scoped claim-language boundary is still accepted.

### 2026-06-08 Data-Conditioned Context-Phase Held-Out Confirmation

Status:

- Ran the exact pre-registered held-out command from `docs/claim_evidence/ups_advection_data_conditioned_pretest_contract.json` once after the pre-test contract merged.
- The held-out ledger recorded measurement key `361686cf61ebc532c34bd43be7920515c8864db05aa61226ca88a670dacdf88a` at `reports/research/sota_loop/data_conditioned_transport_phase/test_ledger.json`.
- Packaged the evidence artifact at `docs/claim_evidence/artifacts/ups_advection_data_conditioned_heldout_light_v1.tar.gz` with SHA256 `6f52e621d356c4e33e1b016eaef2ced8ba43f3e491eef87c005189ae0437275f`.
- Added `docs/claim_evidence/ups_advection_data_conditioned_heldout_light_v1_evidence.json` plus a validator for artifact hashes, ledger discipline, command-derived measurement key, summary metrics, and scoped claim language.
- Held-out result: overall `decoded_rollout_nrmse = 0.1808155304023394`, advection rollout `0.18345021264323003`, advection h1 `0.7336025534824926`, advection h16 `0.004232970377814687`, Burgers `0.17446879896821743`, Darcy `0.20909553062258152`.
- The result improves over the frozen CT8 primary metric by `0.2357665290245483` absolute overall and over the CT1 scoped variant by `0.02095739856448125` absolute overall.

Decision:

- This is the best scoped held-out `light-v1` UPS variant recorded so far by overall `decoded_rollout_nrmse`.
- It remains a separate `light-v1 data-conditioned context-phase UPS variant`, not a replacement for the frozen CT8 primary claim, not a no-context autonomous rollout claim, and not an external-paper reproduction.
- The key tradeoff is now explicit and measurable: one observed transition gives a strong phase estimate, but that makes the inference contract data-conditioned/online rather than initial-condition-only.

Next checkpoint:

- Run the held-out evidence validator, external mapping validator, audit, targeted tests, lint/formatting, and the full suite.
- If checks pass, open a PR for the evidence package and scoped-variant claim updates.
- Next technical path after merge: pursue a validation-only learned warp/transport sidecar or model-side objective that preserves the metric win while reducing the one-transition teacher-forced dependency before any broader primary-claim replacement.

### 2026-06-09 Medium Confirmation Contract and Next Validation Goals

Status:

- Added `docs/claim_evidence/medium_v1_confirmation_evidence.json` to make the existing medium-v1 confirmation machine-readable instead of relying on the manual `--medium-confirmed` audit flag.
- Medium confirmation recorded: `ups_medium_shared_context_transport` on `medium-v1` test reached `decoded_rollout_nrmse = 0.30616533327650614` versus `persistence_medium_v1_test = 0.5725109200102603`, a `0.46522359211765046` improvement fraction.
- Added `scripts/validate_medium_confirmation_evidence.py` and tests for sample-count floor, no test tuning, improvement threshold, and artifact-handle requirements.
- Updated `scripts/audit_universal_sota_status.py` so validated medium evidence satisfies the `medium_or_larger_confirmation` readiness check without a manual flag.
- Added `docs/claim_evidence/ups_advection_next_validation_contracts.json` and `scripts/validate_next_validation_contracts.py` to define the next safe experiment goals: P2 learned warp/transport sidecar validation and data-conditioned weaker/no-context ablation.

Decision:

- The default audit can now close the medium-or-larger confirmation gate from committed evidence for the CT8/shared-context primary path.
- This does not promote the data-conditioned scoped variant to a primary claim. That variant remains the best scoped light-v1 held-out metric but still needs separate medium-scale evidence and reduced-context ablation before broader claim language.
- The next experiments remain validation-only: no new held-out command is authorized by these contracts.

Next checkpoint:

- Run the medium confirmation validator, next-validation contract validator, audit, focused tests, lint/formatting, and full pytest.
- If checks pass, open a PR for the medium evidence/audit wiring and next-validation contracts.
- Next technical path after merge: implement the P2 learned warp/transport sidecar or run the required ablation matrix on validation, then package evidence before any new pretest contract.

### 2026-06-09 Data-Conditioned Context Ablation Matrix

Status:

- Added `scripts/run_data_conditioned_ablation_matrix.py` to run the required validation-only ablation matrix from `docs/claim_evidence/ups_advection_next_validation_contracts.json`.
- Added `docs/claim_evidence/ups_advection_data_conditioned_ablation_val_evidence.json` plus `scripts/validate_ups_advection_data_conditioned_ablation_evidence.py`.
- Ran the matrix on `data/pdebench`, split `val`, task `advection1d`, `max_samples = 32`, `rollout_steps = 16`, with no held-out test reads and no ledger writes.
- Full context-shift variant: validation NRMSE `0.0005432125951258969`, predicted validation shift mean `40.999999914583334`.
- Weaker bounded context-shift variant with candidate shifts `[-8, 8]`: validation NRMSE `0.421816253136215`, absolute delta versus full context `0.4212730405410891`.
- No-data-conditioning variant with only a train-fitted bias shift: validation NRMSE `0.5027994693398018`, absolute delta versus full context `0.5022562567446759`.
- Durable matrix artifacts are committed under `docs/claim_evidence/artifacts/ups_advection_data_conditioned_ablation_matrix.json` and `docs/claim_evidence/artifacts/ups_advection_data_conditioned_ablation_matrix_parts/`.

Decision:

- The ablation confirms the current data-conditioned win is strongly dependent on broad online context-shift estimation.
- The result supports keeping the data-conditioned candidate scoped as teacher/context-dependent evidence rather than promoting it as a no-context primary model-capacity claim.
- This evidence does not authorize a new held-out run. It makes P2 the next optimal technical path: a validation-only learned warp/transport sidecar that tries to retain the phase win while reducing context dependence.

Next checkpoint:

- Run the ablation evidence validator, focused tests, lint/formatting, and full pytest.
- If checks pass, open a PR for the ablation runner, evidence package, and roadmap update.
- Next technical path after merge: implement the P2 learned warp/transport sidecar with acceptance gates below the full context-shift validation metrics recorded here.

### 2026-06-09 P2 Parameter-Conditioned Transport Sidecar Validation

Status:

- Ran a validation-only reduced-context transport-sidecar probe using `scripts/run_parameter_conditioned_transport_shift_gate.py`.
- Data scope: `data/pdebench_official_advection_light`, train split `256` rows, validation split `64` rows, `rollout_steps = 16`, dense candidate shifts `[-80, 80]`, fractional refinement step `0.5`, no held-out test split, and no ledger writes.
- The sidecar fits a linear shift rule from official Advection `beta` metadata on train only: `shift = 10.236877359639507 * beta - 0.08098891730605368`.
- Validation result: train NRMSE `0.0012031011109454775`, validation NRMSE `0.001981674036057911`.
- The validation metric clears the P2 support threshold derived from the data-conditioned validation candidate reference `0.1379312547168074`.
- Added `docs/claim_evidence/ups_advection_p2_parameter_conditioned_sidecar_val_evidence.json`, durable report artifacts under `docs/claim_evidence/artifacts/`, and `scripts/validate_p2_parameter_conditioned_sidecar_evidence.py`.
- Also recorded a dense context-inferred comparison on `data/pdebench`: validation NRMSE `0.26177843531949696`, showing the beta-conditioned official-shard route is the stronger reduced-context signal.

Decision:

- This is real P2 progress: known PDE parameter metadata can recover advection transport phase without observed context transitions.
- It is not yet a primary `light-v1` decoded claim replacement because it is an advection-only direct transport-sidecar measurement on official balanced shards, not a full multitask decoded rollout through the frozen `data/pdebench` claim protocol.
- No held-out test is authorized by this evidence. The next step is to integrate this parameter-conditioned sidecar into the decoded evaluator or model-side conditioning path and evaluate it on validation before any pretest contract.

Next checkpoint:

- Run the P2 sidecar evidence validator, focused tests, lint/formatting, and full pytest.
- If checks pass, open a PR for the evidence package and validator.
- Next technical path after merge: implement a default-off decoded evaluator hook for parameter-conditioned transport sidecars, then validate it against the full light-v1 decoded rollout contract.

### 2026-06-09 P2 Parameter-Conditioned Decoded Evaluator Integration

Status:

- Added official-source-provenance `beta` derivation to `PDEBenchDataset` when `data.param_keys = [beta]` and an HDF5 shard has `source_file_index` plus `source_paths`.
- Extended the default-off decoded evaluator data-conditioned roll-shift path with explicit `param:<name>` features, so the locked P2 sidecar can use `param:beta` without observed context transitions.
- Ran validation-only decoded evaluation with the frozen checkpoint, original three-task conditioning vocabulary, `data/pdebench_official_advection_light`, split `val`, `max_samples = 64`, `decoded_rollout_steps = 16`, no `--extra-eval-split test`, and no ledger writes.
- Locked estimator: `feature_names = [param:beta, bias]`, `shift = 10.236877359639507 * beta - 0.08098891730605368`, `mode = roll_persistence`, `min_horizon = 1`.
- Result: decoded validation `decoded_rollout_nrmse = 0.0019816594876579004`, advection h1 `0.002066357660843727`, advection h16 `0.0019494709566432954`, mean applied shift `19.62499999809323`.
- The decoded result matches the standalone sidecar within `1.4548390010510294e-08` absolute NRMSE, so the P2 transport signal is now proven inside the decoded evaluator path rather than only in a bespoke sidecar script.
- Added `docs/claim_evidence/ups_advection_p2_parameter_decoded_sidecar_val_evidence.json`, summary artifact `docs/claim_evidence/artifacts/ups_advection_p2_parameter_decoded_sidecar_val_summary.json`, and validator `scripts/validate_p2_parameter_decoded_sidecar_evidence.py`.

Decision:

- This is a stronger P2 checkpoint than the standalone sidecar because it exercises the actual decoded rollout evaluator and checkpoint-loading path.
- It still is not a full multitask `light-v1` primary candidate: the official beta-provenance root contains advection only, so Burgers and Darcy were explicitly skipped while preserving the original three-task conditioning vocabulary.
- No held-out test is authorized by this evidence. It reduces observed-context dependency but introduces explicit PDE-parameter metadata dependency.

Next checkpoint:

- Run the decoded sidecar evidence validator, focused tests, lint/formatting, audit, and full pytest.
- If checks pass, open a PR for the evaluator integration, decoded evidence, validator, and roadmap update.
- Next technical path after merge: build a full multitask parameter-aware validation root or add model-side PDE-parameter conditioning so this transport win can be tested under the complete light-v1 task mix before any broader pretest contract.

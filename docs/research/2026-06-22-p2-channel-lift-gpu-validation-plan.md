# P2 Channel-Lift GPU Validation Plan

Date: 2026-06-22

Status: executed 2026-06-23 on Vast.ai for train/validation only. No held-out
test or claim evidence change was performed.

## Purpose

Run the first real P2.2 train/validation measurement for Poseidon ScOT
`channel_lift` Option A after the local CPU smoke passed. This tests whether
the pretrained Poseidon interface plus a tiny trainable channel adapter can
clear roadmap gate G2a under the frozen `light-v1` validation contract.

## Preconditions Already Satisfied

- Official Poseidon source restored at `/tmp/poseidon-official`.
- Source commit verified:
  `b8fa28f59bd7f7673323f28d11a12c6f3a215c61`.
- `scOT.model.ScOT` and `ScOTConfig` import successfully.
- Cached Poseidon-T checkpoint SHA256 verified in the CPU smoke:
  `e97428c93a16cbb52a41bc4794eb71be3aed436fb9cc547d9eeebb20f3940fb2`.
- 2-sample validation-only CPU smoke passed:
  `reports/research/sota_loop/external_baselines/poseidon_scot_channel_lift_smoke_val_light_v1/summary.json`.
- CPU smoke metric was `decoded_rollout_nrmse = 0.31116372295004086`, but this
  is not a gate result because it used only 2 train/eval samples and 4 rollout
  steps.

## Executed Approval Scope

User-approved scope:

- Run one bounded GPU train/validation measurement locally or on the existing
  GPU workflow.
- Use only `train` for fitting and `val` for selection.
- Do not use `test`, `--allow-held-out-test-eval`, or held-out ledger writes.
- Do not mutate external services beyond the approved GPU execution path.
- Stop after one Option A measurement unless the user separately approves an
  Option B or LoRA follow-up.

No held-out test was run under this scope.

## Proposed Command

Command shape used by the remote wrapper:

```bash
python scripts/run_external_poseidon_scot_finetune.py \
  --config configs/train_multitask_heterogeneous_light_best.yaml \
  --name poseidon_scot_channel_lift_val_light_v1_e30_lr1e2_roll4 \
  --output-root reports/research/sota_loop/external_baselines \
  --train-split train \
  --eval-split val \
  --max-train-samples 32 \
  --max-eval-samples 32 \
  --rollout-steps 16 \
  --poseidon-model-size T \
  --checkpoint-file model.safetensors \
  --device cuda \
  --time-value 1.0 \
  --data-root data/pdebench \
  --poseidon-repo /tmp/poseidon-official \
  --expected-checkpoint-sha256 e97428c93a16cbb52a41bc4794eb71be3aed436fb9cc547d9eeebb20f3940fb2 \
  --tasks advection1d burgers1d darcy2d \
  --epochs 30 \
  --learning-rate 0.01 \
  --weight-decay 0.0001 \
  --batch-size 32 \
  --grad-clip-norm 1.0 \
  --adapter-mode channel_lift \
  --rollout-loss-steps 4 \
  --rollout-loss-weight 1.0
```

Expected summary path:

```text
reports/research/sota_loop/external_baselines/poseidon_scot_channel_lift_val_light_v1_e30_lr1e2_roll4/summary.json
```

## Acceptance Checks

The run is valid only if:

- `status = "validation_finetune_measurement_complete"`.
- `split = "val"`.
- `held_out_test_used = false`.
- `details.adapter_mode = "channel_lift"`.
- `details.model.embedding_recovery_replaced = false`.
- `details.contract.pretrained_embedding_recovery_intact = true`.
- `details.trainable_parameters.trainable_parameter_count = 13` unless task
  modulation is intentionally added in a later approved run.
- `metrics.decoded_rollout_nrmse` is finite.
- Per-task metrics show no task collapse near `1.0`.

## Decision Gates

Use the existing roadmap gates:

- Clear G2a: validation `decoded_rollout_nrmse <= 0.363424243629033` with no
  task collapsing near `1.0`. This authorizes drafting a held-out pre-test
  contract, not running held-out test immediately.
- Continue zone: `0.3634 < decoded_rollout_nrmse <= 0.5`. This can justify a
  separately approved Option B shallow nonlinear lift/readout or controlled
  LoRA path.
- Stop current Poseidon Option A/B path: above `0.5` after clean Option A and
  Option B evidence. DPOT becomes the primary fallback.

## Post-Run Recording

Final valid run:

- Vast artifact:
  `b2://pdebench/remote-runs/poseidon-channel-lift/poseidon_channel_lift_light-v1_20260623T015718Z.tar.gz`.
- Local summary:
  `reports/research/sota_loop/external_baselines/poseidon_scot_channel_lift_val_light_v1_e30_lr1e2_roll4/summary.json`.
- Status: `validation_finetune_measurement_complete`.
- Split: `val`; train split: `train`; `held_out_test_used = false`.
- Aggregate validation `decoded_rollout_nrmse = 0.35782889238675264`.
- Per-task validation NRMSE:
  `advection1d = 0.4937043430599529`,
  `burgers1d = 0.15674926288225416`,
  `darcy2d = 0.2071060212271272`.
- Adapter/source integrity: `adapter_mode = channel_lift`, 13 trainable
  parameters, `embedding_recovery_replaced = false`,
  `pretrained_embedding_recovery_intact = true`, Poseidon source commit
  `b8fa28f59bd7f7673323f28d11a12c6f3a215c61`, checkpoint SHA256
  `e97428c93a16cbb52a41bc4794eb71be3aed436fb9cc547d9eeebb20f3940fb2`.

Decision:

- G2a cleared on aggregate validation because `0.35782889238675264 <=
  0.363424243629033`.
- This authorizes a held-out pre-test contract and evidence manifest only. It
  does not authorize running held-out test in the same action.
- Advection/transport remains the weak family and should be called out in the
  pre-test contract despite no task-level collapse near `1.0`.

Follow-up:

- Record summary path, metric, split, trainable parameter count, source commit,
  and checkpoint SHA256 in `docs/current-state.md` and
  `docs/experiments/ledger.md`.
- Prepare a pre-test contract and evidence manifest in a separate step. Do not
  run held-out test in the same action.
- If the run fails or is invalid, record the exact failure class before
  proposing Option B, LoRA, or DPOT fallback.

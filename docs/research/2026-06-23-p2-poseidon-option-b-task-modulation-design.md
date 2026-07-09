# P2 Poseidon Option B Task-Modulation Design

Date: 2026-06-23

Status: runner scaffolding and local CPU smoke complete. No provider/GPU work
ran for this step, no held-out test was used, no claim evidence changed, and
no public language changed.

## Purpose

Poseidon `channel_lift` Option A cleared aggregate validation G2a but failed
the single ledger-protected held-out pretest, dominated by advection/transport
generalization. DPOT Tiny `channel_lift` then missed the validation-only GPU
gate, also with weak transport behavior. The highest-signal next branch is a
small Poseidon Option B adapter that keeps the pretrained ScOT
embedding/recovery intact while adding task-specific calibration around the
already-working `channel_lift` path.

This is not a claim path yet. It is a validation-only branch intended to test
whether task conditioning can improve transport/advection without destroying
Burgers or Darcy.

## Adapter

Implementation:

- Runner: `scripts/run_external_poseidon_scot_finetune.py`
- Adapter mode: `channel_lift_task_modulated`
- Wrapper: `TaskModulatedChannelLiftScOT`
- Tests: `tests/unit/test_external_poseidon_scot_finetune.py`

Parameterization:

- Keep the frozen native-channel Poseidon ScOT backbone unchanged.
- Keep the base 1x1 scalar-to-native-channel lift.
- Keep the base 1x1 native-channel-to-scalar readout.
- Add per-task affine gain/bias before the frozen backbone.
- Add per-task affine gain/bias after scalar readout.
- Initialize all task gains to `1.0` and biases to `0.0`.

For Poseidon-T with 4 native channels and three `light-v1` tasks, the trainable
parameter count is:

- Base `channel_lift`: `13` parameters.
- Task modulation: `3 * (4 lift gains + 4 lift biases + 1 readout gain + 1 readout bias) = 30`.
- Total: `43` trainable parameters.

The identity initialization matters because Option B starts equivalent to
Option A at the task-conditioning boundary. Any improvement or regression
comes from a small trainable calibration layer, not from replacing Poseidon's
pretrained embedding/recovery path.

## Local CPU Smoke

Command:

```bash
python scripts/run_external_poseidon_scot_finetune.py \
  --config configs/train_multitask_heterogeneous_light_best.yaml \
  --name poseidon_scot_task_modulated_channel_lift_smoke_val_light_v1 \
  --output-root reports/research/sota_loop/external_baselines \
  --train-split train \
  --eval-split val \
  --max-train-samples 2 \
  --max-eval-samples 2 \
  --rollout-steps 4 \
  --poseidon-model-size T \
  --checkpoint-file model.safetensors \
  --device cpu \
  --time-value 1.0 \
  --data-root data/pdebench \
  --poseidon-repo /tmp/poseidon-official \
  --expected-checkpoint-sha256 e97428c93a16cbb52a41bc4794eb71be3aed436fb9cc547d9eeebb20f3940fb2 \
  --epochs 1 \
  --learning-rate 0.01 \
  --weight-decay 0.0001 \
  --batch-size 2 \
  --adapter-mode channel_lift_task_modulated \
  --rollout-loss-steps 0 \
  --rollout-loss-weight 1.0 \
  --seed 17 \
  --tasks advection1d burgers1d darcy2d
```

Summary:

`reports/research/sota_loop/external_baselines/poseidon_scot_task_modulated_channel_lift_smoke_val_light_v1/summary.json`

Contract checks:

- `status = validation_finetune_measurement_complete`
- `train_split = train`
- `split = val`
- `held_out_test_used = false`
- `held_out_test_data_read = false`
- `claim_comparable = false`
- `published_numbers_directly_comparable = false`
- `adapter_mode = channel_lift_task_modulated`
- `details.model.embedding_recovery_replaced = false`
- `details.contract.pretrained_embedding_recovery_intact = true`
- trainable parameters: `43`
- Poseidon source commit:
  `b8fa28f59bd7f7673323f28d11a12c6f3a215c61`
- checkpoint SHA256:
  `e97428c93a16cbb52a41bc4794eb71be3aed436fb9cc547d9eeebb20f3940fb2`
- summary validator: `validate_poseidon_finetune_summary(summary) == []`

Metrics:

| Metric | Value |
| --- | ---: |
| Aggregate decoded rollout NRMSE | `0.3422384139670503` |
| Advection decoded rollout NRMSE | `0.3623806561813393` |
| Burgers decoded rollout NRMSE | `0.22147624438612523` |
| Darcy decoded rollout NRMSE | `0.7516511659226303` |

This smoke proves the import/checkpoint/task-conditioned adapter path works.
It is not a gate result because it used only two train/eval samples, four
rollout steps, one CPU epoch, and no rollout loss. The weak Darcy smoke number
is a warning that the full validation gate must protect every task, not only
aggregate or advection.

## Validation-Only GPU Plan

Run only if provider work remains allowed and the run can complete without a
credential change, Vast top-up, unknown billing, held-out test access, or a
new public/claim-evidence action.

Command shape:

```bash
python scripts/run_external_poseidon_scot_finetune.py \
  --config configs/train_multitask_heterogeneous_light_best.yaml \
  --name poseidon_scot_task_modulated_channel_lift_val_light_v1_e30_lr1e2_roll4 \
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
  --adapter-mode channel_lift_task_modulated \
  --rollout-loss-steps 4 \
  --rollout-loss-weight 1.0
```

Expected summary path:

```text
reports/research/sota_loop/external_baselines/poseidon_scot_task_modulated_channel_lift_val_light_v1_e30_lr1e2_roll4/summary.json
```

## Gates

The GPU validation run is worth considering only if all contract checks pass:

- `split = val`
- `held_out_test_used = false`
- `held_out_test_data_read = false`
- `details.adapter_mode = channel_lift_task_modulated`
- `details.model.embedding_recovery_replaced = false`
- `details.contract.pretrained_embedding_recovery_intact = true`
- `details.task_modulation.task_to_index` records all task IDs.
- trainable parameter count is `43`.

Decision gates:

- Aggregate decoded rollout NRMSE must be `<= 0.363424243629033`.
- Advection decoded rollout NRMSE must be `<= 0.4866576789288726`.
- Burgers should not materially regress from Option A validation
  `0.15674926288225416`.
- Darcy should not materially regress from Option A validation
  `0.2071060212271272`.
- No task may approach collapse near `1.0`.

Passing these gates would authorize only a new held-out pretest contract and
evidence manifest. It would not authorize a held-out run in the same step and
would not authorize public claim changes.

## Stop Conditions

Stop before or during execution if:

- the command would read `split=test`;
- a held-out ledger write is requested;
- Poseidon source or checkpoint provenance does not match;
- the summary validator reports errors;
- provider setup requires a credential change, top-up, or broader external
  mutation than the bounded validation run;
- aggregate or per-task validation metrics miss the gates above.

If Option B misses the strict validation gate, do not run held-out. Return to a
no-provider branch decision between DPOT mechanism design and UPS-side
transport/refiner work.

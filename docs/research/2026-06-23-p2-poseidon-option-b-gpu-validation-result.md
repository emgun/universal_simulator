# P2 Poseidon Option B Validation-Only GPU Result

Date: 2026-06-23

Status: completed validation-only GPU measurement. The run cleared the
aggregate G2a threshold but missed the strict advection/transport gate. No
held-out test was used, no claim evidence changed, and no public language
changed.

## Run

- Vast instance: `42284571`
- Vast offer: `32941004`, RTX 4090, resolved at `$0.3347222222222222/hr`
  before launch.
- Instance status after run: destroyed manually after completion because the
  remote container could not power itself off cleanly.
- Git ref: `codex/poseidon-channel-lift-vast`
- Commit containing Option B runner and remote-wrapper support: `7f1a8fd`
- Runner: `scripts/run_external_poseidon_scot_finetune.py`
- Remote wrapper: `scripts/run_remote_poseidon_channel_lift.sh`
- Launcher: `scripts/launch_remote_poseidon_channel_lift_vast.sh`
- Source: `https://github.com/camlab-ethz/poseidon`
- Source commit: `b8fa28f59bd7f7673323f28d11a12c6f3a215c61`
- Checkpoint: `camlab-ethz/Poseidon-T`, `model.safetensors`
- Checkpoint SHA256:
  `e97428c93a16cbb52a41bc4794eb71be3aed436fb9cc547d9eeebb20f3940fb2`
- B2 artifact:
  `b2://pdebench/remote-runs/poseidon-channel-lift/poseidon_channel_lift_light-v1_20260623T235710Z.tar.gz`
- Local summary:
  `reports/research/sota_loop/external_baselines/poseidon_scot_task_modulated_channel_lift_val_light_v1_e30_lr1e2_roll4/summary.json`

Command shape:

```bash
DRY_RUN=0 \
GIT_REF=codex/poseidon-channel-lift-vast \
EXTRA_PIPELINE_ARGS='RUN_NAME=poseidon_scot_task_modulated_channel_lift_val_light_v1_e30_lr1e2_roll4 ADAPTER_MODE=channel_lift_task_modulated EXPECTED_TRAINABLE_PARAMETERS=43 ADVECTION_NRMSE_GATE=0.4866576789288726 BURGERS_NRMSE_GATE=0.15674926288225416 DARCY_NRMSE_GATE=0.2071060212271272' \
bash scripts/launch_remote_poseidon_channel_lift_vast.sh
```

The remote wrapper ran:

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
  --epochs 30 \
  --learning-rate 0.01 \
  --weight-decay 0.0001 \
  --batch-size 32 \
  --grad-clip-norm 1.0 \
  --adapter-mode channel_lift_task_modulated \
  --rollout-loss-steps 4 \
  --rollout-loss-weight 1.0 \
  --seed 17 \
  --tasks advection1d burgers1d darcy2d
```

## Contract Checks

- `train_split = train`
- `split = val`
- `held_out_test_used = false`
- `held_out_test_data_read = false`
- `claim_comparable = false`
- `published_numbers_directly_comparable = false`
- `adapter_mode = channel_lift_task_modulated`
- trainable parameters: `43`
- `details.model.embedding_recovery_replaced = false`
- `details.contract.pretrained_embedding_recovery_intact = true`
- `details.contract.task_modulated_channel_lift = true`
- `details.task_modulation.task_to_index` recorded all task IDs
- source commit matched
- checkpoint SHA256 matched
- local summary validator: `validate_poseidon_finetune_summary(summary) == []`

## Metrics

| Metric | Value | Gate | Status |
| --- | ---: | ---: | --- |
| Aggregate decoded rollout NRMSE | `0.3566052737393018` | `<= 0.363424243629033` | pass |
| Advection decoded rollout NRMSE | `0.4967493071208899` | `<= 0.4866576789288726` | miss |
| Burgers decoded rollout NRMSE | `0.14460934384484475` | `<= 0.15674926288225416` | pass |
| Darcy decoded rollout NRMSE | `0.18262014873452226` | `<= 0.2071060212271272` | pass |

Compared with Poseidon Option A validation:

- Aggregate improved slightly: `0.35782889238675264` -> `0.3566052737393018`.
- Burgers improved: `0.15674926288225416` -> `0.14460934384484475`.
- Darcy improved: `0.2071060212271272` -> `0.18262014873452226`.
- Advection regressed slightly: `0.4937043430599529` -> `0.4967493071208899`.

The wrapper printed:

```text
SUMMARY_VALIDATION_OK split=val adapter=channel_lift_task_modulated decoded_rollout_nrmse=0.3566052737393018 gate=0.363424243629033 decision=cleared_g2a advection1d=0.4967493071208899 gate=0.4866576789288726 miss burgers1d=0.14460934384484475 gate=0.15674926288225416 pass darcy2d=0.18262014873452226 gate=0.2071060212271272 pass
```

## Decision

Do not move Poseidon Option B to held-out pretest. It clears aggregate G2a and
improves Burgers/Darcy, but the current roadmap gate was intentionally stricter
after the Option A held-out failure: advection/transport must be protected
before any new held-out access. Option B misses that gate and is therefore a
mixed validation result, not a promotion result.

The useful learning is precise: small task modulation helps the non-transport
families and aggregate score, but it does not repair the transport failure
mode. The next move should not be another held-out pretest or an aggregate-only
Poseidon tweak. Run a no-provider branch check that compares:

- a targeted transport-aware Poseidon adapter or temporal modulation;
- DPOT escalation only if there is a specific transport mechanism and kill
  gate;
- returning to UPS-side transport/refiner work with validation pressure that
  explicitly catches advection drift.

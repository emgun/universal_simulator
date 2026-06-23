# P2 DPOT Validation-Only GPU Plan

Date: 2026-06-23

Status: prepared plan. No GPU/provider work ran, no held-out test was used, no
claim evidence changed, and no public language changed.

## Purpose

The DPOT Tiny CPU smoke passed the import, checkpoint, adapter, split, and
schema gate. The next useful question is whether the same DPOT
`channel_lift` adapter can clear a real train/validation signal on `light-v1`
without reading held-out test.

This plan is a validation-only gate, not a public claim path.

## Inputs

- DPOT source: `https://github.com/HaoZhongkai/DPOT`
- Source commit: `dcd2f9a9359765e19ad63e2f3f879a2a8ce1aa17`
- Checkpoint: `model_Ti.pth`
- Checkpoint SHA256:
  `074c337f9b3a3c70253f8022ce6be7e7dfb809a91a7b00e46fbfedf9611d767f`
- Runner: `scripts/run_external_dpot_finetune.py`
- Remote wrapper: `scripts/run_remote_dpot_channel_lift.sh`
- Vast launcher: `scripts/launch_remote_dpot_channel_lift_vast.sh`
- Train split: `train`
- Eval split: `val`
- Tasks: `advection1d burgers1d darcy2d`
- Held-out split: forbidden for this plan.

## Proposed Command

Use one bounded Vast GPU instance only if provider work is allowed and the
instance already has or can receive the pinned repo, local data, and Tiny
checkpoint without secrets beyond existing project access.

Dry-run first:

```bash
DRY_RUN=1 GIT_REF=codex/poseidon-channel-lift-vast \
  bash scripts/launch_remote_dpot_channel_lift_vast.sh
```

```bash
python scripts/run_external_dpot_finetune.py \
  --config configs/train_multitask_heterogeneous_light_best.yaml \
  --name dpot_tiny_channel_lift_val_light_v1_e30_lr1e2_roll4 \
  --output-root reports/research/sota_loop/external_baselines \
  --train-split train \
  --eval-split val \
  --max-train-samples 0 \
  --max-eval-samples 0 \
  --rollout-steps 4 \
  --dpot-repo /tmp/dpot-official \
  --dpot-source-commit dcd2f9a9359765e19ad63e2f3f879a2a8ce1aa17 \
  --checkpoint-file model_Ti.pth \
  --expected-checkpoint-sha256 074c337f9b3a3c70253f8022ce6be7e7dfb809a91a7b00e46fbfedf9611d767f \
  --device cuda \
  --data-root data/pdebench \
  --tasks advection1d burgers1d darcy2d \
  --epochs 30 \
  --learning-rate 0.01 \
  --weight-decay 0.0001 \
  --batch-size 8 \
  --adapter-mode channel_lift \
  --history-steps 10 \
  --history-init repeat_current
```

For this DPOT runner, `--max-train-samples 0` and `--max-eval-samples 0` mean
uncapped/full split; targeted unit coverage protects this convention.

## Validation Gate

Use a strict validation-only gate:

- Aggregate decoded rollout NRMSE:
  `<= 0.363424243629033`
- Advection decoded rollout NRMSE:
  `<= 0.4866576789288726`
- Darcy decoded rollout NRMSE:
  must improve materially over the CPU smoke and must not approach collapse.
- Burgers decoded rollout NRMSE:
  must not regress toward collapse.
- No task decoded rollout NRMSE may approach `1.0`.
- `held_out_test_used` and `held_out_test_data_read` must both be `false`.

The aggregate threshold preserves the prior Poseidon validation gate. The
per-task constraints are necessary because the Poseidon held-out failure showed
that aggregate validation can hide transport weakness, and the DPOT smoke
showed Darcy can be the weak task under this adapter.

## Stop Conditions

Stop before running if:

- held-out test would be read;
- checkpoint SHA256 does not match;
- DPOT source commit does not match;
- GPU/provider work would require new paid spend, top-up, unknown billing, or
  credentials beyond existing project scope;
- the runner summary validator reports errors.

Stop after the first completed validation summary. Do not run a held-out
pretest or public claim update from this plan alone.

## Required Artifacts

- Local summary JSON under
  `reports/research/sota_loop/external_baselines/dpot_tiny_channel_lift_val_light_v1_e30_lr1e2_roll4/summary.json`
- Remote run archive if executed on Vast/B2.
- A short result note under `docs/research/` with source commit, checkpoint
  SHA256, command, metrics, artifact paths, and gate decision.
- Updates to `docs/current-state.md` and `docs/experiments/ledger.md`.

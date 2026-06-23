# P2 DPOT CPU Smoke Result

Date: 2026-06-23

Status: passed validation-only CPU/import/checkpoint smoke. No GPU/provider
work ran, no held-out test was used, no claim evidence changed, and no public
language changed.

## Command

```bash
python scripts/run_external_dpot_finetune.py \
  --config configs/train_multitask_heterogeneous_light_best.yaml \
  --name dpot_tiny_channel_lift_smoke_val_light_v1 \
  --output-root reports/research/sota_loop/external_baselines \
  --train-split train \
  --eval-split val \
  --max-train-samples 2 \
  --max-eval-samples 2 \
  --rollout-steps 4 \
  --dpot-repo /tmp/dpot-official \
  --dpot-source-commit dcd2f9a9359765e19ad63e2f3f879a2a8ce1aa17 \
  --checkpoint-file model_Ti.pth \
  --expected-checkpoint-sha256 074c337f9b3a3c70253f8022ce6be7e7dfb809a91a7b00e46fbfedf9611d767f \
  --device cpu \
  --data-root data/pdebench \
  --tasks advection1d burgers1d darcy2d \
  --epochs 1 \
  --learning-rate 0.01 \
  --weight-decay 0.0001 \
  --batch-size 2 \
  --adapter-mode channel_lift \
  --history-steps 10 \
  --history-init repeat_current
```

## Evidence

- DPOT source: `/tmp/dpot-official`
- Source commit: `dcd2f9a9359765e19ad63e2f3f879a2a8ce1aa17`
- Import check: `from models.dpot import DPOTNet` succeeded.
- Checkpoint: `/tmp/dpot-official/model_Ti.pth`
- Checkpoint SHA256:
  `074c337f9b3a3c70253f8022ce6be7e7dfb809a91a7b00e46fbfedf9611d767f`
- Summary:
  `reports/research/sota_loop/external_baselines/dpot_tiny_channel_lift_smoke_val_light_v1/summary.json`
- Summary validator: `validate_dpot_finetune_summary(summary) == []`

## Result

The smoke completed with status
`validation_finetune_measurement_complete`.

| Metric | Value |
| --- | ---: |
| Aggregate decoded rollout NRMSE | `0.4056234877403711` |
| Advection decoded rollout NRMSE | `0.46377715332535735` |
| Burgers decoded rollout NRMSE | `0.31818835051545247` |
| Darcy decoded rollout NRMSE | `0.5280264566767598` |

Contract checks:

- `train_split = train`
- `split = val`
- `held_out_test_used = false`
- `held_out_test_data_read = false`
- `claim_comparable = false`
- `published_numbers_directly_comparable = false`
- `adapter_mode = channel_lift`
- `history_steps = 10`
- `history_init = repeat_current`
- trainable parameters: `13`
- training pairs: `24`
- evaluation pairs: `8` per task

## Interpretation

This is a mechanics smoke, not a model-quality gate. It proves that the pinned
DPOT source imports, the Tiny checkpoint loads under the current PyTorch
runtime after SHA verification, the scalar channel-lift adapter trains, and
the validation-only artifact schema is usable.

The smoke does not establish DPOT as better than the current UPS or Poseidon
branches. With only two train samples, two validation samples, four rollout
steps, and one CPU epoch, the aggregate NRMSE is not directly comparable to the
full validation gates. The useful signal is that advection did not immediately
collapse in this tiny probe, while Darcy is weak enough that any larger DPOT
validation plan should protect all per-task metrics, not just advection.

## Decision

Proceed to a bounded DPOT validation-only GPU plan if provider work is allowed.
Do not run held-out test. Do not update claim evidence or public language.

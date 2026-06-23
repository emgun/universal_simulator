# P2 DPOT Validation-Only GPU Result

Date: 2026-06-23

Status: completed validation-only GPU measurement. The run missed the DPOT
validation gate. No held-out test was used, no claim evidence changed, and no
public language changed.

## Run

- Vast instance: `42278129`
- Instance status after run: destroyed manually after completion because the
  remote container could not power itself off cleanly.
- Git ref: `codex/poseidon-channel-lift-vast`
- Commit containing launcher/runtime patch: `321b686`
- Runner: `scripts/run_external_dpot_finetune.py`
- Remote wrapper: `scripts/run_remote_dpot_channel_lift.sh`
- Launcher: `scripts/launch_remote_dpot_channel_lift_vast.sh`
- Source: `https://github.com/HaoZhongkai/DPOT`
- Source commit: `dcd2f9a9359765e19ad63e2f3f879a2a8ce1aa17`
- Checkpoint: `model_Ti.pth`
- Checkpoint SHA256:
  `074c337f9b3a3c70253f8022ce6be7e7dfb809a91a7b00e46fbfedf9611d767f`
- B2 artifact:
  `b2://pdebench/remote-runs/dpot-channel-lift/dpot_channel_lift_light-v1_20260623T221057Z.tar.gz`
- Local summary:
  `reports/research/sota_loop/external_baselines/dpot_tiny_channel_lift_val_light_v1_e30_lr1e2_roll4/summary.json`

An earlier Vast instance, `42277502`, failed before training because the
remote PyTorch 2.2 image did not expose `torch.serialization.safe_globals`.
That instance was destroyed. The runner now falls back for older PyTorch after
checkpoint SHA verification, and the launcher default image was moved to
`pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime`.

## Contract Checks

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
- source commit matched
- checkpoint SHA256 matched
- summary validator: `validate_dpot_finetune_summary(summary) == []`

## Metrics

| Metric | Value | Gate |
| --- | ---: | ---: |
| Aggregate decoded rollout NRMSE | `0.7136888249949349` | `<= 0.363424243629033` |
| Advection decoded rollout NRMSE | `0.8575561454613253` | `<= 0.4866576789288726` |
| Burgers decoded rollout NRMSE | `0.588255711789389` | no collapse/regression |
| Darcy decoded rollout NRMSE | `0.28923145953251056` | no collapse/regression |

The run used `1536` training pairs. Evaluation used `32` validation samples per
task and `128` evaluated rollout pairs per task.

## Decision

DPOT Tiny `channel_lift` should not proceed to held-out pretest. It misses the
aggregate validation gate and repeats the main failure mode of the Poseidon
held-out result: transport/advection generalization is too weak. Burgers is
also materially worse than the Poseidon validation result. Darcy improved
relative to the 2-sample CPU smoke, but that does not offset the aggregate and
transport miss.

The next useful move is a no-held-out branch check. Compare:

- DPOT escalation beyond Tiny/channel-lift, such as Small checkpoint or a
  task-conditioned adapter;
- Poseidon Option B/task modulation with an advection-aware validation gate;
- returning to the UPS learned-operator/refiner roadmap with transport-specific
  validation pressure.

Do not run another DPOT GPU experiment until that branch check states the
specific hypothesis, extra cost, expected evidence, and kill condition.

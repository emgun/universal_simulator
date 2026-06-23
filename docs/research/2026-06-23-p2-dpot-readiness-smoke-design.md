# P2 DPOT Readiness And Smoke Design

Date: 2026-06-23

Status: readiness/design note. No DPOT source was cloned, no checkpoint was
downloaded, no GPU/provider work ran, no held-out test was used, no claim
evidence changed, and no public language changed.

## Trigger

Poseidon ScOT `channel_lift` Option A cleared aggregate validation but failed
the single ledger-protected held-out pre-test, dominated by advection:

- Validation aggregate decoded rollout NRMSE: `0.35782889238675264`
- Validation advection decoded rollout NRMSE: `0.4937043430599529`
- Held-out aggregate decoded rollout NRMSE: `0.5551415687535287`
- Held-out advection decoded rollout NRMSE: `0.7840223655431167`

The post-held-out branch check selected DPOT readiness as the next primary
validation-only branch:
`docs/research/2026-06-23-p2-post-heldout-branch-check.md`.

## Live Source And Checkpoint Candidate

Primary source candidate:

- Repository: `https://github.com/HaoZhongkai/DPOT`
- Main branch HEAD checked on 2026-06-23:
  `dcd2f9a9359765e19ad63e2f3f879a2a8ce1aa17`
- Relevant model file: `models/dpot.py`
- README loader example constructs `DPOTNet` with `img_size=128`,
  `patch_size=8`, `mixing_type='afno'`, `in_channels=4`,
  `in_timesteps=10`, `out_timesteps=1`, `out_channels=4`,
  `embed_dim=512`, `modes=32`, `depth=4`, `n_blocks=4`, `mlp_ratio=1`,
  `out_layer_dim=32`, `n_cls=12`, then loads `model_Ti.pth`.

Primary checkpoint candidate:

- Hugging Face model repo: `hzk17/DPOT`
- Repo SHA checked on 2026-06-23:
  `2adec1cf9a55942f1456aa7463cd7ade908398d0`
- Use Tiny first: `model_Ti.pth`
- Tiny checkpoint LFS SHA256:
  `074c337f9b3a3c70253f8022ce6be7e7dfb809a91a7b00e46fbfedf9611d767f`
- Tiny checkpoint size: `90475962` bytes

Other HF checkpoint files exist (`model_S.pth`, `model_M.pth`,
`model_L.pth`, `model_H.pth`) but are much larger. Do not start with them.
The Tiny checkpoint is the correct readiness target because the first goal is
source/import/schema proof, not best validation score.

## Tensor Contract For `light-v1`

DPOT Tiny expects a 4-channel, 10-input-timestep, 1-output-timestep, 128x128
field contract. `light-v1` evaluation currently uses scalar fields and 16-step
decoded rollout over three tasks (`advection1d`, `burgers1d`, `darcy2d`).

Adapter contract:

- Convert each scalar field step to 128x128 pixels using the same
  light-to-pixel path already exercised by the Poseidon adapter.
- Build a 10-frame input window for DPOT. For the first CPU smoke, initialize
  missing history by repeating the current scalar frame rather than reading
  future frames.
- Lift scalar frames from `1` channel to DPOT native `4` channels with a
  trainable 1x1 channel lift initialized to replicate.
- Keep the DPOT backbone frozen for the first smoke.
- Read out DPOT's 4-channel one-step prediction to scalar with a trainable
  1x1 readout initialized to channel mean.
- Roll out autoregressively in scalar space: scalar prediction -> append to the
  10-frame history -> lift -> frozen DPOT -> readout.

This mirrors the Poseidon channel-lift principle while respecting DPOT's
temporal input contract. It also directly tests the failure mode exposed by the
Poseidon held-out miss: long-horizon transport/advection stability.

## Frozen And Trainable Parameters

First smoke:

- Frozen: all DPOT pretrained parameters.
- Trainable: scalar-to-4-channel lift and 4-channel-to-scalar readout only.
- Expected trainable count if implemented as per-frame 1x1 lift/readout:
  `13` parameters, matching the Poseidon Option A adapter.
- Optional later extension, validation-only: task-conditioned channel
  gains/biases or a shallow nonlinear temporal adapter. Do not include this in
  the first readiness smoke.

The smoke should record full parameter names, trainable/frozen counts, source
commit, checkpoint SHA256, and adapter initialization.

## Local CPU/Import Smoke

Before any GPU run, implement a minimal DPOT runner that can execute this
readiness command locally:

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

Pre-smoke local checks:

```bash
git clone https://github.com/HaoZhongkai/DPOT /tmp/dpot-official
git -C /tmp/dpot-official checkout dcd2f9a9359765e19ad63e2f3f879a2a8ce1aa17
python - <<'PY'
import sys
sys.path.insert(0, "/tmp/dpot-official")
from models.dpot import DPOTNet
print(DPOTNet.__name__)
PY
```

Checkpoint download is intentionally not part of this heartbeat tick. When the
runner exists, the smoke may download only `model_Ti.pth`, verify the SHA256,
and stop before any larger checkpoint or GPU work.

## Artifact And Schema Expectations

Expected summary path:

`reports/research/sota_loop/external_baselines/dpot_tiny_channel_lift_smoke_val_light_v1/summary.json`

Minimum summary fields:

- `schema_version = 1`
- `status = "validation_finetune_measurement_complete"` or an explicit
  fail-closed invalid status with validation errors
- `measurement_type = "dpot_finetune_validation_measurement"`
- `train_split = "train"`
- `split = "val"`
- `held_out_test_used = false`
- `held_out_test_data_read = false`
- `claim_comparable = false`
- `published_numbers_directly_comparable = false`
- `details.dpot_source.commit = "dcd2f9a9359765e19ad63e2f3f879a2a8ce1aa17"`
- `details.pretrained_checkpoint.sha256 =
  "074c337f9b3a3c70253f8022ce6be7e7dfb809a91a7b00e46fbfedf9611d767f"`
- `details.adapter_mode = "channel_lift"`
- `details.history_steps = 10`
- `details.history_init = "repeat_current"`
- `details.trainable_parameters.trainable_parameter_count = 13`
- `metrics.decoded_rollout_nrmse`
- per-task decoded rollout metrics for `advection1d`, `burgers1d`, and
  `darcy2d`

Add unit tests before running the real smoke:

- CLI blocks `--eval-split test` unless a future explicit held-out gate exists.
- Summary validator rejects missing checkpoint hash.
- Summary validator rejects held-out flags on validation summaries.
- Channel lift/readout initializes to replicate/mean.
- Frozen-backbone assertion catches accidental DPOT parameter unfreezing.
- History-window builder is deterministic and does not read future frames.

## Validation Gate For Any Later GPU Run

The Poseidon failure showed that aggregate validation alone is too permissive.
Any DPOT validation GPU plan must use a stricter gate:

- Aggregate validation:
  `decoded_rollout_nrmse <= 0.363424243629033`
- Advection validation:
  `task_advection1d_decoded_rollout_nrmse <= 0.4866576789288726`
- Prefer h16/advection horizon protection if exposed by the runner:
  `task_advection1d_decoded_h16_nrmse <= 0.44444171136384397`
- No task decoded rollout NRMSE may approach collapse near `1.0`.
- Held-out test remains forbidden until a separate pre-test contract exists.

The CPU smoke itself is not expected to clear this gate because it uses only
two samples and four rollout steps. It only proves import, checkpoint hash,
adapter mechanics, split discipline, and artifact schema.

## Decision

Proceed to DPOT runner implementation only up to the 2-sample validation smoke.
Do not run GPU yet. Do not download larger checkpoints. Do not run held-out
test. Do not update claim evidence or public docs.

If the Tiny import/checkpoint/schema path fails, record the failure class before
deciding whether to repair DPOT, return to Poseidon Option B/task modulation, or
move back to the Phase 1 learned-operator roadmap.

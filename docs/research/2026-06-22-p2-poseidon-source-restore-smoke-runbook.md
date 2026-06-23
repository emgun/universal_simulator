# P2 Poseidon Source Restore And CPU Smoke Runbook

Date: 2026-06-22

Status: prepared runbook only. No network restore, source clone, GPU run,
held-out test, credential use, or claim evidence change was performed.

## Purpose

Unblock the P2.2 Poseidon `channel_lift` CPU smoke after explicit approval to
restore the official Poseidon source checkout. This is the smallest useful next
step before any GPU validation spend.

## Approval Scope Needed

Recommended approval:

- Restore or clone official Poseidon source from `https://github.com/camlab-ethz/poseidon`
  into `/tmp/poseidon-official`.
- Pin checkout to commit `b8fa28f59bd7f7673323f28d11a12c6f3a215c61`, matching
  prior recorded evidence.
- Do not run GPU jobs, held-out tests, external services, or credentialed
  commands.
- Run only local import verification and a train/validation CPU smoke with
  `max_train_samples=2`, `max_eval_samples=2`.

## Source Restore Commands

Run only after approval:

```bash
rm -rf /tmp/poseidon-official
git clone https://github.com/camlab-ethz/poseidon /tmp/poseidon-official
git -C /tmp/poseidon-official checkout b8fa28f59bd7f7673323f28d11a12c6f3a215c61
```

Verify source identity:

```bash
git -C /tmp/poseidon-official rev-parse HEAD
test -f /tmp/poseidon-official/scOT/model.py
test -f /tmp/poseidon-official/scOT/train.py
test -f /tmp/poseidon-official/scOT/inference.py
test -f /tmp/poseidon-official/scOT/problems/base.py
python - <<'PY'
import sys
sys.path.insert(0, "/tmp/poseidon-official")
from scOT.model import ScOT, ScOTConfig
print("scOT_import=ok")
print(ScOT, ScOTConfig)
PY
```

Expected commit:

```text
b8fa28f59bd7f7673323f28d11a12c6f3a215c61
```

## CPU Smoke Command

This command is validation-only and uses no held-out test split:

```bash
python scripts/run_external_poseidon_scot_finetune.py \
  --config configs/train_multitask_heterogeneous_light_best.yaml \
  --name poseidon_scot_channel_lift_smoke_val_light_v1 \
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
  --tasks advection1d burgers1d darcy2d \
  --epochs 1 \
  --learning-rate 0.01 \
  --weight-decay 0.0001 \
  --batch-size 2 \
  --grad-clip-norm 1.0 \
  --adapter-mode channel_lift \
  --rollout-loss-steps 2 \
  --rollout-loss-weight 1.0
```

Expected output path:

```text
reports/research/sota_loop/external_baselines/poseidon_scot_channel_lift_smoke_val_light_v1/summary.json
```

## Smoke Acceptance Criteria

The smoke passes if:

- The command exits successfully.
- `summary.json` has `status = "validation_finetune_measurement_complete"`.
- `split = "val"`.
- `held_out_test_used = false`.
- `details.adapter_mode = "channel_lift"`.
- `details.model.embedding_recovery_replaced = false`.
- `details.contract.pretrained_embedding_recovery_intact = true`.
- `details.trainable_parameters.trainable_parameter_count` is positive and
  small for the adapter path.
- `metrics.decoded_rollout_nrmse` is finite.

This smoke is not a gate result and should not be used as claim evidence. Its
purpose is to prove the source import, channel-lift wrapper, checkpoint hash,
data path, and validation-only summary contract work together before any GPU
run.

## Next Decision After Smoke

If the CPU smoke passes, request a separate bounded approval for the P2.2
validation run:

- train/validation only;
- no held-out test;
- `channel_lift` Option A first;
- GPU budget under the estimate in
  `docs/research/2026-06-11-p2-poseidon-adapter-design.md`;
- stop if validation is above the roadmap stop threshold after clean Option A/B.

If the CPU smoke fails, record the exact failure and decide whether it is a
source/API mismatch, a runner bug, or an adapter design issue before requesting
GPU spend.


# Public Results Rollout Preview Artifact Support Plan

Goal: make qualitative rollout previews repeatable and claim-linked without
promoting ignored local `reports/` files or latent-space debug previews as public
evidence.

## Scope

- Add an optional manifest path for compact rollout preview artifacts.
- Validate artifact hash, required arrays, and shape conventions before rendering.
- Render a qualitative target/prediction/error panel only when the manifest and
  artifact exist.
- Keep public figures gated until a real decoded preview artifact is
  committed.

## Implementation

- [x] Extend `scripts/build_public_assets.py` with optional rollout preview
  manifest validation.
- [x] Add conditional `generated/rollout_preview_summary.tsv` and
  `generated/rollout_preview_panel.png` outputs for valid artifacts.
- [x] Add unit coverage for present-manifest status and panel generation.
- [x] Update public docs with the manifest path and conditional generated files.
- [x] Attempt artifact generation from the current checkout and record the
  blocker without promoting invalid preview evidence.
- [x] Produce a real decoded `light-v1` preview artifact from a current
  claim-linked run.

## Claim Boundary

The renderer support does not create a new qualitative claim by itself. It only
defines the repeatable path. The next artifact must record command, split,
metric, source summary, artifact SHA-256, and whether the preview is
validation-only or authorized held-out evidence. The committed preview generated
on 2026-06-21 is validation-only.

## 2026-06-21 Artifact Generation Attempt

Outcome: resolved through the ignored repo `.env` and the already-published
small `light-v1` B2 validation shard. The final artifact is validation-only and
does not use held-out test data.

Live checkout findings:

- The only local HDF5 source under `data/` is
  `data/pdebench/burgers1d_train.h5`, a tiny local Burgers training shard. The
  advection validation shard was hydrated from B2 into ignored `data/` for the
  export.
- No local validation HDF5 shards were present under `data/` before B2
  hydration.
- `reports/evaluation_preview.npz` remains excluded because it is an old
  latent-space debug preview, not a decoded physical-space rollout tied to the
  current claim evidence.
- Existing committed summary JSON files may include `details.preview_predicted`
  and `details.preview_target`, but those arrays come from latent evaluation
  summaries and are not acceptable inputs for the public decoded rollout
  contract.
- The committed primary evidence tar
  `docs/claim_evidence/artifacts/ups_light_shared_context_transport_guarded_57108bc.tar.gz`
  contains summaries, resolved configs, and checkpoints. It was extracted into
  `/tmp` and used as the checkpoint source for the preview export.
- The smallest official validation-only hydration plan found one required
  official advection train file with an estimated download size of 7.67 GiB.
  The local volume had only about 1.94 GiB free, so local hydration is blocked
  before download. That large official hydration path remains separate from the
  committed compact `light-v1` preview.
- The inherited shell process did not export B2/W&B keys, but ignored `.env`
  contained `B2_KEY_ID`, `B2_APP_KEY`, `B2_BUCKET`, `B2_S3_ENDPOINT`,
  `B2_S3_REGION`, `WANDB_API_KEY`, `WANDB_PROJECT`, and `WANDB_ENTITY`.
- B2 `light-v1/advection1d/advection1d_val.h5` existed remotely and was small
  enough for local hydration.

Committed outputs:

- `docs/claim_evidence/artifacts/rollout_preview_ups_light_shared_context_transport_guarded_advection1d_val.npz`
- `docs/claim_evidence/artifacts/rollout_preview_ups_light_shared_context_transport_guarded_advection1d_val_summary.json`
- `docs/claim_evidence/rollout_preview_manifest.json`
- `docs/results/generated/rollout_preview_summary.tsv`
- `docs/results/generated/rollout_preview_panel.png`

Repeatable validation-only preview path:

```bash
CLEAN_OLD_SPLITS=0 \
B2_ENV_FILE=.env \
B2_PREFIX=light-v1 \
DATA_ROOT=data/pdebench \
scripts/fetch_datasets_b2.sh advection1d/advection1d_val

rm -rf /tmp/ups_rollout_preview_checkpoint
mkdir -p /tmp/ups_rollout_preview_checkpoint
tar -xzf \
  docs/claim_evidence/artifacts/ups_light_shared_context_transport_guarded_57108bc.tar.gz \
  -C /tmp/ups_rollout_preview_checkpoint
```

Then export the compact preview:

```bash
python scripts/export_rollout_preview_artifact.py \
  --config configs/train_multitask_heterogeneous_light_best.yaml \
  --checkpoint-source \
    /tmp/ups_rollout_preview_checkpoint/ups_light_shared_context_transport_guarded \
  --run-name ups_light_shared_context_transport_guarded \
  --task advection1d \
  --data-task burgers1d \
  --data-task advection1d \
  --data-task darcy2d \
  --skip-missing-tasks \
  --split val \
  --data-root data/pdebench \
  --max-samples 1 \
  --rollout-steps 16 \
  --device cpu \
  --override 'operator.conditioning.sources={"task_id":3,"equation_signature":15}' \
  --eval-override 'evaluation.decoded_persistence_residual_alpha=0.0' \
  --eval-override 'evaluation.decoded_context_roll_shift_estimator={candidate_shifts: [-4, -2, -1, 0, 1, 2, 4, 8, 16, 24, 32, 40, 48, 64], context_transitions: 8, coefficients: {slope: 0.9974352988185539, intercept: 0.0}, families: [transport, conservation], mode: roll_persistence, calibration_scope: shared_1d_transport}'
```

This path intentionally uses train/validation only. Held-out test access should
remain gated until validation passes and an explicit held-out preview decision
is recorded.

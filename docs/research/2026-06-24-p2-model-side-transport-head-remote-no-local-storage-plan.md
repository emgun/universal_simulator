# P2 Model-Side Transport Head Remote No-Local-Storage Plan

Date: 2026-06-24

Status: remote execution path prepared because local disk is insufficient for
official Advection beta-provenance hydration. No held-out test was run, no
claim evidence changed, and no public language changed.

## Why This Route

The model-side beta transport-head real-shard plan requires official Advection
train files so validation samples carry beta provenance. Local preflight found
only about `5.7 GiB` free on the workspace volume, while the smallest
sequential official file step needs about `9.47 GiB` free and the full selected
download is about `61.34 GiB`. That makes local hydration invalid even with
sequential download-convert-delete.

The safe path is a remote scratch run that keeps the validation boundary:

- fetch only standard `light-v1` validation shards needed for Burgers/Darcy;
- sequentially download official Advection train files, convert sampled rows,
  and delete raw files as it goes;
- build a validation-only full-task beta root;
- restore the small ignored UPS checkpoint archive from B2;
- run CPU-only validation with `model_side_transport_head` enabled;
- validate the summary and publish only small evidence artifacts.

## New Wrappers

- `scripts/run_remote_model_side_transport_head_real_shard.sh`
- `scripts/launch_remote_model_side_transport_head_vast.sh`

The remote runner intentionally does not enable held-out test data, does not
run a post-validation test stage, and does not enable evaluator roll-shift
estimators. It uses the model-side summary validator as the final gate.

## Remote Data And Checkpoint Inputs

Standard validation shards:

```bash
B2_PREFIX=light-v1
STANDARD_DATA_KEYS=burgers1d/burgers1d_val.h5,darcy2d/darcy2d_val.h5
```

Official Advection beta-provenance source:

```bash
HYDRATION_PLAN_JSON=reports/research/sota_loop/official_advection_hydration_plan.json
```

Checkpoint source archive:

```bash
CHECKPOINT_SOURCE=reports/research/sota_loop/learned_capacity_gate/ups_light_local_joint_rollout4_residual_ft_val
CHECKPOINT_SOURCE_B2_KEY=remote-runs/checkpoints/ups_light_task_signature_trained_residual_20260526T1928Z.tar.gz
```

The checkpoint archive was inspected by listing B2 metadata and the tar
manifest only. It is small, about `978202` bytes, and contains the expected UPS
checkpoint files under `ups_light_task_signature_trained_residual/`; the remote
runner relocates that directory to the expected checkpoint-source path.

## Dry-Run Command

```bash
DRY_RUN=1 GIT_REF=codex/poseidon-channel-lift-vast \
  bash scripts/launch_remote_model_side_transport_head_vast.sh
```

Default remote scratch request is `DISK_GB=48`, which is enough for sequential
official hydration plus the small validation artifacts while avoiding a local
download.

## Actual Launch Command

Use only after dry-run review and branch push:

```bash
DRY_RUN=0 GIT_REF=codex/poseidon-channel-lift-vast \
  bash scripts/launch_remote_model_side_transport_head_vast.sh
```

The launcher passes B2 credentials from `.env` to the remote host using the
existing Vast launcher mechanism. The remote script defaults to `DRY_RUN=0`
inside the launched instance, because the launcher dry-run is the local guard.

## Validation Gates

The remote run must produce:

- summary:
  `reports/research/sota_loop/model_side_transport_head_real_shard/ups_light_p2_model_side_beta_transport_head_val/summary.json`;
- validator success from `scripts/validate_model_side_transport_head_summary.py`;
- `held_out_test_used = false`;
- `held_out_test_data_read = false`;
- `extra.model_side_transport_head_metrics.beta_missing_count = 0`;
- no active evaluator roll-shift sidecar;
- aggregate, advection rollout, advection h16, Burgers, and Darcy gates from
  `docs/research/2026-06-24-p2-model-side-transport-head-real-shard-validation-plan.md`.

## Stop Conditions

Stop and record the blocker if:

- B2 checkpoint archive is missing or extraction cannot create the expected
  checkpoint source;
- standard validation shards are missing from `light-v1`;
- official Advection download/convert/delete fails;
- generated full-task root cannot prove beta provenance;
- summary validator fails;
- Vast launch cannot resolve an offer or cannot be torn down;
- any command attempts to read `test`.

No held-out pretest, claim-evidence update, or public-language change is
authorized by this route.

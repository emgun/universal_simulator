# P2 Model-Side Transport Head Remote Result

Date: 2026-06-25

Status: validation-only remote run passed the model-side summary schema gate. No
held-out test was requested or used.

## Run

- Git ref: `codex/poseidon-channel-lift-vast`.
- Launch-state commit: `596daf0`.
- Vast contract: `42450012`.
- Offer: `41175200`, RTX 4090, `48 GB` disk.
- Observed rate: `0.37555555555555553` dollars/hour.
- Remote route: `scripts/run_remote_model_side_transport_head_real_shard.sh`.
- Artifact:
  `b2://pdebench/remote-runs/model-side-transport-head/model_side_transport_head_real_shard_20260625T022059Z.tar.gz`.
- Local downloaded artifact:
  `reports/research/sota_loop/model_side_transport_head_real_shard_remote_artifacts/model_side_transport_head_real_shard_20260625T022059Z.tar.gz`.
- Artifact SHA256:
  `9778317b2942728e0d5e9bd503baadbecd66ee08ef44968e9ed60eb2dff9e905`.
- Summary inside artifact:
  `reports/research/sota_loop/model_side_transport_head_real_shard/ups_light_p2_model_side_beta_transport_head_val/summary.json`.

## Verification

The summary validator passed locally:

```bash
python scripts/validate_model_side_transport_head_summary.py \
  /tmp/model_side_transport_head_real_shard_20260625T022059Z/reports/research/sota_loop/model_side_transport_head_real_shard/ups_light_p2_model_side_beta_transport_head_val/summary.json
```

Validator output:

```json
{
  "errors": [],
  "passed": true
}
```

Additional boundary checks from the artifact:

- `full_task_beta_val_root_manifest.json` has
  `held_out_test_data_read = false` and `test_ledger_writes = []`.
- `official_advection_hydration_plan_run.json` records
  `test_split_downloaded = false` and `test_split_sharded = false`.
- The run used the validation split only; no claim-evidence or public-language
  artifact was updated.
- Vast contract `42450012` was destroyed after completion; no active route
  instance remains.

## Metrics

Decoded rollout NRMSE:

- aggregate: `0.11122069837659315`
- advection rollout: `0.0017868115829009724`
- advection h16: `0.001784282965734058`
- Burgers: `0.14738121133726986`
- Darcy: `0.18897951477635447`

Model-side transport-head telemetry:

- mean shift: `3.5019181072711945`
- applied samples: `512`
- skipped samples: `0`
- beta-missing samples: `0`
- trainable parameters: `2`
- required parameter: `beta`
- mode: `periodic_roll`
- apply point: `decoded_rollout`

## Decision

This is accepted validation evidence for the scoped model-side beta transport
head branch. It is not public claim evidence by itself because the mechanism
depends on beta provenance that is not yet part of the universal public
inference contract.

Next safe move: run a no-provider protocol/evidence mapping review that decides
whether the model-side beta head should advance toward a pre-registered held-out
pretest contract, remain a scoped mechanism, or be adapted to a broader
inference contract. Do not run held-out tests, update claim evidence, or change
public language without that separate mapping review and explicit user
direction.

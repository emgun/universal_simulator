# P2 Model-Side Transport Head Remote Launch

Date: 2026-06-24

Status: remote validation reached summary generation but failed the model-side
summary schema gate; instance destroyed. No held-out test was requested or
authorized by this launch note.

## Scope

This launch follows
`docs/research/2026-06-24-p2-model-side-transport-head-remote-no-local-storage-plan.md`
because local disk is insufficient for official Advection beta-provenance
hydration. The run uses remote scratch, sequential official Advection
hydrate-convert-delete, standard `light-v1` validation shards from B2 for
Burgers/Darcy, and CPU-only `model_side_transport_head` validation.

## Local Verification Before Launch

Commands/checks:

```bash
df -h /Users/emerygunselman/Code/universal_simulator /Users/emerygunselman /tmp
python scripts/preflight_transport_hydration.py --mode sequential \
  --output-json reports/research/sota_loop/official_advection_hydration_preflight.json
python scripts/recommend_transport_hydration_storage.py --mode sequential \
  --output-json reports/research/sota_loop/official_advection_hydration_storage_recommendation.json
bash -n scripts/run_remote_model_side_transport_head_real_shard.sh \
  scripts/launch_remote_model_side_transport_head_vast.sh
python -m pytest tests/unit/test_build_p2_parameter_full_task_root.py \
  tests/unit/test_validate_model_side_transport_head_summary.py -q
git diff --check
DRY_RUN=1 GIT_REF=codex/poseidon-channel-lift-vast \
  bash scripts/launch_remote_model_side_transport_head_vast.sh
DRY_RUN=1 PUBLISH_ARTIFACTS=0 \
  bash scripts/run_remote_model_side_transport_head_real_shard.sh
```

Findings:

- Local free space was about `5.7 GiB`; sequential hydration requires about
  `9.47 GiB` for the largest official raw file plus safety margin.
- B2 checkpoint metadata contained
  `remote-runs/checkpoints/ups_light_task_signature_trained_residual_20260526T1928Z.tar.gz`
  with size `978202` bytes.
- The checkpoint archive tar manifest contains the expected UPS checkpoint
  files under `ups_light_task_signature_trained_residual/`; the remote wrapper
  relocates that directory to
  `reports/research/sota_loop/learned_capacity_gate/ups_light_local_joint_rollout4_residual_ft_val`.
- Focused tests passed with `5 passed`.
- Shell syntax checks and `git diff --check` passed.
- Launcher and runner dry-runs showed no held-out stage, no evaluator sidecar,
  and no local hydration.

## Launch Attempts

First attempt:

- Offer: `38186209`, RTX 4090, quoted at about `$0.3232407407407408/hr`.
- Contract: `42401525`.
- Outcome: created a contract but stayed in `loading` / `stopped` with no
  usable uptime.
- Action: destroyed `42401525` to avoid dead-instance storage charges.

Second attempt:

- Offer: `41528131`, RTX 4090.
- Contract: `42401821`.
- Observed rate from instance status: about `$0.37555555555555553/hr`.
- Disk: `48 GB`.
- Git ref: `codex/poseidon-channel-lift-vast`.
- Remote script:
  `scripts/run_remote_model_side_transport_head_real_shard.sh`.
- Initial status: `running` / `loading`, status message `Download complete`.

Final outcome:

- Sanitized monitor status showed contract `42401821` still running after about
  `183.88` minutes with no files published under
  `b2://pdebench/remote-runs/model-side-transport-head/`.
- A redacted container-log check found the remote wrapper failed before
  official Advection hydration:
  `FileNotFoundError: reports/research/sota_loop/official_advection_hydration_plan.json`.
- The failure was a packaging/dependency bug, not a validation result. The
  hydration plan existed only as an ignored local artifact, so the remote git
  checkout could not read it.
- Contract `42401821` was destroyed after the failure was confirmed; sanitized
  follow-up status returned `not_found_active_instances`.

## Expected Artifact

If successful, the remote wrapper should publish a small artifact under:

```text
b2://pdebench/remote-runs/model-side-transport-head/
```

and include:

- `reports/research/sota_loop/model_side_transport_head_real_shard/ups_light_p2_model_side_beta_transport_head_val/summary.json`;
- `reports/research/sota_loop/model_side_transport_head_real_shard/full_task_beta_val_root_manifest.json`;
- official hydration validation/run JSONs.

Do not publish hydrated raw data.

## Wrapper Fix

The remote wrapper now regenerates a missing hydration plan from tracked repo
sources before the sequential hydrator reads it:

```bash
python scripts/plan_transport_official_hydration.py \
  --output-json reports/research/sota_loop/official_advection_hydration_plan.json
```

Code/test coverage:

- `scripts/run_remote_model_side_transport_head_real_shard.sh`
- `tests/unit/test_run_remote_model_side_transport_head_real_shard.py`

Verification after the fix:

```bash
bash -n scripts/run_remote_model_side_transport_head_real_shard.sh \
  scripts/launch_remote_model_side_transport_head_vast.sh
python -m pytest tests/unit/test_run_remote_model_side_transport_head_real_shard.py \
  tests/unit/test_build_p2_parameter_full_task_root.py \
  tests/unit/test_validate_model_side_transport_head_summary.py -q
git diff --check
```

Result: `7 passed`; shell syntax and diff whitespace checks passed.

## Next Step

## Fixed Relaunch

The wrapper fix was committed and pushed as `fdae16c` on
`codex/poseidon-channel-lift-vast`. A single bounded relaunch of the same
no-held-out remote route was started from that ref:

- Offer: `41528131`, RTX 4090.
- Contract: `42412831`.
- Observed rate from sanitized instance status: about
  `$0.37555555555555553/hr`.
- Disk: `48 GB`.
- Git ref: `codex/poseidon-channel-lift-vast`.
- Remote script:
  `scripts/run_remote_model_side_transport_head_real_shard.sh`.
- Latest sanitized status before this note update: `actual_status = running`,
  `cur_state = running`, `intended_status = running`, status message
  `success, running pytorch/pytorch_2.2.0-cuda12.1-cudnn8-runtime/ssh`.
- B2 result prefix check immediately after launch found no files yet under
  `b2://pdebench/remote-runs/model-side-transport-head/`.

## Next Monitoring Step

Final outcome:

- Sanitized monitor status showed contract `42412831` running after about
  `239.31` minutes with no files published under
  `b2://pdebench/remote-runs/model-side-transport-head/`.
- A redacted log tail showed the fixed wrapper passed the previous hydration
  blocker: official Advection sequential hydration completed, the
  beta-provenance full-task validation root was built, and
  `scripts/run_light_experiment.py` wrote
  `reports/research/sota_loop/model_side_transport_head_real_shard/ups_light_p2_model_side_beta_transport_head_val/summary.json`
  with aggregate decoded rollout NRMSE `0.11122069865446772`.
- The run then failed the required summary validator:
  `scripts/validate_model_side_transport_head_summary.py` reported missing
  `extra.model_side_transport_head` and
  `extra.model_side_transport_head_metrics`, plus the dependent task/mode/apply
  checks. Because the validator failed, this is not accepted validation
  evidence and no public/claim surface should change from it.
- No B2 result artifact was published before failure, and no held-out data was
  used.
- Contract `42412831` was destroyed after the schema failure was confirmed;
  sanitized follow-up status returned `not_found_active_instances`.

## Next Step

The no-provider summary plumbing blocker is repaired locally:
`scripts/run_light_experiment.py` now keeps the existing decoded-prefixed extras
and also preserves `extra.model_side_transport_head` and
`extra.model_side_transport_head_metrics` at top-level summary scope for the
model-side validator. Regression coverage in
`tests/unit/test_light_experiment_runner.py` exercises the `_evaluate_once`
summary surface used by the remote wrapper and calls
`scripts/validate_model_side_transport_head_summary.py` on the returned summary.
Focused verification passed with pytest, Black, Ruff, py_compile, and
`git diff --check`.

Next safe step: after the repair is committed and pushed, relaunch the same
bounded no-held-out remote route from the repaired git ref, monitor sanitized
Vast status, fetch only the small B2 result artifact if published, and validate
the summary before treating any metric as evidence.

Do not run held-out tests, update claim evidence, or change public language from
this launch alone.

## Schema-Repaired Relaunch Attempt

The summary propagation repair was committed and pushed as `41356df` on
`codex/poseidon-channel-lift-vast`. A bounded relaunch of the same no-held-out
remote route was attempted from that ref:

- Offer: `41528131`, RTX 4090.
- Contract: `42442415`.
- Observed rate from sanitized instance status: about
  `$0.37555555555555553/hr`.
- Disk: `48 GB`.
- Git ref: `codex/poseidon-channel-lift-vast`.
- Remote script:
  `scripts/run_remote_model_side_transport_head_real_shard.sh`.
- Launcher dry-run first confirmed the intended route: repaired branch,
  remote scratch, B2 hydration, no held-out stage, and no local hydration.
- Launch response returned a new contract but reported `success: false`.
- Sanitized status showed `actual_status = loading`, `cur_state = stopped`,
  `intended_status = stopped`, and no status message. A short recheck did not
  find a usable running state.
- Contract `42442415` was destroyed.

Outcome: no remote wrapper ran, no official hydration occurred, no B2 result
artifact was published, no held-out data was used, and no metric was produced.
This is an offer/instance launch failure, not validation evidence.

Next safe step: do not blindly reuse explicit offer `41528131`. Search/select an
alternate bounded Vast offer or choose a non-provider reroute, then relaunch
only under the same no-held-out artifact and summary-validator contract.

## Alternate-Offer Relaunch

A fresh sanitized offer search was run after contract `42442415` stopped
immediately. Offer `41528131` was excluded. The search found multiple verified
RTX 4090 offers under the `$0.45/hr` stop threshold; offer `41175200` was chosen
because it was below the recent price baseline and had strong reported network
and disk throughput.

Launcher dry-run:

- Git ref: `codex/poseidon-channel-lift-vast`.
- Offer: `41175200`, RTX 4090.
- Disk: `48 GB`.
- Route: `scripts/run_remote_model_side_transport_head_real_shard.sh`.
- Scope confirmed: repaired branch, remote scratch, B2 hydration, no held-out
  stage, no local hydration.

Launch:

- Contract: `42450012`.
- Launch response: `success: true`.
- Latest sanitized status before this note update: `actual_status = running`,
  `cur_state = running`, `intended_status = running`, RTX 4090, `48 GB` disk,
  observed `dph_total = 0.37555555555555553`, status message
  `success, running pytorch/pytorch_2.2.0-cuda12.1-cudnn8-runtime/ssh`.
- Expected result prefix:
  `b2://pdebench/remote-runs/model-side-transport-head/`.

Local B2 prefix inspection was not available through `rclone` on this machine
because no local `b2` rclone config is present; this does not affect the remote
wrapper's explicit B2 environment. A later tick should use the project-configured
B2 path/tooling to inspect any small published result artifact.

Next monitoring step: poll sanitized status for contract `42450012`. If it is
still running or plausibly loading, stay quiet and continue monitoring. If it
publishes the expected small B2 artifact, fetch only that artifact and validate
the summary with `scripts/validate_model_side_transport_head_summary.py` before
accepting any metric. If the instance fails, stalls, or does not tear down,
destroy it and record the failure.

## Alternate-Offer Final Outcome

Contract `42450012` completed and reached `actual_status = exited`,
`cur_state = stopped`, `intended_status = stopped`. A redacted log tail showed:

- official Advection sequential hydration completed;
- the beta-provenance full-task validation root was built;
- `scripts/run_light_experiment.py` wrote
  `reports/research/sota_loop/model_side_transport_head_real_shard/ups_light_p2_model_side_beta_transport_head_val/summary.json`;
- `scripts/validate_model_side_transport_head_summary.py` returned
  `errors = []`, `passed = true`;
- the artifact was published to
  `b2://pdebench/remote-runs/model-side-transport-head/model_side_transport_head_real_shard_20260625T022059Z.tar.gz`.

The artifact was downloaded locally to
`reports/research/sota_loop/model_side_transport_head_real_shard_remote_artifacts/model_side_transport_head_real_shard_20260625T022059Z.tar.gz`.
Its SHA256 is
`9778317b2942728e0d5e9bd503baadbecd66ee08ef44968e9ed60eb2dff9e905`.

Validated metrics:

- aggregate decoded rollout NRMSE: `0.11122069837659315`
- advection rollout NRMSE: `0.0017868115829009724`
- advection h16 NRMSE: `0.001784282965734058`
- Burgers rollout NRMSE: `0.14738121133726986`
- Darcy rollout NRMSE: `0.18897951477635447`
- model-side shift mean: `3.5019181072711945`
- applied samples: `512`
- skipped samples: `0`
- beta-missing samples: `0`

Boundary checks:

- `full_task_beta_val_root_manifest.json` records
  `held_out_test_data_read = false` and `test_ledger_writes = []`.
- `official_advection_hydration_plan_run.json` records test split not
  downloaded or sharded.
- No claim-evidence or public-language artifact was updated.
- Contract `42450012` was destroyed after completion; no active route instance
  remains.

Decision: this is accepted validation evidence for the scoped model-side beta
transport-head branch. It is not public claim evidence by itself because beta
provenance remains outside the universal public inference contract.

Next safe step: run a no-provider protocol/evidence mapping review before any
held-out pretest contract, claim-evidence update, or public-language change.

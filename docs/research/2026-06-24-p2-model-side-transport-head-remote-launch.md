# P2 Model-Side Transport Head Remote Launch

Date: 2026-06-24

Status: remote validation route launched; result pending. No held-out test has
been requested or authorized by this launch note.

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

## Next Monitoring Step

Poll contract `42401821`. If it completes, fetch/inspect the B2 artifact and
validate the summary locally if the artifact is small enough. If it stalls,
fails, or auto-shutdown does not tear it down, destroy the instance and record
the failure before relaunching.

Do not run held-out tests, update claim evidence, or change public language from
this launch alone.

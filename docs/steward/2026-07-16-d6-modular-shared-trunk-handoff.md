# D6 Modular Shared-Trunk Handoff

Date: 2026-07-16

## Protocol-integrity update

Independent pre-merge review stopped the original D6 plan before execution.
The original plan self-hash
`ec36aead4c537267fae78c71de8d14156fba253899f90ec72fe867dd6bce80e8`
is abandoned and must never be launched. No D6 run occurred under it.

The executable replacement D6 v4 contract is
`docs/superpowers/plans/2026-07-16-modular-shared-trunk-d6-v4.md`. Its
implementation commit is `4a003fa1952a0995574052c5bc5e1e5d8e119815` and
its executable plan is
`docs/research/artifacts/strat_v1_modular_shared_trunk_plan_v4.json`, with
self-hash
`88bcb9c70eefa1f7bda97577ff65dcd82e080022594cb9a3b5181b9418b06487`.
It binds 84 source/runtime files and the same six train/validation objects.

The repairs make adapter placement exact, independently recompute the
parameter-shuffle degradation, cryptographically binds the stage report, and
persist self-hashed per-arm resource evidence across resume. V4 independently
requires the canonical six-object map as well as exact list length, uniqueness,
roles, and checksum algorithms; it rejects all retired plan identities before
provider access. PR #127 must be updated with v4 and rerun CI before merge.

Use this document to resume D6 in a fresh Codex thread. Start by inspecting
live Git, GitHub, Vast, and artifact state; do not assume this snapshot is
still current.

## Objective and north star

Determine whether one universal simulator can retain a genuinely shared
conditioned operator while using only small task-specific modules to avoid the
cross-task interference observed in D5. The experiment must distinguish model
quality from consolidation economics and must not use held-out data.

D6 is the last planned shared-model test at this scale. If either U1 or U2
fails, stop broadening the shared model and move to a unified product interface
over family-specific models. Do not respond to a failure with extra seeds,
longer training, relaxed gates, more datasets, or a replacement run.

## Current repository state

- Canonical repository: `/Users/emerygunselman/Code/universal_simulator`
- Active worktree at handoff:
  `/Users/emerygunselman/.codex/worktrees/e5a4/universal_simulator`
- PR branch: `codex/modular-shared-trunk`
- Remote branch: `origin/codex/modular-shared-trunk`
- Base at branch creation: `ebe3d6701fae17f936e0a5ea0db0f87c28ee6196`
- Executable replacement implementation commit:
  `4a003fa1952a0995574052c5bc5e1e5d8e119815`
- Worktree was clean and fully pushed at handoff.

PR #127 is open. The original head passed CI, but review stopped it before
merge. Update the PR with D6 v4 and require a fresh green CI run. No D6 run has
been launched.

Before acting, run:

```bash
git status -sb
git fetch origin
git log -3 --oneline --decorate
git ls-remote origin refs/heads/codex/modular-shared-trunk refs/heads/main
```

Then check whether a PR already exists before creating another one.

## What was implemented

D6 adds paired task-routed residual adapters around the existing shared latent
operator:

- tasks: `advection1d`, `burgers1d`, `darcy2d`;
- one bottleneck-16 input adapter and one bottleneck-16 output adapter per task;
- dense routing through the existing one-hot `task_id` condition;
- input adapters after time/AdaLN conditioning and immediately before the PDE
  Transformer; output adapters after shared output normalization and before the
  outer state residual;
- zero-initialized output projections, preserving the initial D5 operator function;
- unchanged shared grid encoder, AnyPoint decoder, conditioning schema, and
  PDE-Transformer trunk.

Four arms are frozen:

1. `joint-modular`
2. `ablation-advection1d`
3. `ablation-burgers1d`
4. `ablation-darcy2d`

Every arm contains the full three-task adapter inventory. This keeps the module
graph and checkpoint inventory comparable. The single-task arms train on only
their assigned task; there is no separate `trainable_adapters` configuration.

Important files:

- `src/ups/models/latent_operator.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- `configs/d6_strat_v1_modular_shared_trunk.yaml`
- `scripts/run_strat_v1_modular_shared_trunk.py`
- `scripts/materialize_strat_v1_modular_shared_trunk.py`
- `scripts/plan_strat_v1_modular_shared_trunk.py`
- `scripts/run_remote_strat_v1_modular_shared_trunk.sh`
- `scripts/launch_strat_v1_modular_shared_trunk_vast.sh`

The final rationale and gates are in
`docs/superpowers/plans/2026-07-16-modular-shared-trunk-d6-v4.md`.

## Frozen plan and data boundary

Executable plan:
`docs/research/artifacts/strat_v1_modular_shared_trunk_plan_v4.json`

- Plan SHA-256:
  `88bcb9c70eefa1f7bda97577ff65dcd82e080022594cb9a3b5181b9418b06487`
- Bound implementation commit:
  `4a003fa1952a0995574052c5bc5e1e5d8e119815`
- Bound source/runtime files: 84
- Seed: 17
- Mode: validation only
- Held-out access: forbidden
- Measurement-lock access: forbidden

Training lock:
`docs/data/releases/strat-v1/universal/9d43d283f04f5b8d17cf6126ad189075c53307e715d7d4f61af440c2fed155c1/training.lock.json`

It contains exactly three training and three validation objects. It contains no
test-role object. D5 metrics are immutable comparison references only; D5 is
not retrained.

Do not regenerate or edit the plan unless the experiment is explicitly
abandoned and a new experiment is separately preregistered. A merge commit may
descend from the bound implementation commit; the source files themselves must
still match the 84 hashes in the plan.

## Gates and accounting

U1 compares the joint arm with frozen D5 references:

- macro NRMSE at most `0.7584231366`;
- Advection at most `0.6388001070`;
- Burgers at most `0.6895805941`;
- Darcy at most `0.1403299866`;
- beat persistence and corrected regime spread at most `1.5` on every task;
- shuffled-parameter degradation at least `5%`;
- joint checkpoint smaller than the frozen D5 specialist ensemble;
- joint initialized tensor elements below the three matched D6 ablations;
- held-out reads exactly zero.

U2 compares the joint arm with the new matched ablations:

- joint macro at most `1.05x` ablation macro;
- no joint task above `1.10x` its matched ablation;
- exact equality of scheduled per-task source examples and rollout-weighted
  scheduled compute.

Do not claim per-task optimizer-step equality. A joint batch can contain
multiple tasks. Total scheduled optimizer updates are computed at whole-arm
scope and reported as efficiency evidence. Process RSS is a cumulative child
process-family high-water mark, not GPU peak memory.

Every configured stage must complete its exact preregistered epoch count.
Early stopping is disabled. D6 sets `training.fail_on_oom: true`; an OOM must
fail the run rather than skip a batch or sample.

## Verification already completed

The focused integrated suite passed 82 tests:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q \
  tests/unit/test_modular_latent_operator.py \
  tests/unit/test_strat_v1_modular_shared_trunk.py \
  tests/unit/test_modular_shared_trunk_harness.py \
  tests/unit/test_strat_v1_modular_shared_trunk_remote.py \
  tests/unit/test_strat_v1_shared_tier_b.py \
  tests/unit/test_strat_v1_shared_tier_b_pipeline.py \
  tests/unit/test_finalize_d5_presigned_transfer.py \
  tests/unit/test_d5_presigned_io.py
```

Black, Ruff, Python compilation, YAML duplicate checks, shell parsing, and
`git diff --check` also passed. A separate broader local training-logger test
hit the macOS sandbox restriction on `torch_shm_manager`; the failure occurred
while starting a multiprocessing DataLoader and was not a D6 assertion
failure. Let Linux CI provide the authoritative broader result.

## GitHub handoff

1. Inspect the branch and confirm the implementation, preregistration, and
   handoff commits plus a clean tree.
2. Check for an existing PR.
3. If absent, open a ready PR from `codex/modular-shared-trunk` to `main`.
4. Include the architecture, frozen validation-only boundary, signed Vast/B2
   path, 82-test result, and plan SHA in the PR body.
5. Monitor every required CI check. Diagnose and fix real failures; do not
   weaken tests or experiment boundaries.
6. Merge only when green.
7. Update both the active worktree and canonical checkout to the merged `main`
   without discarding unrelated user changes.

Do not launch from an unmerged or unpushed commit. Record the final merged
commit and confirm that the bound implementation commit is its ancestor.

## Vast and B2 execution

The remote path is intentionally ephemeral:

- local credentials generate short-lived, object-scoped B2 capabilities;
- Vast receives no B2 key or application secret;
- the six frozen objects are staged immediately before training;
- failure preserves sealed resume/log slots where possible;
- success uploads an ingress archive;
- trusted local finalization verifies the digest, creates an immutable
  content-addressed B2 object, and cleans temporary ingress objects;
- managed Vast teardown must succeed before final publication.

The launcher is bounded to one verified RTX 4090, disk at least 96 GB,
`$0.45/hour`, 600 minutes, and a maximum modeled cost of `$4.50`.

After merge and CI, first run a dry run from the merged checkout:

```bash
DRY_RUN=1 \
GIT_REF=<full-merged-main-commit> \
bash scripts/launch_strat_v1_modular_shared_trunk_vast.sh
```

Verify the selected ref, remote script, signed-transfer mode, managed teardown,
cost bound, and plan path. Then launch exactly once:

```bash
DRY_RUN=0 \
GIT_REF=<full-merged-main-commit> \
bash scripts/launch_strat_v1_modular_shared_trunk_vast.sh
```

The launcher reads local `.env` aliases for B2 configuration and uses the
existing Vast CLI account. Never print or copy secrets into logs or the remote
command. Check Vast balance before launch; add credit only if actually needed
and explicitly authorized. Preserve the launcher receipt and verify the
instance is destroyed even on failure.

Expected local result path after successful finalization:
`reports/research/strat_v1_modular_shared_trunk_result.json`

Expected B2 prefix:
`remote-runs/strat-v1-modular-shared-trunk`

## Result handling

The remote wrapper runs the independent materializer after the four arms and
the joint parameter-shuffle diagnostic complete. The materializer rechecks:

- plan and summary self-hashes;
- training-lock and config bindings;
- validation-only held-out evidence;
- arm and adapter inventory;
- full stage completion and exposure parity;
- checkpoint and adapter tensor evidence;
- U1 and U2 independently.

After download/finalization:

1. Verify the immutable B2 handle and archive SHA.
2. Inspect the stage, summary, and result artifacts directly.
3. Confirm `heldout_reads == 0` before discussing metrics.
4. Record the Vast contract/instance ID and destroyed state.
5. Commit only compact plan/result/provenance artifacts and documentation; do
   not commit raw data, caches, checkpoints, logs, or the run bundle.
6. Update `docs/current-state.md`, `docs/experiments/ledger.md`, and the D6 plan
   status with the exact evidence.

Interpretation is fail-closed:

- U1 failure: stop modular shared-model research at this scale.
- U1 pass and U2 failure: sharing still causes unacceptable negative transfer;
  stop the branch.
- U1 and U2 pass: D6 validates the architecture only on validation data. It may
  justify a separately preregistered U3/U4 experiment, but it does not itself
  authorize held-out access.

## Explicitly forbidden shortcuts

- No test-role or measurement-lock access.
- No extra seed, replacement run, epoch extension, or threshold relaxation.
- No dataset, normalization, optimizer, architecture, or dependency change.
- No additional PDEBench families or The Well during D6.
- No retraining D5 controls.
- No treating frozen D5 specialists as the matched U2 controls.
- No describing process RSS as GPU memory.
- No committing credentials, raw data, checkpoints, or large archives.

## Recommended first message in the fresh thread

> Resume the D6 modular shared-trunk handoff in
> `docs/steward/2026-07-16-d6-modular-shared-trunk-handoff.md`. Inspect live
> local, GitHub, Vast, and B2 state first. Then create or locate the PR, get CI
> green, merge, and execute the single preregistered validation-only Vast run.
> Do not access held-out data or alter the frozen plan.

# Universal Latent E1 Handoff

Date: 2026-07-17

## Current decision

E1 confirms codec-path negative transfer in D6 before the latent operator is
called. Do not implement task/family routing and do not run another shared
operator yet. Qualify a canonical codec, then test paired grid/mesh physical
states.

Canonical repository: `/Users/emerygunselman/Code/universal_simulator`

E1 worktree:
`/Users/emerygunselman/.codex/worktrees/e5a4/universal_simulator`

Branch: `codex/universal-latent-e1`

Base: merged main `4846966788d507c8efc60e9c59ffb6d57e4b840f`

## Evidence

Exact locked validation bytes staged under
`/tmp/universal-latent-e1-valid`:

- `advection1d_val.h5`: `47,288,356` bytes, SHA-256 `0671198b...`
- `burgers1d_val.h5`: `32,289,060` bytes, SHA-256 `496a66bc...`
- `darcy2d_val.h5`: `3,918,783` bytes, SHA-256 `2b345a58...`

Recovered D6 arms remain under
`/tmp/d6-result-extract/reports/research/strat_v1_modular_shared_trunk/arms`.

Canonical result:
`docs/research/artifacts/strat_v1_d6_universal_latent_codec_audit.json`

Result note:
`docs/research/2026-07-17-universal-latent-encoder-e1-result.md`

Joint-to-matched global codec NRMSE ratios:

- Advection input: `2.4922x`
- Burgers input: `1.2396x`
- Darcy coefficient: `2.9357x`
- Darcy solution: `1.0842x` globally; both codecs are poor

No operator was instantiated or called. No training, parameter update,
held-out read, or measurement-lock access occurred during measurement.

## Mechanism

- Joint/matched latent CKA exceeds `0.998` on all tasks, but cross-decoding is
  poor. Similar geometry does not imply an interchangeable latent basis.
- Darcy coefficient rank contracts from `7.245` physical to about `2.81`
  latent in both joint and matched encoders.
- Swapping only the Darcy decoder moves global coefficient NRMSE from about
  `1.4` to `0.5`; decoder behavior dominates that codec gap.
- `train_decoder` previously reconstructed coefficient `fields` but ignored
  Darcy solution `targets`. Solution decoding was learned only through coupled
  operator rollout stages.
- D6's `encoder.pt`/`encoder_joint.pt` and decoder equivalents contain
  identical tensors because joint training overwrote the base paths. Its true
  pre-joint state is unrecoverable.

## Repairs implemented

- `scripts/train.py` now includes coefficient plus solution in standalone
  canonical steady-operator codec supervision.
- Joint training now preserves `pre_joint/operator.pt`,
  `pre_joint/encoder.pt`, and `pre_joint/decoder.pt` below the checkpoint
  directory before compatibility paths are overwritten.
- `scripts/run_universal_latent_codec_audit.py` provides exact-data checks,
  2x2 encoder/decoder swaps, global and sample metrics, spectral diagnostics,
  geometry, CKA, checkpoint provenance, and a task probe.

## Verification

Focused unit/encoder suite: `12 passed`.

Focused PyTorch integration tests for single-task and variable-grid multitask
joint training: `2 passed` outside the macOS sandbox so
`torch_shm_manager` could run.

## Next gate

Specify a codec-only qualification contract before training:

1. direct reconstruction gates for temporal states, steady coefficients, and
   steady solutions;
2. a frozen canonical latent-basis/cross-decoding contract;
3. preserved pre/post checkpoints and source-bound artifacts;
4. paired same-state grid/mesh alignment and remeshing invariance for E2.

Only a codec that passes those gates may enter E3 shared-versus-specialized
operator testing. No held-out access is authorized.

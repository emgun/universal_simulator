# Canonical Latent E2 Foundation Handoff

Date: 2026-07-17

## Current decision

The original backbone lineage confirms the user's encoder-first direction.
Do not add a task/family/modality router. UPS had not faithfully implemented
the UPT common encoder: its grid and mesh/particle paths were separate, and its
mesh path used order-dependent chunk averaging instead of geometry-aware
supernodes, transformer processing, and learned Perceiver queries.

## Implemented

- `src/ups/io/enc_canonical_point.py`: one opt-in point-set encoder for regular
  grids, irregular meshes, and particles; deterministic geometry supernodes,
  local aggregation, transformer hierarchy, and shared learned latent queries.
- `src/ups/eval/latent_qualification.py`: paired latent alignment, symmetric
  retrieval, CKA, effective rank, and full encoder-source/query-discretization
  codec matrices.
- strict field-semantic/channel validation and point-order invariance tests.
- primary-source lineage note and frozen codec qualification contract.

Existing `GridEncoder` and `MeshParticleEncoder` remain unchanged as historical
and matched-control paths. No training, operator invocation, provider call,
held-out read, or scientific E2 measurement occurred.

## Verification

- New and adjacent encoder/decoder/codec suite: `19 passed`.
- Full `tests/unit` suite: passed outside the macOS sandbox. The sandboxed run
  failed only where localhost sockets and `torch_shm_manager` are prohibited;
  the same tests passed with those OS facilities available.
- Ruff, Black check, Python compilation, and `git diff --check`: passed.

## Interpretation

This branch repairs the architectural interface and evaluation mechanics; it
does not establish a trained universal latent space. The status is
`not_qualified` until the paired benchmark passes every reconstruction,
cross-decoding, retrieval, rank, remeshing, and boundary gate.

## Next coherent arc

1. Add an explicit config-selected codec-only training path using
   `CanonicalPointEncoder` plus one `AnyPointDecoder`.
2. Generate analytic physical states and sample each as regular grid, two
   independent irregular meshes, and a neutral canonical query set.
3. Freeze manifests, group split, units, identities, config, and thresholds.
4. Train matched grid-only, mesh-only, and canonical codecs with equal exposure.
5. Run the qualification matrix without instantiating a latent operator.

Only a full pass authorizes freezing the codec for E3 dynamics isolation.

# strat-v1 Advection Root: First Gate-Verified Stratified+Disjoint Shards

Date: 2026-07-09

Status: historical first root; superseded for forward execution on 2026-07-13.
Data construction only; nothing trained or scored.

The first published root omitted an explicit `beta` dataset in the shards and
used mutable remote paths. Its scientific allocation remains valid historical
evidence, but it is not a runnable forward release. The universal builder
reconstructed a canonical source with explicit beta provenance and emitted the
replacement immutable root. Its task release is
`docs/data/releases/strat-v1/advection1d/531c9c400721da3b46ac25a47fe1b1357d35f941e9bc48850d51b0f1360f5af8/`;
shard SHA-256 values are `aeaf3cc5...` train, `0671198b...` validation, and
`17b0741b...` test. The new root was fetched from B2 by the frozen universal
locks and re-verified before A3 froze.

## What was built

`scripts/make_light_hdf5_shards.py` applied the stratified block policy (block size 48 per official beta file; train offset 0 x32, val offset 32 x8, test offset 40 x8) to the official beta-provenance hydrated root, producing the first strat-v1 root:

- `b2://pdebench/strat-v1/advection1d/advection1d_{train,val,test}.h5` + `manifest.yaml`
- Sizes 256/64/64; SHA256: train `b427ca3f972972353a4a97da89bb4579b50f44225f39be4f1850e89def4c6a12`, val `945058d21b18b1cc905307f71a0da6b4e14bddc18672ec46a490497f2fc44c11`, test `e9b1dba0b537757f3ddc3ef367780aee7cf90da1da19490af73cc9187a48448c`.

## Build gates (both pass)

- **Zero cross-split overlap** (`scripts/audit_split_integrity.py`): train<->val 0, train<->test 0, val<->test 0. Artifact: `docs/research/artifacts/strat_v1_advection_integrity_audit.json`.
- **Every regime in every split**: provenance ground truth is exactly balanced — 32 train / 8 val / 8 test samples per beta for all eight betas. Regime-estimator artifact: `docs/research/artifacts/strat_v1_advection_regime_audit.json`; manifest copy: `docs/research/artifacts/strat_v1_advection_manifest.yaml`.

This root replaces light-v1 advection's "same ICs, disjoint speeds" construction with disjoint ICs and balanced speeds: in-distribution generalization becomes measurable, and beta-extrapolation stays measurable via the legacy roots.

## Remaining A2 work

- **Burgers (critical path):** regime stratification needs the official multi-viscosity sources (`1D_Burgers_Sols_nu*.hdf5` from Darus) hydrated with provenance, reusing the advection sequential-hydration machinery. Until then no honest Burgers split exists anywhere (light/medium test is contained in train; full-tier provenance unknown).
- **Darcy:** light splits are already clean/disjoint. **New finding: the full-tier Darcy train shard (40,000 samples) is one 10,000-sample source repeated exactly four times** (verified by exact array equality at indices i, i+10k, i+30k; field-statistics period 10,000). 75% of the file is duplicate data with no provenance attributes. The light Darcy splits were evidently sliced inside one block, which is why they are clean. Consequence: every full-tier shard must pass a dedup/provenance audit before strat-v1 or universal-v1 consumes it; `scripts/audit_split_integrity.py`'s `unique_keys_per_split` field already detects within-file duplication.
- **Medium tier (512/128/128):** requires hydrating more official samples per beta (current official root has 48/file); same pipeline, larger `samples_per_file`.

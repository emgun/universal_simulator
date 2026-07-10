# strat-v1 Advection Root: First Gate-Verified Stratified+Disjoint Shards

Date: 2026-07-09

Status: Track A2 progress (universal-baseline roadmap). Data construction only; nothing trained or scored; the reserved test shard was built and read only by the two protocol audits, recorded truthfully in their artifacts.

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
- **Darcy:** light splits are already clean/disjoint; full-tier train (2.6 GB) is being inspected for regime provenance to decide whether stratification applies or the existing construction is adopted into strat-v1 as-is.
- **Medium tier (512/128/128):** requires hydrating more official samples per beta (current official root has 48/file); same pipeline, larger `samples_per_file`.

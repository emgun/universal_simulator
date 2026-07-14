# Canonical Burgers and Darcy Source Plan

Date: 2026-07-12

Status: Darcy source hydration and the balanced `260/65/65` root are complete.
Burgers inspection, hydration, and the balanced `288/72/72` local root are
also complete; see `2026-07-13-strat-v1-burgers-root.md`. Durable Burgers
publication remains. No candidate evaluation or model run occurred.

## Decision

Build both task roots from the official files listed in `docs/pdebench_manifest.yaml`. Do not repair or reuse the provenance-free Darcy 40,000-row derivative, and do not reuse legacy Burgers shards.

`scripts/hydrate_official_canonical_source.py` now provides the shared source contract:

- verifies every required manifest regime, file size, and MD5 before conversion;
- requires an explicit physical-field dataset and exact raw per-sample shape
  rather than guessing a dataset or accepting any numeric tensor;
- materializes stable `source_file_id`, `source_file_index`, `source_sample_index`, and the task parameter (`nu` or `beta`);
- rejects non-finite fields and duplicate fields that invalidate the task contract;
- writes atomically with a source catalog and manifest digest;
- produces the schema required by the universal split builder.

## Burgers

The official inventory contains 12 train files at `nu = 0.001, 0.002, 0.004, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1, 2, 4`, totaling 98,795,619,744 bytes (about 92.0 GiB).

Twelve regimes make the earlier `256/64/64` target impossible under exact balance. The smallest clean light-tier counts above those targets are:

- train: 288 (24 per viscosity)
- validation: 72 (6 per viscosity)
- test: 72 (6 per viscosity)
- canonical source requirement: 36 rows per viscosity

After inspecting one official file, update
`docs/protocols/canonical_source_schema.yaml` from `pending_inspection` to
`frozen` with the confirmed key, shape, and semantic role. The CLI does not
accept schema overrides. Only then will the expected build run:

```bash
python scripts/hydrate_official_canonical_source.py \
  --task burgers1d \
  --manifest docs/pdebench_manifest.yaml \
  --raw-root /workspace/pdebench_raw \
  --out /workspace/canonical/burgers1d_train.h5 \
  --samples-per-regime 36 \
  --output-json reports/research/strat_v1/burgers_source.json

python scripts/make_light_hdf5_shards.py \
  --root /workspace/canonical \
  --out-root /workspace/strat-v1/burgers1d \
  --tasks burgers1d \
  --train-count 288 --val-count 72 --test-count 72 \
  --provenance-dataset source_file_id \
  --provenance-dataset source_sample_index \
  --regime-dataset nu \
  --field-kind temporal --time-axis 1 \
  --manifest /workspace/strat-v1/burgers1d/manifest.yaml \
  --version strat-v1 --remote-prefix strat-v1 --overwrite
```

## Darcy

The authoritative inventory contains five train files at `beta = 0.01, 0.1, 1, 10, 100`, totaling 6,553,622,984 bytes (about 6.10 GiB). Rehydrating these files restores the source identity that cannot be recovered from the repeated 40,000-row derivative.

Five regimes likewise require adjusted balanced counts:

- train: 260 (52 per beta)
- validation: 65 (13 per beta)
- test: 65 (13 per beta)
- canonical source requirement: 78 rows per beta

First-file inspection is complete: official `nu` has shape `[N,128,128]` and
is the heterogeneous coefficient input; `tensor` has shape `[N,1,128,128]`
and is the continuous steady solution target. The frozen canonical form stores
both as `[N,1,128,128,1]`, where the singleton leading axis is an operator
state axis, not time. All five files were hydrated with that contract:

```bash
python scripts/hydrate_official_canonical_source.py \
  --task darcy2d \
  --manifest docs/pdebench_manifest.yaml \
  --raw-root /workspace/pdebench_raw \
  --out /workspace/canonical/darcy2d_train.h5 \
  --samples-per-regime 78 \
  --output-json reports/research/strat_v1/darcy_source.json

python scripts/make_light_hdf5_shards.py \
  --root /workspace/canonical \
  --out-root /workspace/strat-v1/darcy2d \
  --tasks darcy2d \
  --train-count 260 --val-count 65 --test-count 65 \
  --provenance-dataset source_file_id \
  --provenance-dataset source_sample_index \
  --regime-dataset beta \
  --field-kind steady \
  --manifest /workspace/strat-v1/darcy2d/manifest.yaml \
  --version strat-v1 --remote-prefix strat-v1 --overwrite
```

## Stop Conditions

- Stop if any official size or MD5 differs from the committed manifest.
- Stop if either explicit Darcy input/target key or expected rank is wrong.
- Freeze the inspected field keys, semantic roles, dtypes, and exact sample shapes
  in the run record before hydrating the remaining files.
- Permit a matched coefficient only at the same `source_sample_index` across
  beta regimes; stop on duplicate coefficient-regime pairs or cross-index
  coefficient collisions.
- Stop on duplicate Burgers trajectories within a viscosity or cross-split initial-field overlap.
- Do not publish until the exact artifact set and hashes pass the publication validator.
- Do not run a baseline or candidate until all three task roots and the protocol contract are frozen.

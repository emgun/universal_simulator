# Data pipeline architecture

The data pipeline has one authority path for PDEBench, The Well, and future
sources:

```text
pinned source inventory + protocol
              |
              v
       immutable run lock
              |
              v
  verified content-addressed cache
              |
              v
       local zero-copy run view
              |
              v
 source adapter -> trajectory/model input
```

## Components

- `catalog/`: reviewed upstream identity, license, citation, objects, checksums,
  sizes, and exact mirrors. Metadata-only catalogs cannot resolve.
- `protocols/`: split authority, identity fields, stable selection,
  normalization, coverage, and held-out policy.
- `src/ups/data/manifests.py`: strict parsing and deterministic run-lock
  resolution.
- `src/ups/data/staging.py`: disk preflight, resumable exact transfer,
  checksum-before-promotion, cache locking, and source-relative run views.
- `src/ups/data/pdebench.py`: worker-safe lazy local HDF5 access.
- `src/ups/data/well.py`: lazy mapping from The Well's native loader into the
  source-neutral contract in `trajectory.py`.
- `src/ups/data/normalization.py`: incremental train-only statistics bound to
  the run lock and sample selection.
- `src/ups/data/cli.py`: `resolve`, `plan`, `stage`, `verify`, `fit-stats`, and
  safe dry-run-first `evict` commands.

## Non-negotiable boundaries

- Training locks never contain test objects.
- Held-out bytes require a separate measurement lock and contract identifier.
- Remote URIs are transfer mirrors, not live random-access training mounts.
- A checksum is verified before bytes receive their final cache identity.
- Normalization is never inferred from validation or test data.
- Source adapters preserve source semantics; The Well is not transcoded into a
  lowest-common-denominator PDEBench file.

## The Well pilot inventory

The `turbulent_radiative_layer_2D` pilot is pinned to The Well package v1.2.0
(Git commit `bf4e08a3d0231f590cc796052e32c1a25fb816f9`) and Hugging Face dataset
commit `2ee7756575ff6f90981d0308cbcb6a2ab5995bdc`. The source catalog contains all
27 native HDF5 objects and their Hub Git LFS SHA-256 values:

- train: 9 objects, 5,888,802,816 bytes
- valid: 9 objects, 981,467,136 bytes
- test: 9 objects, 981,467,136 bytes
- total: 27 objects, 7,851,737,088 bytes

The integration pilot selects the `tcool=0.03` object inside each authoritative
upstream split. Its train-plus-validation lock is 763,363,328 bytes; the
separately authorized test object is 109,051,904 bytes. This is an integration
slice, not a representative benchmark claim.

Regenerate the catalog and protocol using metadata only:

```bash
PYTHONPATH=src python scripts/inventory_the_well_hf.py \
  --repo polymathic-ai/turbulent_radiative_layer_2D \
  --revision 2ee7756575ff6f90981d0308cbcb6a2ab5995bdc \
  --package-version v1.2.0 \
  --package-commit bf4e08a3d0231f590cc796052e32c1a25fb816f9 \
  --pilot-parameter 0.03 \
  --source-output docs/data/catalog/the_well.yaml \
  --protocol-output docs/data/protocols/the_well_native_v1.yaml
```

The script calls only the Hugging Face repository metadata API. It rejects
mutable revisions, missing LFS checksums, inconsistent sizes, and a pilot that
does not select exactly one native object from every upstream split.

## Current validation state

The Well training pilot has been staged from its exact lock, verified in the
content-addressed cache, opened through the native adapter, and read through a
spawned `DataLoader` worker. Source-relative train/valid paths are deliberately
preserved because the two upstream objects share a basename. This proves the
local data-plane integration; the pilot remains too narrow for benchmark or
model-quality claims.

The three-task `strat-v1` release is frozen. Advection `256/64/64`, Burgers
`288/72/72`, and Darcy `260/65/65` are regime-balanced and provenance-disjoint.
All nine shards and the three compact canonical sources are published at
content-addressed `b2://pdebench/strat-v1/immutable/sha256/...` keys. A clean
combined training fetch transferred six objects into a training-only cache; a
separate measurement contract fetched the three reserved test objects into a
different cache. Every object then passed an independent size and SHA-256
verification. Exact task and universal releases are under `releases/strat-v1/`.

# Data control, staging, and loading

UPS treats dataset identity as part of the scientific protocol. Training no longer
accepts a dataset nickname, mutable remote alias, tar archive, or W&B dataset
artifact. A run begins from two reviewed inputs:

- a source manifest: immutable revision, exact objects, byte sizes, checksums,
  mirrors, license, and citation;
- a protocol manifest: split authority, sample identity and selection, adapter
  revision, normalization policy, and test-access policy.

The checked-in catalogs under `docs/data/catalog/` describe PDEBench and The Well.
The Well catalog is an exact, reviewed inventory and can produce runnable locks.
The complete PDEBench upstream inventory is checksum-audited metadata, but it
cannot produce a source lock until reviewed protocol roles are assigned to exact
objects. Metadata-only catalogs deliberately cannot produce runnable locks.

## Resolve an immutable run lock

```bash
PYTHONPATH=src python -m ups.data.cli resolve \
  --source path/to/pdebench-source.yaml \
  --protocol docs/data/protocols/strat_v1.yaml \
  --roles train valid \
  --output runs/example/training.data.lock.json
```

The lock is canonical JSON with its own SHA-256 identity. Training locks cannot
contain test objects. Test evaluation requires a separately resolved measurement
lock and an explicit measurement contract:

```bash
PYTHONPATH=src python -m ups.data.cli resolve \
  --source path/to/pdebench-source.yaml \
  --protocol docs/data/protocols/strat_v1.yaml \
  --purpose measurement --roles test \
  --measurement-contract-id heldout-v1 \
  --output runs/example/measurement.data.lock.json
```

## Plan and stage

```bash
PYTHONPATH=src python -m ups.data.cli plan \
  --lock runs/example/training.data.lock.json \
  --cache /scratch/ups-data

PYTHONPATH=src python -m ups.data.cli stage \
  --lock runs/example/training.data.lock.json \
  --cache /scratch/ups-data \
  --run-dir data/pdebench \
  --report runs/example/data-stage.json

PYTHONPATH=src python -m ups.data.cli verify \
  --lock runs/example/training.data.lock.json \
  --cache /scratch/ups-data
```

The stager supports exact local/file and HTTP(S) mirrors, resumes partial transfers,
verifies size and checksum before atomic promotion, and stores each object once in a
content-addressed cache. The run directory contains source-relative hard links (or
symlinks when a hard link is unavailable), so native layouts and repeated basenames
remain valid without duplicating bytes. PDEBench canonical shards retain their flat
`<task>_<split>.h5` names. A B2 object is useful only when its exact URI is one
mirror in the lock; B2 is not a separate discovery or streaming mode.

## Normalization

Normalization is off unless checksum-bound training statistics are supplied. Fit
them only from a training lock and its staged training view:

```bash
PYTHONPATH=src python -m ups.data.cli fit-stats \
  --lock runs/example/training.data.lock.json \
  --root data/pdebench --task burgers1d \
  --output runs/example/burgers1d.normalization.json
```

Set `data.normalize=true`, `data.normalization_path`, `data.data_lock_sha256`, and
`data.selection_sha256` in a training config. The loader rejects statistics whose
recorded identities do not match.

Darcy inputs and targets have different physical meanings and distributions. Fit
them separately with `--component fields` and `--component targets`, then set
`data.normalization_path` and `data.target_normalization_path`, respectively.
The loader rejects normalized Darcy data unless both checksum-bound files exist.

## Runtime loading and storage tradeoff

PDEBench HDF5 shards are opened lazily per worker; arrays are sliced in
`__getitem__` and are never remotely mounted. The Well stays in its native
chunked dataset representation and is mapped through the source-neutral trajectory
adapter. This is the cost/performance optimum for typical GPU jobs:

1. keep the canonical cold copy upstream;
2. optionally keep only expensive or unavailable mirrors in object storage;
3. stage the exact working set once to job-local SSD;
4. reuse verified cache objects across runs;
5. stream samples locally with worker-aware lazy readers.

Direct remote HDF5 reads save some scratch capacity but amplify random-read latency,
egress, retry complexity, and worker contention. Full permanent local mirrors are
fast but unnecessarily expensive. Selective verified staging occupies the useful
middle.

Cache eviction is dry-run by default and preserves objects referenced by supplied
locks:

```bash
PYTHONPATH=src python -m ups.data.cli evict \
  --cache /scratch/ups-data \
  --lock runs/example/training.data.lock.json
```

Add `--apply` only after reviewing the reported unpinned objects.

## Remote launch

```bash
DATA_LOCK=runs/example/training.data.lock.json \
DATA_CACHE=/scratch/ups-data \
WANDB_PROJECT=universal-simulator \
bash scripts/run_remote_scale.sh
```

The launcher stages only training/validation data and precomputes only those latent
caches. It skips held-out evaluation unless `RUN_TEST_MEASUREMENT=1` and
`MEASUREMENT_DATA_LOCK` names a separately authorized measurement lock. W&B remains
the metrics/checkpoint/evidence plane, not the dataset byte plane.

# Data Artifact Workflow

This guide describes how to convert raw PDEBench data (and future physics datasets) into UPS-ready artifacts that remote jobs can consume via the registry + fetch helper.

## 1. Prepare raw data
Download the official PDEBench release locally or to a scratch volume (e.g., `data/pdebench/raw`). Maintain the original directory layout (`1D/Burgers/Train/*.hdf5`, etc.).

## 2. Convert to UPS format
Use the streaming converters so you never exhaust RAM while merging shards.

### Burgers / Advection / Navier–Stokes grids
```
PYTHONPATH=src python scripts/convert_pdebench_multimodal.py burgers1d \
  --root data/pdebench/raw --out data/pdebench --limit 5 --samples 200
```
Produces `data/pdebench/burgers1d_train.h5`. Repeat for `advection1d`, `navier_stokes2d`. Adjust `--limit` / `--samples` to control dataset size.

For more control (custom globs or splits), fall back to the generic CLI:
```
PYTHONPATH=src python scripts/convert_pdebench.py \
  --pattern 'data/pdebench/raw/1D/Burgers/Train/*.hdf5' \
  --out data/pdebench/burgers1d_train.h5 --limit 5 --samples 200
```

### Mesh / Particle datasets
Mesh/particle subsets are exported to Zarr stores that match the UPS loaders:
```
PYTHONPATH=src python scripts/convert_pdebench_multimodal.py darcy2d_mesh \
  --root data/pdebench/raw --out data/pdebench --limit 10

PYTHONPATH=src python scripts/convert_pdebench_multimodal.py particles_advect \
  --root data/pdebench/raw --out data/pdebench --limit 10
```
Mesh tasks expect `.npz` shards that provide `coords`, `cells`, `edges`, and Laplacian CSR arrays (`laplacian_data`, `laplacian_indices`, `laplacian_indptr`). Particle tasks expect `.npz` shards containing `positions`, `velocities`, and neighbour graph CSR data (`indices`, `indptr`, `edges`). The converter checks for these keys and materialises Zarr groups compatible with `MeshZarrDataset` and `ParticleZarrDataset`.

## 3. (Optional) Precompute latent caches

To avoid re-encoding the HDF5 grids on every training epoch, generate cached latent
sequences once and re-use them across runs:

```
PYTHONPATH=src python scripts/precompute_latent_cache.py \
  --tasks burgers1d advection1d \
  --splits train val test \
  --cache-dir data/latent_cache \
  --latent-dim 512 --latent-len 128 \
  --num-workers 8 --pin-memory
```

The script stores per-sample latent tensors under `data/latent_cache/<task>_<split>/`.
Point `training.latent_cache_dir` in the config to the same directory and future
training/eval stages will stream the cached latents instead of recomputing them.

## 4. Package the dataset
Tarball the converted files to keep artifact uploads atomic:
```
mkdir -p artifacts
 tar -czf artifacts/burgers1d_subset_v1.tar.gz -C data/pdebench burgers1d_train.h5 burgers1d_val.h5 burgers1d_test.h5
```

## 5. Upload to W&B
```
python scripts/upload_artifact.py burgers1d-subset dataset artifacts/burgers1d_subset_v1.tar.gz \
  --metadata '{"samples":20,"resolution":[201,1024],"parameters":["nu"]}'
```
Record the resulting artifact ID and update `docs/dataset_registry.yaml` accordingly (replace placeholder `<entity>/<project>/…`).

## 6. Hydrate on any machine
```
PYTHONPATH=src python scripts/fetch_datasets.py burgers1d_subset_v1 --root data/pdebench --cache artifacts/cache
```
This downloads the artifact, copies train/val/test files into `data/pdebench`, and caches the raw artifact for future runs.

## 7. Launch remote training/eval
```
WANDB_PROJECT=universal-simulator WANDB_ENTITY=your-entity \
WANDB_DATASETS="burgers1d_subset_v1" bash scripts/run_remote_scale.sh
```
The runner fetches datasets, executes all training stages, and runs evaluations. Use `configs/scale_overrides.md` for quick knob tuning.

---

Repeat the steps for each dataset family (Advection, Navier–Stokes, Darcy mesh, particle flows). Document new artifacts in the registry so downstream jobs can consume them without manual steps.

---
date: 2025-11-07T08:00:20-08:00
researcher: Claude Code
git_commit: 3324bfec21c1a268d026716134033792ee1ecb0b
branch: feature--UPT
repository: universal_simulator
topic: "PDEBench Multi-Dataset Scaling Strategy for UPS"
tags: [research, codebase, data-pipeline, pdebench, scaling, multi-task, architecture, b2-storage, vastai, vultr, end-to-end]
status: complete
last_updated: 2025-11-07
last_updated_by: Claude Code
last_updated_note: "Added comprehensive end-to-end strategy + CRITICAL OPTIMIZATION: Pre-compute latent caches once, upload to B2, reuse forever. Saves ~90% startup time (~2-4 hours → 15-30 min) and ~85% cost per run."
---

# Research: PDEBench Multi-Dataset Scaling Strategy for Universal Physics Stack

**Date**: 2025-11-07T08:00:20-08:00
**Researcher**: Claude Code
**Git Commit**: 3324bfec21c1a268d026716134033792ee1ecb0b
**Branch**: feature--UPT
**Repository**: universal_simulator

## Research Question

How can UPS optimally scale to incorporate all available PDEBench datasets (15+ tasks across grid, mesh, and particle modalities) with optimal preprocessing, storage, and training strategies while leveraging the Universal Physics Transformer (UPT) architecture?

## Executive Summary

The Universal Physics Stack has **production-ready infrastructure** for multi-dataset PDEBench training:

**CORRECTED STATUS** (Previous research was outdated):
1. ✅ **Golden Config**: `configs/train_burgers_upt_full.yaml` (UPT Phase 4 - Complete Implementation)
   - Architecture: Pure stacked transformer (`pdet_stack`) with 128 tokens, 128-dim latent
   - Performance: Target NRMSE < 0.055 (>7% improvement over Phase 3's 0.0593 SOTA)
   - Features: **ALL Phase 4 advanced features enabled**:
     - ✅ UPT inverse losses (λ=0.01, every batch)
     - ✅ Query-based training (2048 queries, 20-30% speedup)
     - ✅ Physics priors (boundary enforcement, λ=0.05)
     - ✅ Latent regularization (norm + diversity penalties, λ=1e-4)
     - ✅ Zero-shot super-resolution capability (2x, 4x)
   - Optimizer: Muon Hybrid (lr=1.4e-3)
2. ✅ **Dataset Tooling**: Both `scripts/prepare_data.py` and `scripts/convert_pdebench_multimodal.py` exist and are functional
3. ✅ **Multi-Task Infrastructure**: `build_latent_pair_loader` supports ConcatDataset, mesh/particle encoders operational
4. ⚠️ **Remaining Gaps**: Limited task specs (10/15+ PDEBench tasks), no multi-task training configs, curriculum learning not implemented

**Recommended Path**:
- **Phase 1** (COMPLETE): Dataset tooling ✅
- **Phase 2** (IN PROGRESS): Extend task specs, create multi-task configs
- **Phase 3** (PLANNED): Curriculum learning, hyperparameter scaling
- **Phase 4** (RESEARCH): Production multi-dataset training

## Comprehensive End-to-End PDEBench Strategy

### Complete Pipeline: Source Download → B2 Storage → Remote Training → Evaluation

This section provides a **production-ready, end-to-end strategy** for incorporating all PDEBench datasets into UPS training workflows, covering official data acquisition, preprocessing, cloud storage (B2), remote instance data ingestion (VastAI/Vultr), multi-task training, and evaluation.

---

## **Stage 0: PDEBench Official Data Acquisition**

### **0.1 Clone PDEBench Repository**

The PDEBench repository provides official download scripts and data management tools.

```bash
# Clone PDEBench repo for download utilities
git clone https://github.com/pdebench/PDEBench.git
cd PDEBench

# Install dependencies
pip install -e .
```

**Repository Structure** (relevant components):
- `pdebench/data_download/download_direct.py` - Primary download script (recommended)
- `pdebench/data_download/download_easydataverse.py` - Alternative (slower, may have issues)
- `pdebench/data_download/config/` - Dataset configuration files
- `pdebench/data_download/pdebench_data_urls.csv` - Dataset URL mappings

**Data Source**: DaRUS Repository (University of Stuttgart)
- **Datasets**: DOI: 10.18419/darus-2986
- **Pre-trained Models**: DOI: 10.18419/darus-2987

### **0.2 Download Datasets with Official Scripts**

PDEBench provides 11 distinct PDE datasets with varying sizes:

| Dataset | Command | Size | Priority | Notes |
|---------|---------|------|----------|-------|
| `advection` | `--pde_name advection` | 47 GB | **HIGH** | 1D, fast training |
| `burgers` | `--pde_name burgers` | 93 GB | **HIGH** | 1D, current golden task |
| `diff_sorp` | `--pde_name diff_sorp` | 4 GB | **MEDIUM** | 1D diffusion-sorption |
| `1d_reacdiff` | `--pde_name 1d_reacdiff` | 62 GB | **MEDIUM** | 1D reaction-diffusion |
| `darcy` | `--pde_name darcy` | 6.2 GB | **HIGH** | 2D, includes mesh variant |
| `2d_reacdiff` | `--pde_name 2d_reacdiff` | 13 GB | **MEDIUM** | 2D reaction-diffusion |
| `swe` | `--pde_name swe` | 6.2 GB | **MEDIUM** | 2D shallow water |
| `1d_cfd` | `--pde_name 1d_cfd` | 88 GB | **LOW** | 1D compressible NS |
| `2d_cfd` | `--pde_name 2d_cfd` | 551 GB | **LOW** | 2D compressible NS |
| `3d_cfd` | `--pde_name 3d_cfd` | 285 GB | **LOW** | 3D compressible NS |
| `ns_incom` | `--pde_name ns_incom` | **2.3 TB** | **LOW** | 2D incompressible NS (MASSIVE) |

**Download Commands**:

```bash
# Set download root
export PDEBENCH_ROOT=/path/to/storage/pdebench_raw
mkdir -p $PDEBENCH_ROOT

# Navigate to PDEBench data_download directory
cd PDEBench/pdebench/data_download

# Download high-priority datasets (recommended starting point)
python download_direct.py --root_folder $PDEBENCH_ROOT --pde_name advection
python download_direct.py --root_folder $PDEBENCH_ROOT --pde_name burgers
python download_direct.py --root_folder $PDEBENCH_ROOT --pde_name darcy

# Download medium-priority datasets (for multi-task training)
python download_direct.py --root_folder $PDEBENCH_ROOT --pde_name diff_sorp
python download_direct.py --root_folder $PDEBENCH_ROOT --pde_name 1d_reacdiff
python download_direct.py --root_folder $PDEBENCH_ROOT --pde_name 2d_reacdiff
python download_direct.py --root_folder $PDEBENCH_ROOT --pde_name swe

# CAUTION: Large datasets (500GB+, download only if needed)
# python download_direct.py --root_folder $PDEBENCH_ROOT --pde_name 2d_cfd
# python download_direct.py --root_folder $PDEBENCH_ROOT --pde_name 3d_cfd
# python download_direct.py --root_folder $PDEBENCH_ROOT --pde_name ns_incom  # 2.3TB!
```

**Storage Planning**:
- **Starter Set** (advection + burgers + darcy): ~146 GB
- **Multi-Task Training** (7 datasets): ~231 GB
- **Full Suite** (all 11 datasets): ~3.1 TB

**Expected Download Times** (100 Mbps connection):
- advection (47 GB): ~1 hour
- burgers (93 GB): ~2 hours
- ns_incom (2.3 TB): **~50+ hours**

### **0.3 Dataset Organization**

Downloaded datasets follow PDEBench's original directory structure:

```
$PDEBENCH_ROOT/
├── 1D/
│   ├── Advection/
│   │   ├── Train/*.h5
│   │   ├── Valid/*.h5
│   │   └── Test/*.h5
│   ├── Burgers/
│   │   ├── Train/*.h5
│   │   └── ...
│   ├── CFD/  # 1d_cfd
│   ├── ReacDiff/  # 1d_reacdiff
│   └── diffusion-sorption/  # diff_sorp
├── 2D/
│   ├── DarcyFlow/
│   │   ├── regular/  # Grid-based
│   │   └── irregular/  # Mesh-based (.npz)
│   ├── CFD/  # 2d_cfd
│   ├── diffusion-reaction/  # 2d_reacdiff
│   ├── shallow-water/  # swe
│   └── NavierStokes/
│       ├── incompressible/  # ns_incom
│       └── compressible/  # 2d_cfd
└── 3D/
    └── CFD/  # 3d_cfd
```

---

## **Stage 1: Local Preprocessing & UPS Format Conversion**

Convert raw PDEBench datasets into UPS-compatible HDF5/Zarr formats optimized for training.

### **1.1 Convert Grid Tasks to HDF5**

Use `scripts/convert_pdebench_multimodal.py` to consolidate sharded HDF5 files:

```bash
cd /path/to/universal_simulator
export PDEBENCH_ROOT=/path/to/storage/pdebench_raw
mkdir -p data/pdebench

# 1D Tasks (Fast, ~5-10 min each, produces ~2-5 GB per task)
PYTHONPATH=src python scripts/convert_pdebench_multimodal.py burgers1d \
  --root $PDEBENCH_ROOT --out data/pdebench --limit 100 --samples 1000

PYTHONPATH=src python scripts/convert_pdebench_multimodal.py advection1d \
  --root $PDEBENCH_ROOT --out data/pdebench --limit 100 --samples 1000

# 2D Tasks (Moderate, ~15-30 min each, produces ~5-15 GB per task)
PYTHONPATH=src python scripts/convert_pdebench_multimodal.py reaction_diffusion2d \
  --root $PDEBENCH_ROOT --out data/pdebench --limit 50 --samples 500

PYTHONPATH=src python scripts/convert_pdebench_multimodal.py darcy2d \
  --root $PDEBENCH_ROOT --out data/pdebench --limit 50 --samples 500

PYTHONPATH=src python scripts/convert_pdebench_multimodal.py navier_stokes2d \
  --root $PDEBENCH_ROOT --out data/pdebench --limit 50 --samples 500

PYTHONPATH=src python scripts/convert_pdebench_multimodal.py shallow_water2d \
  --root $PDEBENCH_ROOT --out data/pdebench --limit 50 --samples 500
```

**Expected Output Structure**:
```
data/pdebench/
├── burgers1d_train.h5          (~2-5 GB)
├── burgers1d_val.h5            (~500 MB)
├── burgers1d_test.h5           (~500 MB)
├── advection1d_train.h5        (~2-5 GB)
├── reaction_diffusion2d_train.h5 (~10-20 GB)
└── ... (per-task HDF5 files)
```

### **1.2 Convert Mesh/Particle Tasks to Zarr**
```bash
# Mesh tasks
PYTHONPATH=src python scripts/convert_pdebench_multimodal.py darcy2d_mesh \
  --root data/pdebench/raw --out data/pdebench --limit 20

# Particle tasks
PYTHONPATH=src python scripts/convert_pdebench_multimodal.py particles_advect \
  --root data/pdebench/raw --out data/pdebench --limit 20
```

**Expected Outputs**:
```
data/pdebench/
├── burgers1d_train.h5          (~2-5 GB)
├── burgers1d_val.h5            (~500 MB)
├── burgers1d_test.h5           (~500 MB)
├── advection1d_train.h5        (~2-5 GB)
├── reaction_diffusion2d_train.h5 (~10-20 GB)
├── darcy2d_train.h5            (~5-10 GB)
├── navier_stokes2d_train.h5    (~20-50 GB)
├── darcy2d_mesh_train.zarr/    (~1-2 GB)
└── particles_advect_train.zarr/ (~500 MB - 1 GB)
```

---

## **Stage 2: B2 Cloud Storage Upload & Organization**

### **2.1 B2 Bucket Setup**

UPS uses **Backblaze B2** cloud storage for training data distribution to remote instances (VastAI, Vultr). This approach provides:
- **Fast downloads**: Direct S3-compatible access via rclone
- **Cost-effective storage**: ~$6/TB/month (vs WandB artifacts)
- **Parallel downloads**: Instances can download multiple files concurrently
- **Versioning**: Easy dataset version management

**B2 Credentials** (from `scripts/vast_launch.py:163-170`):
```bash
# Set B2 credentials as environment variables
export B2_KEY_ID=your_key_id
export B2_APP_KEY=your_application_key
export B2_S3_ENDPOINT=s3.us-west-002.backblazeb2.com
export B2_S3_REGION=us-west-002
export B2_BUCKET=pdebench  # Your B2 bucket name
```

**Configure rclone for B2**:
```bash
# Create rclone config for B2
cat >> ~/.config/rclone/rclone.conf << EOF
[B2TRAIN]
type = s3
provider = Other
access_key_id = $B2_KEY_ID
secret_access_key = $B2_APP_KEY
endpoint = $B2_S3_ENDPOINT
region = $B2_S3_REGION
acl = private
no_check_bucket = true
EOF
```

### **2.2 Organize B2 Bucket Structure**

**Recommended Bucket Organization**:
```
B2TRAIN:pdebench/
├── full/                      # Full datasets (for production)
│   ├── burgers1d/
│   │   ├── burgers1d_train_000.h5
│   │   ├── burgers1d_val.h5
│   │   └── burgers1d_test.h5
│   ├── advection1d/
│   │   └── ...
│   ├── darcy2d/
│   └── ... (other tasks)
├── subset/                    # Reduced datasets (for testing)
│   ├── burgers1d_small/
│   │   ├── burgers1d_train_small.h5  (~500 MB, 100 samples)
│   │   └── ...
│   └── ...
└── mesh_particle/             # Non-grid datasets
    ├── darcy2d_mesh/
    │   └── darcy2d_mesh_train.zarr.tar.gz
    └── particles_advect/
        └── particles_advect_train.zarr.tar.gz
```

**Alternative Organization** (from `scripts/vast_launch.py:178`):
```
B2TRAIN:PDEbench/pdebench/     # Alternative path structure
├── burgers1d_full_v1/
│   ├── burgers1d_val.h5
│   └── burgers1d_test.h5
└── ...
```

### **2.3 Upload Datasets to B2**

**Upload Converted Datasets**:
```bash
# Upload burgers1d dataset
rclone copy data/pdebench/burgers1d_train_000.h5 \
  B2TRAIN:pdebench/full/burgers1d/ --progress

rclone copy data/pdebench/burgers1d_val.h5 \
  B2TRAIN:pdebench/full/burgers1d/ --progress

rclone copy data/pdebench/burgers1d_test.h5 \
  B2TRAIN:pdebench/full/burgers1d/ --progress

# Upload advection1d dataset
rclone copy data/pdebench/advection1d_train.h5 \
  B2TRAIN:pdebench/full/advection1d/ --progress

# Upload all 2D tasks
rclone copy data/pdebench/reaction_diffusion2d_train.h5 \
  B2TRAIN:pdebench/full/reaction_diffusion2d/ --progress

rclone copy data/pdebench/darcy2d_train.h5 \
  B2TRAIN:pdebench/full/darcy2d/ --progress

# For mesh/particle Zarr stores, create tarballs first
cd data/pdebench
tar -czf darcy2d_mesh_train.zarr.tar.gz darcy2d_mesh_train.zarr
tar -czf particles_advect_train.zarr.tar.gz particles_advect_train.zarr

# Upload mesh/particle datasets
rclone copy darcy2d_mesh_train.zarr.tar.gz \
  B2TRAIN:pdebench/mesh_particle/darcy2d_mesh/ --progress

rclone copy particles_advect_train.zarr.tar.gz \
  B2TRAIN:pdebench/mesh_particle/particles_advect/ --progress
```

**Upload Optimization**:
```bash
# Use parallel transfers for faster uploads
rclone copy data/pdebench/ B2TRAIN:pdebench/full/ \
  --include "burgers1d*.h5" \
  --include "advection1d*.h5" \
  --transfers 8 \
  --progress

# Set lifecycle policies (optional, via B2 web UI):
# - Delete incomplete uploads after 7 days
# - Transition old versions to cheaper storage after 30 days
```

### **2.4 Verify B2 Uploads**

```bash
# List files in B2 bucket
rclone ls B2TRAIN:pdebench/full/burgers1d/

# Check file sizes match local
rclone size B2TRAIN:pdebench/full/burgers1d/

# Download and verify checksum
rclone copy B2TRAIN:pdebench/full/burgers1d/burgers1d_train_000.h5 /tmp/verify/ --progress
md5sum /tmp/verify/burgers1d_train_000.h5
md5sum data/pdebench/burgers1d_train_000.h5  # Should match
```

---

## **Stage 3: Remote Instance Data Ingestion (VastAI / Vultr)**

### **3.1 VastAI Data Download Strategy**

The VastAI onstart script (from `scripts/vast_launch.py:172-191`) uses **parallel rclone downloads** to fetch training data:

**Current Implementation** (Burgers1D only):
```bash
# Configure rclone for B2 (auto-generated in onstart script)
export RCLONE_CONFIG_B2TRAIN_TYPE=s3
export RCLONE_CONFIG_B2TRAIN_PROVIDER=Other
export RCLONE_CONFIG_B2TRAIN_ACCESS_KEY_ID="$B2_KEY_ID"
export RCLONE_CONFIG_B2TRAIN_SECRET_ACCESS_KEY="$B2_APP_KEY"
export RCLONE_CONFIG_B2TRAIN_ENDPOINT="$B2_S3_ENDPOINT"
export RCLONE_CONFIG_B2TRAIN_REGION="$B2_S3_REGION"

mkdir -p data/pdebench

# Parallel downloads (backgrounded with &)
if [ ! -f data/pdebench/burgers1d_train_000.h5 ]; then
  rclone copy B2TRAIN:pdebench/full/burgers1d/burgers1d_train_000.h5 \
    data/pdebench/ --progress &
fi

if [ ! -f data/pdebench/burgers1d_val.h5 ] && [ ! -f data/pdebench/burgers1d_valid.h5 ]; then
  rclone copy B2TRAIN:PDEbench/pdebench/burgers1d_full_v1/burgers1d_val.h5 \
    data/pdebench/ --progress &
fi

if [ ! -f data/pdebench/burgers1d_test.h5 ]; then
  rclone copy B2TRAIN:PDEbench/pdebench/burgers1d_full_v1/burgers1d_test.h5 \
    data/pdebench/ --progress &
fi

# Wait for all downloads to complete
wait

# Create symlinks for expected filenames
if [ -f data/pdebench/burgers1d_val.h5 ] && [ ! -f data/pdebench/burgers1d_valid.h5 ]; then
  mv -f data/pdebench/burgers1d_val.h5 data/pdebench/burgers1d_valid.h5 || true
fi
ln -sf burgers1d_train_000.h5 data/pdebench/burgers1d_train.h5 || true
ln -sf burgers1d_valid.h5 data/pdebench/burgers1d_val.h5 || true
```

**Performance**:
- Parallel downloads: 3 files simultaneously (~3-5 min for burgers1d on A100 instance)
- Conditional downloads: Skip if file already exists (cached from previous runs)
- Onstart overhead: ~3-4 min total (including git clone, pip install, data download)

### **3.2 Multi-Task Data Download Extension**

For multi-task training, modify the onstart script to download multiple tasks:

**Example: 3-Task Download** (burgers1d + advection1d + reaction_diffusion2d):
```bash
# Download all training datasets in parallel
TASKS=("burgers1d" "advection1d" "reaction_diffusion2d")

for task in "${TASKS[@]}"; do
  if [ ! -f data/pdebench/${task}_train.h5 ]; then
    rclone copy B2TRAIN:pdebench/full/${task}/${task}_train.h5 \
      data/pdebench/ --progress &
  fi
  if [ ! -f data/pdebench/${task}_val.h5 ]; then
    rclone copy B2TRAIN:pdebench/full/${task}/${task}_val.h5 \
      data/pdebench/ --progress &
  fi
  if [ ! -f data/pdebench/${task}_test.h5 ]; then
    rclone copy B2TRAIN:pdebench/full/${task}/${task}_test.h5 \
      data/pdebench/ --progress &
  fi
done

# Wait for all downloads
wait
```

**Estimated Download Times** (VastAI A100 instance, ~1-2 Gbps):
- burgers1d (93 GB total): ~2-3 min
- advection1d (47 GB total): ~1-2 min
- reaction_diffusion2d (13 GB total): ~30-60 sec
- **3-task total**: ~5-7 min parallel download

### **3.3 Vultr Data Download Strategy**

Vultr instances use similar rclone-based downloads (from `scripts/vultr_launch.py`):

**Vultr-Specific Considerations**:
- **Network speeds**: Vultr typically offers 1-10 Gbps, similar to VastAI
- **Block storage**: Vultr supports persistent block storage volumes (mount to `/mnt/cache`)
- **Data persistence**: For repeated runs, mount persistent volume with cached datasets

**Vultr Bootstrap Script** (conceptual, based on VastAI pattern):
```bash
#!/bin/bash
# Mount block storage for dataset caching
mkdir -p /mnt/cache/pdebench
ln -sf /mnt/cache/pdebench data/pdebench

# Download if not in cache
if [ ! -f /mnt/cache/pdebench/burgers1d_train.h5 ]; then
  rclone copy B2TRAIN:pdebench/full/burgers1d/ /mnt/cache/pdebench/ \
    --include "burgers1d*.h5" --transfers 4 --progress
fi

# Continue with training...
```

### **3.4 Selective Download Strategy (Large Datasets)**

For massive datasets (2d_cfd: 551GB, ns_incom: 2.3TB), use **selective downloading**:

**Option A: Shard-Based Download**
```bash
# Download only specific shards for testing
rclone copy B2TRAIN:pdebench/full/ns_incom/ data/pdebench/ \
  --include "ns_incom_train_000.h5" \  # First shard only
  --progress
```

**Option B: Subset Datasets**
```bash
# Use pre-created subset for rapid iteration
rclone copy B2TRAIN:pdebench/subset/ns_incom_small/ data/pdebench/ \
  --progress  # ~5 GB instead of 2.3 TB
```

**Option C: On-Demand Streaming** (Future Work)
- Stream data directly from B2 using HDF5 virtual datasets (VDS)
- Requires implementation in `src/ups/data/pdebench.py`
- Trade download time for runtime I/O overhead

---

## **Stage 4: Latent Cache Precomputation & Distribution** ⚡ **OPTIMIZATION CRITICAL**

### **Problem: Current Approach is Highly Inefficient**

**Current Workflow (from `scripts/vast_launch.py:224`)**:
```bash
# ❌ INEFFICIENT: Recomputes latent cache on EVERY remote training run
echo "Precomputing latent caches…"
PYTHONPATH=src python scripts/precompute_latent_cache.py --config $config ...
```

**Inefficiency Analysis**:
- **Wasted GPU time**: 2-4 hours per run (encoding the same data repeatedly)
- **Wasted bandwidth**: Downloads raw data (~150 GB for 3-task) instead of caches (~50 GB)
- **Slow training startup**: Total overhead ~2.5-4.5 hours before training begins
- **Cost**: ~$3-8 wasted per run @ $1.89/hr (A100)

### **4.1 Optimal Strategy: Pre-Compute Once, Upload to B2, Reuse Forever**

**Recommended Workflow**:

#### **Step 1: Precompute Latent Caches Locally (ONE TIME)**

```bash
# Run on local GPU or dedicated instance (do this ONCE per encoder version)
cd /path/to/universal_simulator

# Ensure you have the encoder from the golden config
# This must match the encoder architecture in configs/train_burgers_upt_full.yaml

# Precompute caches for all tasks
PYTHONPATH=src python scripts/precompute_latent_cache.py \
  --tasks burgers1d advection1d reaction_diffusion2d darcy2d navier_stokes2d \
  --splits train val test \
  --cache-dir data/latent_cache_upt_phase4_128d \
  --latent-dim 128 \
  --latent-len 128 \
  --patch-size 1 \
  --num-workers 8 \
  --pin-memory \
  --cache-dtype float16 \
  --device cuda

# Expected time: ~2-4 hours for 5 tasks (A100 GPU) - BUT ONLY ONCE!
# Expected cache size: ~50-100 GB total (float16 compression)
```

**Cache Organization**:
```
data/latent_cache_upt_phase4_128d/
├── burgers1d_train/
│   ├── sample_00000.pt       # [128, 128] float16 latent tensor
│   ├── sample_00001.pt
│   └── ...
├── burgers1d_val/
├── burgers1d_test/
├── advection1d_train/
├── reaction_diffusion2d_train/
└── ... (per-task, per-split directories)
```

#### **Step 2: Upload Latent Caches to B2**

```bash
# Create versioned cache bucket path
# Version by encoder architecture to allow multiple cache versions
CACHE_VERSION="upt_phase4_128d_128tok"

# Upload entire latent cache directory to B2
rclone copy data/latent_cache_upt_phase4_128d/ \
  B2TRAIN:pdebench/latent_caches/$CACHE_VERSION/ \
  --transfers 8 \
  --progress

# Expected upload time: ~30-60 min for 50-100 GB (depends on connection)
# Expected B2 storage cost: ~$0.30-0.60/month for 50-100 GB
```

**B2 Latent Cache Structure**:
```
B2TRAIN:pdebench/latent_caches/
├── upt_phase4_128d_128tok/          # Current version
│   ├── burgers1d_train/
│   ├── burgers1d_val/
│   ├── advection1d_train/
│   └── ...
├── upt_phase3_64d_128tok/           # Previous version (deprecated)
│   └── ...
└── README.md                         # Cache version changelog
```

#### **Step 3: Update VastAI/Vultr Onstart Script to Download Pre-Computed Caches**

**Modified Onstart Script** (replace `scripts/vast_launch.py:172-224`):

```bash
#!/bin/bash
# ... (existing setup code) ...

mkdir -p data/latent_cache

# ✅ OPTIMIZED: Download pre-computed latent caches from B2 (MUCH FASTER)
CACHE_VERSION="upt_phase4_128d_128tok"

# Download latent caches for all tasks (parallel download)
TASKS=("burgers1d" "advection1d" "reaction_diffusion2d")
for task in "${TASKS[@]}"; do
  for split in train val test; do
    # Check if cache already exists (from previous run on same instance)
    if [ ! -d "data/latent_cache/${task}_${split}" ]; then
      rclone copy B2TRAIN:pdebench/latent_caches/$CACHE_VERSION/${task}_${split}/ \
        data/latent_cache/${task}_${split}/ \
        --transfers 4 --progress &
    fi
  done
done

# Wait for all cache downloads to complete
wait

# ✅ SKIP latent cache precomputation entirely!
# The caches are already downloaded from B2

# Verify cache integrity (optional but recommended)
echo "Verifying latent cache integrity..."
python -c "
from pathlib import Path
cache_dir = Path('data/latent_cache')
for task_dir in cache_dir.glob('*_train'):
    sample_count = len(list(task_dir.glob('sample_*.pt')))
    print(f'✓ {task_dir.name}: {sample_count} samples')
"

# Continue with training...
export WANDB_MODE=online
python scripts/run_fast_to_sota.py --train-config $config ...
```

### **4.2 Performance Comparison**

| Approach | Data Download | Cache Compute | Total Overhead | Cost (A100 @ $1.89/hr) |
|----------|---------------|---------------|----------------|------------------------|
| **❌ Current** (raw data + recompute) | 5-7 min | **2-4 hours** | **~2.5-4.5 hours** | **~$5-8** |
| **✅ Optimized** (pre-computed caches) | **15-30 min** | **0 min** | **~15-30 min** | **~$0.50-1.00** |
| **Savings** | -70% data size | **-100% compute** | **~90% faster** | **~85% cost reduction** |

**Why This Works**:
- **Latent caches are ~50 GB** vs raw data ~150 GB (3-task)
- **No encoding overhead**: Skip 2-4 hours of GPU compute
- **Faster downloads**: Smaller file size, parallel rclone transfers
- **Reusable across runs**: Upload once, download many times

### **4.3 Cache Versioning Strategy**

**Cache Version Naming Convention**:
```
{architecture}_{latent_dim}d_{latent_tokens}tok
```

Examples:
- `upt_phase4_128d_128tok` - UPT Phase 4 golden config
- `upt_phase3_64d_128tok` - UPT Phase 3 config (deprecated)
- `upt_phase4_128d_192tok` - Experimental with more tokens

**When to Recompute Caches**:
1. **Encoder architecture changes** (e.g., Phase 3 → Phase 4)
2. **Latent dimensions change** (e.g., 64d → 128d)
3. **Latent token count changes** (e.g., 128 → 192 tokens)
4. **Dataset updates** (new PDEBench release, preprocessing changes)

**Cache Invalidation**:
- Tag old caches as deprecated in B2 (rename to `deprecated_upt_phase3_64d_128tok/`)
- Keep for 30 days for rollback, then delete
- Update `CACHE_VERSION` in launch scripts

### **4.4 Alternative: WandB Artifacts for Latent Caches**

**Pros**:
- Integrated with training pipeline
- Automatic versioning and lineage tracking
- Downloadable via `wandb artifact get`

**Cons**:
- **Slower downloads** than B2 (WandB throttling)
- **Storage costs**: WandB charges for artifact storage
- **Size limits**: May hit quota with large caches (50-100 GB per version)

**Recommendation**: Use **B2 for primary cache storage**, WandB for small experimental caches only.

### **4.5 Where to Precompute Caches: M4 Mac vs Remote GPU**

You have two options for the one-time cache precomputation:

#### **Option A: M4 Mac (Local)**

**Hardware Specs**:
- Apple M4 chip with GPU cores (10-core or 16-core GPU)
- Unified memory (16-48 GB depending on model)
- PyTorch MPS (Metal Performance Shaders) backend

**Setup**:
```bash
# Enable MPS backend for M4 Mac
PYTHONPATH=src python scripts/precompute_latent_cache.py \
  --tasks burgers1d advection1d reaction_diffusion2d \
  --splits train val test \
  --cache-dir data/latent_cache_upt_phase4_128d \
  --latent-dim 128 --latent-len 128 \
  --device mps \
  --batch-size 4 \
  --num-workers 4 \
  --cache-dtype float16
```

**Pros**:
- ✅ **Zero cost** (use existing hardware)
- ✅ **No data transfer needed**: Raw datasets already local (or download once)
- ✅ **Can run overnight**: No hourly billing concerns
- ✅ **Full control**: Pause/resume at will
- ✅ **Iterative development**: Easy to test encoder changes locally

**Cons**:
- ❌ **Slower performance**: M4 GPU ~10-20× slower than A100 for large models
  - **Estimated time**: 20-40+ hours for 5 tasks (vs 2-4 hours on A100)
- ❌ **MPS compatibility**: PyTorch MPS backend may have edge cases/bugs
  - Some operations not yet optimized for Metal
  - Potential numerical precision differences vs CUDA
- ❌ **Memory constraints**: Unified memory shared between CPU/GPU
  - May need smaller batch sizes (--batch-size 2-4)
- ❌ **Encoder architecture dependency**: Caches computed on M4 must match training encoders
  - If encoder uses CUDA-specific ops, may fail on MPS

**Best For**:
- Small datasets (1-2 tasks, testing)
- Overnight/weekend runs where time doesn't matter
- Tight budget constraints

---

#### **Option B: Remote GPU Instance (VastAI/Vultr)**

**Hardware Specs**:
- NVIDIA A100 (40GB or 80GB)
- CUDA 11.8+
- ~1-2 Gbps network for B2 uploads

**Setup**:
```bash
# Launch dedicated cache precompute instance
vastai search offers 'gpu_name=A100 reliability > 0.95' --order 'dph_total'

# SSH into instance or run via onstart script
PYTHONPATH=src python scripts/precompute_latent_cache.py \
  --tasks burgers1d advection1d reaction_diffusion2d darcy2d navier_stokes2d \
  --splits train val test \
  --cache-dir data/latent_cache_upt_phase4_128d \
  --latent-dim 128 --latent-len 128 \
  --device cuda \
  --batch-size 16 \
  --num-workers 8 \
  --cache-dtype float16
```

**Pros**:
- ✅ **Fast performance**: A100 completes 5-task cache in **2-4 hours**
- ✅ **CUDA compatibility**: Guaranteed compatibility with training pipeline
- ✅ **No encoder mismatch risk**: Same hardware/backend as training runs
- ✅ **High throughput**: Can cache all 11 PDEBench tasks in 8-12 hours
- ✅ **Efficient for large-scale**: Once GPU is rented, maximize utilization

**Cons**:
- ❌ **Hourly cost**: ~$1.89/hr for A100 (40GB)
  - **Total cost**: ~$4-8 for 5-task cache (2-4 hours)
  - **Full PDEBench**: ~$15-25 for all 11 tasks (8-12 hours)
- ❌ **Data transfer overhead**:
  - Download raw datasets from B2/PDEBench: **1-2 hours** (150+ GB)
  - Upload computed caches back to B2: **30-60 min** (50-100 GB)
  - Total instance time: **4-6 hours** including transfers
- ❌ **Setup complexity**: Need to provision instance, configure B2, manage SSH
- ❌ **Time pressure**: Hourly billing creates urgency

**Best For**:
- Production cache generation (all tasks)
- Time-sensitive workflows (need caches quickly)
- Large-scale multi-task training

---

#### **Cost-Benefit Analysis**

| Metric | M4 Mac (Local) | Remote GPU (A100) |
|--------|----------------|-------------------|
| **Time (5 tasks)** | 20-40 hours | 2-4 hours |
| **Cost (5 tasks)** | $0 | ~$4-8 |
| **Time (11 tasks)** | 60-100 hours | 8-12 hours |
| **Cost (11 tasks)** | $0 | ~$15-25 |
| **Data transfer** | None (local) | ~2-3 hours |
| **Setup complexity** | Low | Medium |
| **CUDA compatibility** | ⚠️ MPS differences | ✅ Identical to training |
| **Iterative testing** | ✅ Easy | ❌ Costly |

---

#### **Hybrid Approach (Recommended)**

**Best strategy**: Use **both** for different purposes:

1. **M4 Mac for Development**:
   ```bash
   # Cache 1-2 tasks for local testing/iteration
   PYTHONPATH=src python scripts/precompute_latent_cache.py \
     --tasks burgers1d \
     --splits train \
     --cache-dir data/latent_cache_test \
     --device mps \
     --batch-size 2

   # Run overnight, verify encoder works correctly
   ```

2. **Remote GPU for Production**:
   ```bash
   # After encoder validated on M4, cache all tasks on A100
   # Rent A100 for 6-8 hours, cache all 11 PDEBench tasks
   # Upload to B2, use for all future training runs
   ```

**Workflow**:
```
Day 1 (M4 Mac):
  → Download burgers1d dataset
  → Precompute small cache overnight (8-12 hours)
  → Verify encoder works, test cache integrity
  → Fix any issues

Day 2 (Remote GPU):
  → Rent A100 instance
  → Precompute all 11 task caches (8-12 hours)
  → Upload to B2
  → Destroy instance
  → Total cost: ~$15-25

Future:
  → All training runs download pre-computed caches
  → Save $5-8 per run × dozens of runs = hundreds saved
```

---

#### **Recommendation Based on Use Case**

**If you're doing iterative encoder development**:
→ Use **M4 Mac** for testing (1-2 tasks), then **remote GPU** for full production caches

**If you need caches immediately for training**:
→ Use **remote GPU** (A100), accept $15-25 one-time cost

**If you have time and want zero cost**:
→ Use **M4 Mac** overnight/weekend runs (20-100 hours total)

**If you're caching all 11 PDEBench tasks**:
→ **Definitely use remote GPU** - M4 would take 3-5 days continuous runtime

---

#### **Implementation: MPS Support for M4 Mac**

The codebase **already supports MPS** via generic `torch.device()`:

```python
# From scripts/precompute_latent_cache.py:365-370
if args.device is not None:
    device = torch.device(args.device)  # ✅ Works with "mps"
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**To use M4 Mac**:
```bash
# Just specify --device mps
python scripts/precompute_latent_cache.py \
  --device mps \
  --tasks burgers1d \
  ...
```

**Verify MPS availability**:
```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Should print:
# MPS available: True
# MPS built: True
```

**Potential MPS Issues**:
- Some PyTorch ops not yet optimized for Metal (slower than expected)
- Numerical precision differences (rare, but possible)
- Memory management differences (unified memory vs dedicated VRAM)

**Mitigation**:
- Test small cache first (burgers1d, train split only)
- Compare cache tensors between MPS and CUDA (if available)
- Monitor memory usage with Activity Monitor

---

### **4.6 Implementation Checklist**

**For M4 Mac Approach**:
- [ ] Verify PyTorch MPS backend: `torch.backends.mps.is_available()`
- [ ] Download burgers1d dataset locally
- [ ] Precompute test cache: `--device mps --tasks burgers1d --splits train`
- [ ] Verify cache integrity (check tensor shapes, dtypes)
- [ ] Run overnight for full 5-task cache
- [ ] Upload to B2: `rclone copy data/latent_cache_upt_phase4_128d/ B2TRAIN:...`

**For Remote GPU Approach**:
- [ ] Search for A100 instance: `vastai search offers 'gpu_name=A100'`
- [ ] Launch instance with onstart script
- [ ] Download raw datasets from B2 or PDEBench
- [ ] Precompute all task caches (8-12 hours for 11 tasks)
- [ ] Upload caches to B2
- [ ] Destroy instance
- [ ] Measure total time/cost

**For Both**:
- [ ] Update `scripts/vast_launch.py` to download pre-computed caches
- [ ] Add cache integrity verification in onstart script
- [ ] Document cache version in `docs/data_artifacts.md`
- [ ] Test cache download + training on VastAI instance
- [ ] Measure actual time/cost savings vs. current approach

---

## **Stage 5: Multi-Task Training Configs**

### **5.1 Three-Task Starter Config** (`configs/train_pdebench_3task_upt.yaml`)
```yaml
# Based on configs/train_burgers_upt_full.yaml (UPT Phase 4 golden config)
# Start with 3 similar 1D/2D tasks for validation

seed: 42
deterministic: false
benchmark: true

data:
  task: [burgers1d, advection1d, reaction_diffusion2d]  # NEW: Multi-task list
  split: train
  root: data/pdebench
  patch_size: 1

  download:
    test_val_datasets: burgers1d_full_v1,advection1d_full_v1,reaction_diffusion2d_full_v1

latent:
  dim: 128          # Match UPT Phase 4 golden config
  tokens: 128       # Match UPT Phase 4 golden config

operator:
  architecture_type: pdet_stack    # Pure transformer (from Phase 4)
  pdet:
    input_dim: 128
    hidden_dim: 384  # Increased from 256 for multi-task capacity (3x latent_dim)
    depth: 10        # Increased from 8 for multi-task
    num_heads: 12    # Increased from 8 for multi-task
    attention_type: standard
    qk_norm: true
    mlp_ratio: 4.0
    drop_path: 0.15  # Increased regularization
    dropout: 0.0

diffusion:
  latent_dim: 128   # Match latent.dim
  hidden_dim: 384   # Match operator.pdet.hidden_dim

training:
  batch_size: 8     # Reduced due to larger model
  accum_steps: 5    # Effective batch = 40
  time_stride: 2
  dt: 0.1
  patience: 10

  num_workers: 8
  use_parallel_encoding: true
  pin_memory: true
  prefetch_factor: 4

  amp: true
  compile: false
  grad_clip: null
  ema_decay: 0.999

  # UPT Inverse Losses (from Phase 4)
  lambda_inv_enc: 0.01
  lambda_inv_dec: 0.01
  use_inverse_losses: true
  inverse_loss_frequency: 1
  inverse_loss_warmup_epochs: 5
  inverse_loss_max_weight: 0.05

  # Query-Based Training (Phase 4.1)
  query_sampling:
    enabled: true
    num_queries: 2048
    strategy: uniform

  # Physics Priors + Latent Regularization (Phase 4.2 + 4.3)
  physics_priors:
    enabled: true
    lambda_divergence: 0.0
    lambda_conservation: 0.0
    lambda_boundary: 0.05
    lambda_positivity: 0.0
    lambda_latent_norm: 1.0e-4
    lambda_latent_diversity: 1.0e-4

  lambda_spectral: 0.05

  # NEW: Per-task metric logging (implement in train.py)
  log_per_task_metrics: true

stages:
  operator:
    epochs: 40      # Same as Phase 4 golden config

    optimizer:
      name: muon_hybrid
      lr: 1.4e-3
      weight_decay: 0.03
      muon_momentum: 0.95
      muon_ns_steps: 5
      muon_backend: auto
      betas: [0.9, 0.999]

  diff_residual:
    epochs: 15
    patience: 5
    optimizer:
      name: muon_hybrid
      lr: 3.0e-4
      weight_decay: 0.01
      muon_momentum: 0.95
      muon_ns_steps: 5

  consistency_distill:
    epochs: 10
    patience: 5
    optimizer:
      name: muon_hybrid
      lr: 3.0e-4
      weight_decay: 0.01
```

**4.2 Five-Task Full Grid Config** (`configs/train_pdebench_5task_grid.yaml`)
```yaml
data:
  task: [burgers1d, advection1d, reaction_diffusion2d, darcy2d, navier_stokes2d]
  # ... (rest similar to 3-task, with capacity scaling)

latent:
  dim: 96          # Scale up for 5 tasks
  tokens: 256      # Scale up for better capacity

operator:
  pdet:
    hidden_dim: 384
    depth: 12
    num_heads: 12
```

**4.3 Mixed-Modality Config** (`configs/train_pdebench_multidiscretization.yaml`)
```yaml
data:
  task: [burgers1d, darcy2d, darcy2d_mesh]  # Grid + Mesh
  # Add mesh-specific encoder params
  supernodes: 2048
  message_passing_steps: 3

# ... (architecture similar to 3-task)
```

---

## **Stage 6: Training Execution Strategy**

### **6.1 Curriculum Learning Schedule** (Implement in `src/ups/training/loop_train.py`)

```python
# Proposed curriculum stages for multi-dataset training
CURRICULUM_SCHEDULE = {
    "stage_1": {
        "epochs": [0, 10],
        "tasks": ["burgers1d", "advection1d"],  # Simple 1D
        "task_weights": {"burgers1d": 0.5, "advection1d": 0.5},
        "rollout_horizon": 1,
    },
    "stage_2": {
        "epochs": [10, 20],
        "tasks": ["burgers1d", "advection1d", "reaction_diffusion2d"],  # Add 2D
        "task_weights": {"burgers1d": 0.33, "advection1d": 0.33, "reaction_diffusion2d": 0.34},
        "rollout_horizon": 2,
    },
    "stage_3": {
        "epochs": [20, 30],
        "tasks": ["burgers1d", "advection1d", "reaction_diffusion2d", "darcy2d"],
        "task_weights": "inverse_nrmse",  # Weight by inverse performance
        "rollout_horizon": 4,
    },
}
```

### **6.2 Training Command (Local)**
```bash
# 3-task multi-dataset training
python scripts/train.py \
  --config configs/train_pdebench_3task_upt.yaml \
  --stage all

# Expected time: ~4-6 hours on A100 (with latent cache)
# Expected NRMSE: ~0.05-0.08 (aggregate), per-task breakdown in WandB
```

### **6.3 Training Command (VastAI Remote)**
```bash
# Use VastAI launcher with multi-task config
python scripts/vast_launch.py launch \
  --config configs/train_pdebench_3task_upt.yaml \
  --auto-shutdown

# Estimated cost: ~$5-10 @ $1.89/hr A100 (4-6 hours)
```

---

## **Stage 7: Evaluation & Metrics**

### **7.1 Per-Task Evaluation** (Extend `scripts/evaluate.py`)
```bash
# Evaluate each task separately after multi-task training
python scripts/evaluate.py \
  --checkpoint checkpoints/op_latest.ckpt \
  --config configs/train_pdebench_3task_upt.yaml \
  --task burgers1d \
  --output reports/eval_burgers1d.json

python scripts/evaluate.py \
  --checkpoint checkpoints/op_latest.ckpt \
  --config configs/train_pdebench_3task_upt.yaml \
  --task advection1d \
  --output reports/eval_advection1d.json

# ... repeat for all tasks
```

### **7.2 Multi-Task Leaderboard** (Extend `src/ups/utils/leaderboard.py`)
```csv
run_id,label,burgers1d_nrmse,advection1d_nrmse,reaction_diffusion2d_nrmse,aggregate_nrmse,tokens,latent_dim
run_20251107_001,3task_upt,0.055,0.062,0.071,0.063,128,64
run_20251107_002,5task_upt,0.058,0.065,0.068,0.064,256,96
```

## Summary

The Universal Physics Stack is well-positioned to scale to multi-dataset PDEBench training. The existing architecture supports:
- **Discretization-agnostic I/O**: GridEncoder (`src/ups/io/enc_grid.py`), MeshParticleEncoder (`src/ups/io/enc_mesh_particle.py`), and universal decoder
- **Multi-task loader infrastructure**: `build_latent_pair_loader` supports `ConcatDataset` for mixing tasks (`src/ups/data/latent_pairs.py:738-748`)
- **Latent caching**: Parallel encoding and RAM-preloaded caching for efficiency (`src/ups/data/parallel_cache.py`)
- **Training pipeline**: Multi-stage training (operator → diffusion → consistency → steady prior) via `scripts/run_fast_to_sota.py`

**Remaining Gaps** (Corrected):
- ✅ Dataset generation scripts exist (`scripts/prepare_data.py`, `scripts/convert_pdebench_multimodal.py`)
- ⚠️ Limited task specifications in `TASK_SPECS` (10/15+ tasks defined, need to add missing 2D/3D variants)
- ❌ No multi-task training configurations (need to create `train_pdebench_3task_upt.yaml`, etc.)
- ⚠️ UPT inverse losses implemented but not tested on multi-dataset scenarios
- ❌ Curriculum learning not implemented (need to extend `src/ups/training/loop_train.py`)
- ❌ Per-task metric logging not implemented (need to extend `scripts/train.py`)

## Detailed Findings

### 1. Current PDEBench Integration Architecture

**Single-Task Grid Training (Production-Ready)**:
- `src/ups/data/pdebench.py:23-37` — `TASK_SPECS` defines 10 tasks:
  - 1D: `burgers1d`, `advection1d`
  - 2D: `darcy2d`, `navier_stokes2d`, `allen_cahn2d`, `cahn_hilliard2d`, `reaction_diffusion2d`, `shallow_water2d`
  - Mesh/Particle: `darcy2d_mesh`, `particles_advect`
- `src/ups/data/pdebench.py:72-158` — `PDEBenchDataset` loads HDF5 shards, concatenates samples, applies normalization, aggregates params/BC metadata
- `src/ups/data/latent_pairs.py:257-413` — `GridLatentPairDataset` wraps PDEBench data, performs encoder calls, handles cache hits/misses, prepares conditioning tensors

**Multi-Task Infrastructure (Partially Implemented)**:
- `src/ups/data/latent_pairs.py:738-748` — `build_latent_pair_loader` supports task lists:
  ```python
  tasks = data_cfg.get("task")
  task_list = list(tasks) if isinstance(tasks, (list, tuple)) else [tasks]
  ```
- `src/ups/data/latent_pairs.py:749-801` — `_make_grid_latent_dataset` factory creates latent datasets per task
- `src/ups/data/latent_pairs.py:803-826` — `_make_graph_latent_dataset` factory handles mesh/particle tasks
- `src/ups/data/latent_pairs.py:828-850` — Loop through task_list, create datasets, concat with `ConcatDataset`

**Latent Caching & Scaling Optimizations**:
- `src/ups/data/parallel_cache.py:33-88` — `RawFieldDataset` defers encoding to main process (multi-worker safe)
- `src/ups/data/parallel_cache.py:90-167` — `PreloadedCacheDataset` loads entire cache into RAM for ~90%+ GPU utilization
- `src/ups/data/parallel_cache.py:258-310` — `build_parallel_latent_loader` wires raw dataset + GPU encoding collate_fn
- `src/ups/data/latent_pairs.py:764-787` — Cache completeness checks, RAM availability checks, automatic fallback

### 2. UPT Architecture Integration

**Universal Physics Transformer Components** (from `UPT_docs/UPT_Arch_Train_Scaling.md`):
- **Encoder**: Grid/mesh/particle → supernodes → transformer blocks → perceiver pooling → fixed latent tokens
- **Approximator**: Transformer in latent space, steps forward by Δt (autoregressive rollouts)
- **Decoder**: Query-based (arbitrary positions), cross-attention to latent tokens

**Current UPT Implementation**:
- `src/ups/io/enc_grid.py` — GridEncoder with patch-based tokenization, per-channel streams
- `src/ups/io/enc_mesh_particle.py` — MeshParticleEncoder with message passing, supernode pooling, perceiver reduction
- `src/ups/io/decoder_anypoint.py` — Query-based decoder with perceiver cross-attention
- `src/ups/models/latent_operator.py` — PDE-Transformer backbone (approximator)

**Inverse Losses for Latent Invertibility** (Phase 1.5 UPT):
- `src/ups/training/losses.py:25-60` — `inverse_encoding_loss`: Ensures latent can decode back to input fields
- `src/ups/training/losses.py:63-99` — `inverse_decoding_loss`: Ensures decoded fields can re-encode to latent
- `src/ups/training/losses.py:134-165` — Curriculum weight scheduling (warmup_epochs=15, max_weight=0.05)
- **Status**: Implemented in codebase, tested on Burgers1D, **not yet tested on multi-dataset scenarios**

### 3. PDEBench Dataset Catalog

**Available Datasets** (from PDEBench paper & GitHub):

| Category | Tasks | Dimensionality | Modality | Status in UPS |
|----------|-------|----------------|----------|---------------|
| **Basic PDEs** | | | | |
| Advection | 1D | 1D | Grid | ✅ TASK_SPECS defined |
| Burgers | 1D | 1D | Grid | ✅ Production-ready |
| Diffusion-Reaction | 1D, 2D | 1D, 2D | Grid | ✅ 2D in TASK_SPECS |
| Diffusion-Sorption | 1D | 1D | Grid | ⚠️ Not in TASK_SPECS |
| Darcy Flow | 2D | 2D | Grid, Mesh | ✅ Both variants in TASK_SPECS |
| **Advanced PDEs** | | | | |
| Navier-Stokes (Compressible) | 1D, 2D, 3D | 1D, 2D, 3D | Grid | ⚠️ Only 2D in TASK_SPECS |
| Navier-Stokes (Incompressible) | 2D | 2D | Grid | ✅ In TASK_SPECS |
| Shallow Water | 2D | 2D | Grid | ✅ In TASK_SPECS |
| Allen-Cahn | 2D | 2D | Grid | ✅ In TASK_SPECS |
| Cahn-Hilliard | 2D | 2D | Grid | ✅ In TASK_SPECS |

**Data Characteristics** (from PDEBench documentation):
- **Format**: HDF5 for grids, Zarr for mesh/particle
- **Dimensions**: Packed as `[b, t, x1, ..., xd, v]` where b=batch, t=time, x=spatial, v=channels
- **Resolutions**: Configurable via `reduced_resolution` and `reduced_resolution_t` parameters
- **Splits**: Train/test/val with configurable `reduced_batch` factors

**Missing from UPS Task Specs**:
- 1D Diffusion-Sorption
- 1D, 3D Compressible Navier-Stokes
- Additional mesh/particle variants beyond Darcy and advection

### 4. Training Pipeline for Multi-Dataset Scaling

**Current Training Flow** (from `TRAINING_PIPELINE_DOCUMENTATION.md`):

```
Stage 1: Operator Training (25 epochs)
  └─ Load: PDEBench → GridLatentPairDataset
  └─ Model: LatentOperator (PDE-Transformer)
  └─ Loss: L_forward + L_rollout + L_spectral + L_inv_enc + L_inv_dec
  └─ Output: checkpoints/operator.pt

Stage 2: Diffusion Residual (8 epochs)
  └─ Load: Operator (frozen) + same dataset
  └─ Model: DiffusionResidual
  └─ Loss: MSE(diffusion_output, operator_residual)
  └─ Output: checkpoints/diffusion_residual.pt

Stage 3: Consistency Distillation (8 epochs)
  └─ Load: Diffusion (teacher) + same dataset
  └─ Model: DiffusionResidual (student)
  └─ Loss: Distillation MSE
  └─ Output: checkpoints/diffusion_residual.pt (overwrite)

Stage 4: Steady Prior (optional, 0 epochs in golden config)
  └─ Output: checkpoints/steady_prior.pt

Stage 5: Evaluation
  └─ Baseline: Operator + optional Diffusion
  └─ TTC: Test-time conditioning with physics rewards
```

**Fast-to-SOTA Orchestration** (`scripts/run_fast_to_sota.py`):
- Lines 664-684: Validation (config, data, dry-run)
- Lines 693-733: Training with multi-stage progression
- Lines 796-876: Small evaluation (proxy gate)
- Lines 936-1048: Full evaluation (promotion gate)
- Lines 1050-1082: Champion promotion if gates passed

### 5. Scaling Strategy Recommendations

**Reference: UPT Staged Scaling Plan** (`UPT_docs/UPT_Arch_Train_Scaling.md:94-111`):

| Stage | ns (supernodes) | n_latent (tokens) | latent_dim | depth | Purpose |
|-------|-----------------|-------------------|------------|-------|---------|
| S0 | 512 | 256 | 192 | 4 | Plumbing + loss sanity |
| S1 | 1024 | 512 | 256 | 8 | Inverse losses + longer rollouts |
| S2 | 2048 | 512-768 | 384 | 8-12 | Drop-path + increased queries |
| S3 | - | - | - | - | Mixed discretizations/BCs |

**Recommended Multi-Dataset Scaling Path**:

#### **Phase 0: Baseline Validation (Current State)**
- ✅ Burgers1D single-task training working (golden config: 16-dim latent, 32 tokens, NRMSE 0.078)
- ✅ Latent caching operational (float32 cache, parallel encoding, RAM preload)
- ✅ Fast-to-SOTA pipeline validated (train → small eval → full eval → leaderboard)

#### **Phase 1: Restore Missing Tooling** (High Priority)
**Goal**: Enable generation of all PDEBench dataset variants

**Tasks**:
1. **Restore `scripts/prepare_data.py`**:
   - Recreate with dataclass `Config` for mesh/particle generation
   - Implement generators for mesh Poisson, particle advection
   - Required by `tests/unit/test_mesh_loader.py:20`, `tests/unit/test_particles_dataset.py:16`

2. **Restore `scripts/convert_pdebench_multimodal.py`**:
   - Streaming converters for grid → HDF5, mesh → Zarr, particle → Zarr
   - CLI parity with `docs/data_artifacts.md:13` documentation

3. **Update Documentation**:
   - `docs/data_artifacts.md` — Reflect restored scripts
   - `README.md` — Update data preparation workflow

**Success Criteria**:
- `python scripts/prepare_data.py --help` runs without errors
- `python scripts/convert_pdebench_multimodal.py burgers1d --root <tmp> --out <tmp>` completes
- Test mesh/particle datasets can be synthesized for unit tests

**Implementation Complexity**: Medium (scripts were removed, need reconstruction based on test expectations and Zarr/HDF5 schemas)

#### **Phase 2: Extend Grid Task Coverage** (Medium Priority)
**Goal**: Support all grid-based PDEBench tasks

**Tasks**:
1. **Expand `TASK_SPECS` in `src/ups/data/pdebench.py:24-37`**:
   ```python
   TASK_SPECS: Dict[str, PDEBenchSpec] = {
       # 1D tasks
       "burgers1d": PDEBenchSpec(field_key="data"),
       "advection1d": PDEBenchSpec(field_key="data"),
       "diffusion_sorption1d": PDEBenchSpec(field_key="data"),  # NEW
       "compressible_ns1d": PDEBenchSpec(field_key="data"),     # NEW

       # 2D tasks (existing)
       "darcy2d": PDEBenchSpec(field_key="data"),
       "navier_stokes2d": PDEBenchSpec(field_key="data"),
       "allen_cahn2d": PDEBenchSpec(field_key="data"),
       "cahn_hilliard2d": PDEBenchSpec(field_key="data"),
       "reaction_diffusion2d": PDEBenchSpec(field_key="data"),
       "shallow_water2d": PDEBenchSpec(field_key="data"),

       # 3D tasks
       "compressible_ns3d": PDEBenchSpec(field_key="data"),     # NEW

       # Mesh/particle (existing)
       "darcy2d_mesh": PDEBenchSpec(field_key="data", kind="mesh"),
       "particles_advect": PDEBenchSpec(field_key="data", kind="particles"),
   }
   ```

2. **Create Multi-Task Training Config** (`configs/train_pdebench_multitask.yaml`):
   ```yaml
   data:
     task: [burgers1d, advection1d, reaction_diffusion2d]  # Start with 3 similar tasks
     split: train
     root: data/pdebench
     patch_size: 1

   latent:
     dim: 32        # Start with proven Burgers baseline
     tokens: 64     # Increase from 32 for multi-task capacity

   operator:
     pdet:
       input_dim: 32
       hidden_dim: 192  # Increase from 96 for multi-task capacity
       depths: [2, 2, 2]
       group_size: 12
       num_heads: 6

   training:
     batch_size: 8   # Reduce from 12 due to multi-dataset overhead
     ...
   ```

3. **Validate Multi-Task Data Loading**:
   - Extend `tests/unit/test_train_pdebench_loader.py:35` to cover multi-task scenarios
   - Test conditioning tensors are properly broadcast across tasks
   - Verify cache directories are task-specific

**Success Criteria**:
- `python scripts/train.py --config configs/train_pdebench_multitask.yaml --stage operator --epochs 1` completes
- Operator loss converges on multi-task training
- Latent cache populated for all tasks

**Implementation Complexity**: Low-Medium (leverages existing infrastructure, mainly config + test work)

#### **Phase 3: Add Mesh/Particle Support** (Medium-High Priority)
**Goal**: Enable training on heterogeneous discretizations

**Tasks**:
1. **Extend Multi-Task Loader** to handle mixed modalities:
   - Modify `src/ups/data/latent_pairs.py:828-850` to support mixed grid/mesh/particle tasks
   - Ensure conditioning tensors align across modalities
   - Implement unified collate function for heterogeneous batches

2. **Create Mesh/Particle Training Config** (`configs/train_multidiscretization.yaml`):
   ```yaml
   data:
     task: [burgers1d, darcy2d, darcy2d_mesh]  # Mix grid + mesh
     split: train
     root: data/pdebench

   # Use MeshParticleEncoder settings for graph tasks
   data:
     supernodes: 2048       # From UPT reference
     message_passing_steps: 3
     use_coords: true
   ```

3. **Test Mesh/Particle Encoding**:
   - Generate test datasets via restored `scripts/prepare_data.py`
   - Run `pytest tests/unit/test_mesh_loader.py tests/unit/test_particles_dataset.py`
   - Verify latent dimensions match across modalities

**Success Criteria**:
- Mixed grid/mesh/particle training completes without dimension mismatches
- Latent operator processes latent tokens uniformly regardless of source modality
- Conditioning metadata includes discretization type

**Implementation Complexity**: Medium-High (requires careful handling of graph structures, neighbor lists, sparse matrices)

#### **Phase 4: Curriculum Learning Across Datasets** (High Priority)
**Goal**: Optimize multi-dataset training with progressive complexity

**Strategy** (from `UPT_docs/UPT_Arch_Train_Scaling.md:108-111`):

```python
# Curriculum stages for multi-dataset scaling
Stage 1 (Epochs 0-10):
  - Simple 1D tasks (Burgers, Advection)
  - Small Δt, periodic BCs
  - Single-step prediction only

Stage 2 (Epochs 10-20):
  - Add 2D tasks (Reaction-Diffusion, Darcy)
  - Introduce harder BCs (Dirichlet, Neumann)
  - Enable 2-4 step rollouts

Stage 3 (Epochs 20-30):
  - Add mesh/particle tasks
  - Mixed discretizations
  - Full rollout horizon (up to 10 steps)

Stage 4 (Epochs 30+):
  - All tasks mixed
  - Shocks, discontinuities
  - Inverse losses enabled (warmup complete)
```

**Implementation**:
- Modify `src/ups/training/loop_train.py:18-24` — `CurriculumConfig` to support dataset mixing schedules
- Add dataset weighting/sampling strategies (uniform vs. performance-based)
- Log per-task metrics to WandB for curriculum tuning

**Success Criteria**:
- Multi-dataset training converges faster than flat mixing
- Per-task NRMSE metrics improve over curriculum stages
- Final model generalizes to held-out tasks

**Implementation Complexity**: Medium (requires training loop modifications, WandB multi-metric logging)

#### **Phase 5: Hyperparameter Scaling for Multi-Dataset** (Medium Priority)
**Goal**: Determine optimal architecture capacity for multi-dataset training

**Scaling Knobs** (from `UPT_docs/UPT_Arch_Train_Scaling.md:30-56`):

| Parameter | Burgers1D (1-task) | Multi-Task (3-5 tasks) | Full PDEBench (10+ tasks) |
|-----------|-------------------|------------------------|---------------------------|
| `latent.dim` | 16 | 32-48 | 64-96 |
| `latent.tokens` | 32 | 64-128 | 128-256 |
| `operator.hidden_dim` | 96 | 192-256 | 384-512 |
| `operator.depths` | [1, 1, 1] | [2, 2, 2] | [2, 4, 2] (U-shape) |
| `operator.num_heads` | 6 | 8 | 8-12 |
| `training.batch_size` | 12 | 6-8 | 4-6 |

**Sweep Strategy**:
- Use `docs/fast_to_sota_playbook.md` guidance for hyperparameter sweeps
- Start with 3-task subset (Burgers1D, Advection1D, Reaction-Diffusion2D)
- Sweep `latent.dim` ∈ {32, 48, 64} while holding other params fixed
- Once optimal latent_dim found, sweep `latent.tokens` ∈ {64, 128, 256}
- Monitor: NRMSE per task, conservation gaps, BC violations, training time

**Success Criteria**:
- Identify Pareto-optimal configs (accuracy vs. compute)
- Document scaling laws: "For N tasks, latent_dim ≈ 16 + 8×N, tokens ≈ 32×√N"
- Update `configs/train_pdebench_multitask_golden.yaml` with validated settings

**Implementation Complexity**: Low (config sweeps, no code changes)

### 6. Missing Components & Action Items (CORRECTED)

**Status Update**: ✅ = Complete, ⚠️ = Partial, ❌ = Not Started

**Critical (Was Blocking, Now Unblocked)**:
1. ✅ `scripts/prepare_data.py` — EXISTS (mesh/particle test dataset generator implemented)
2. ✅ `scripts/convert_pdebench_multimodal.py` — EXISTS (grid/mesh/particle converter implemented)
3. ❌ Multi-task training configs — Still needed (`train_pdebench_3task_upt.yaml`, etc.)
4. ⚠️ Task specification expansion — Only 10/15+ PDEBench tasks defined (need to add 3D variants)

**Important (For Production Multi-Dataset)**:
5. ❌ Curriculum learning infrastructure — Not implemented (need to extend `src/ups/training/loop_train.py`)
6. ❌ Per-task metric logging — WandB logs aggregate loss only (need to extend `scripts/train.py`)
7. ⚠️ Mixed-modality collate function — Current `latent_pair_collate` works but no explicit heterogeneous batch handling
8. ✅ Multi-dataset cache strategy — Per-task cache directories already work via `build_latent_pair_loader`

**Nice-to-Have (For Research/Optimization)**:
9. ❌ Scaling law documentation — No empirical guidelines for latent_dim vs. number of tasks
10. ❌ Multi-dataset evaluation protocol — Leaderboard supports single-task only
11. ❌ Domain-adversarial alignment — UPT paper optional feature, not implemented

### 7. Implementation Priority Matrix (UPDATED)

| Phase | Priority | Complexity | Dependencies | Estimated Effort | Status |
|-------|----------|------------|--------------|------------------|--------|
| ~~Phase 1: Restore Tooling~~ | ~~HIGH~~ | ~~Medium~~ | ~~None~~ | ~~2-3 days~~ | ✅ **COMPLETE** |
| Phase 2: Extend Grid Tasks | **HIGH** | Low | None | 1-2 days | 🔄 IN PROGRESS |
| Phase 3: Create Multi-Task Configs | **HIGH** | Low | Phase 2 | 1 day | ❌ NOT STARTED |
| Phase 4: Curriculum Learning | **MEDIUM** | Medium | Phase 3 | 2-3 days | ❌ NOT STARTED |
| Phase 5: Per-Task Metrics | **MEDIUM** | Low-Medium | Phase 3 | 1-2 days | ❌ NOT STARTED |
| Phase 6: Hyperparameter Scaling | **HIGH** | Low (sweeps) | Phase 3-5 | 1 week | ❌ NOT STARTED |
| Phase 7: Mesh/Particle Multi-Task | **LOW** | Medium-High | Phase 3-6 | 3-4 days | ❌ NOT STARTED |

**Recommended Execution Order (UPDATED)**:
1. **Week 1**: Phase 2 (extend TASK_SPECS) + Phase 3 (create 3-task config) + Phase 5 (per-task logging)
2. **Week 2**: Phase 6 (hyperparameter sweeps on 3-task) + initial multi-task training validation
3. **Week 3**: Phase 4 (curriculum learning) + scale to 5-task config
4. **Week 4**: Phase 7 (mesh/particle multi-task) + full PDEBench suite + documentation

### 8. Implementation Roadmap with Code Changes

#### **Immediate (Next 1-2 Days)**

**Task 2.1: Extend TASK_SPECS for Missing PDEBench Tasks**
- **File**: `src/ups/data/pdebench.py`
- **Change**: Add missing tasks to `TASK_SPECS` dict
  ```python
  # Add to TASK_SPECS (lines 24-37)
  "diffusion_sorption1d": PDEBenchSpec(field_key="data"),
  "compressible_ns1d": PDEBenchSpec(field_key="data"),
  "compressible_ns2d": PDEBenchSpec(field_key="data"),
  "compressible_ns3d": PDEBenchSpec(field_key="data"),
  "diffusion_reaction1d": PDEBenchSpec(field_key="data"),
  ```
- **Test**: `pytest tests/unit/test_pdebench.py -v`
- **Estimated Time**: 30 minutes

**Task 2.2: Update Conversion Script Task Definitions**
- **File**: `scripts/convert_pdebench_multimodal.py`
- **Change**: Add missing tasks to `DEFAULT_TASKS` dict (lines 48-73)
- **Estimated Time**: 30 minutes

**Task 3.1: Create 3-Task Multi-Dataset Config**
- **File**: `configs/train_pdebench_3task_upt.yaml` (new)
- **Template**: Based on `configs/train_burgers_upt_128tokens_pure.yaml`
- **Key Changes**:
  - `data.task: [burgers1d, advection1d, reaction_diffusion2d]`
  - `latent.dim: 64`, `latent.tokens: 128` (proven from golden config)
  - `operator.pdet.hidden_dim: 256` (scale up from 192)
  - `operator.pdet.depth: 10` (scale up from 8)
  - `training.batch_size: 6`, `training.accum_steps: 8`
  - `training.log_per_task_metrics: true` (new flag)
- **Estimated Time**: 1 hour

**Task 5.1: Implement Per-Task Metric Logging**
- **File**: `scripts/train.py`
- **Location**: `train_operator()` function (lines 400-693)
- **Change**: Track per-task losses during training
  ```python
  # After loss computation (around line 614)
  if cfg.get("training", {}).get("log_per_task_metrics"):
      # Extract task ID from batch metadata
      for task_name in task_list:
          task_mask = (batch_meta["task_id"] == task_name)
          if task_mask.sum() > 0:
              task_loss = loss_bundle.total[task_mask].mean()
              wandb_ctx.log_training_metric(
                  stage="operator",
                  metric=f"loss_{task_name}",
                  value=task_loss.item(),
                  step=global_step
              )
  ```
- **Estimated Time**: 2-3 hours (includes testing)

#### **Short-Term (Week 2-3)**

**Task 6.1: Hyperparameter Sweep Configs**
- **Files**: `configs/sweep_pdebench_3task_*.yaml` (create 9 configs)
- **Sweep Grid**:
  - `latent.dim` ∈ {48, 64, 96}
  - `latent.tokens` ∈ {96, 128, 192}
  - `operator.pdet.hidden_dim` = 3× or 4× latent_dim
- **Execution**: Use `scripts/vast_launch.py` with `--config configs/sweep_pdebench_3task_dim64_tokens128.yaml`
- **Estimated Time**: 3-5 days for full sweep (parallel on VastAI)

**Task 4.1: Curriculum Learning Implementation**
- **File**: `src/ups/training/loop_train.py`
- **Change**: Extend `CurriculumConfig` to support task schedules
  ```python
  @dataclass
  class CurriculumConfig:
      stages: List[Dict[str, Any]]  # Existing
      rollout_lengths: List[int]    # Existing
      # NEW:
      task_schedules: Optional[List[Dict[str, Any]]] = None
      # Example: [
      #   {"epochs": [0, 10], "tasks": ["burgers1d"], "weights": {"burgers1d": 1.0}},
      #   {"epochs": [10, 20], "tasks": ["burgers1d", "advection1d"], "weights": {...}},
      # ]
  ```
- **Estimated Time**: 1 week (includes DataLoader sampling logic, testing)

#### **Medium-Term (Month 2)**

**Task 7.1: Mesh/Particle Multi-Task Training**
- **Config**: `configs/train_pdebench_multidiscretization.yaml`
- **Challenges**:
  - Ensure `GraphLatentPairDataset` and `GridLatentPairDataset` produce compatible latent shapes
  - Handle heterogeneous batch collation (grid coords vs. graph adjacency)
  - Test mixed-modality training convergence
- **Estimated Time**: 1-2 weeks

**Task X: Multi-Task Leaderboard Extension**
- **File**: `src/ups/utils/leaderboard.py`
- **Change**: Support per-task metric columns in CSV
- **Estimated Time**: 1-2 days

## Code References

### Current Infrastructure (What Works Today)

**Data Loading**:
- `src/ups/data/pdebench.py:23-37` — Task specifications (10 tasks defined)
- `src/ups/data/pdebench.py:72-158` — PDEBenchDataset (HDF5 loader, normalization, param/BC aggregation)
- `src/ups/data/latent_pairs.py:257-413` — GridLatentPairDataset (encoder calls, caching, conditioning)
- `src/ups/data/latent_pairs.py:472-543` — GraphLatentPairDataset (mesh/particle encoding)
- `src/ups/data/latent_pairs.py:706-850` — `build_latent_pair_loader` (multi-task support via ConcatDataset)

**Encoders**:
- `src/ups/io/enc_grid.py` — GridEncoder (patch-based tokenization)
- `src/ups/io/enc_mesh_particle.py` — MeshParticleEncoder (message passing, supernode pooling)
- `src/ups/io/decoder_anypoint.py` — Query-based decoder (cross-attention to latent)

**Training**:
- `src/ups/training/losses.py:25-99` — Inverse losses (L_inv_enc, L_inv_dec)
- `src/ups/training/losses.py:168-251` — `compute_operator_loss_bundle` (forward, rollout, spectral, inverse)
- `src/ups/training/loop_train.py:27-106` — LatentTrainer (curriculum-driven training loop)
- `scripts/train.py:400-1824` — Multi-stage training orchestration
- `scripts/run_fast_to_sota.py:351-1240` — Fast-to-SOTA pipeline (validation → train → eval → gates)

**Caching**:
- `src/ups/data/parallel_cache.py:33-88` — RawFieldDataset (multi-worker safe encoding)
- `src/ups/data/parallel_cache.py:90-167` — PreloadedCacheDataset (RAM-backed caching)
- `src/ups/data/parallel_cache.py:258-310` — `build_parallel_latent_loader` (parallel encoding)

### Missing Infrastructure (What Needs to Be Built)

**Dataset Generation**:
- ❌ `scripts/prepare_data.py` — Mesh/particle dataset generator (referenced by tests but missing)
- ❌ `scripts/convert_pdebench_multimodal.py` — Multi-modality converter (documented but missing)

**Multi-Task Training**:
- ⚠️ Curriculum dataset mixing scheduler (needs extension to `src/ups/training/loop_train.py`)
- ⚠️ Per-task metric logging (needs WandB integration in `scripts/train.py`)
- ⚠️ Mixed-modality collate function (extension to `src/ups/data/latent_pairs.py:664-685`)

**Configuration**:
- ⚠️ Multi-task training configs (no examples in `configs/`)
- ⚠️ Scaling law documentation (empirical guidelines needed)

## Architecture Documentation

### Current Data Pipeline Flow

```
PDEBench HDF5/Zarr Files
    ↓
PDEBenchDataset (load, normalize, aggregate params/BC)
    ↓
GridLatentPairDataset / GraphLatentPairDataset
    ├─ Check cache → load if hit
    ├─ Encode if miss → GridEncoder / MeshParticleEncoder
    ├─ Store cache (optional)
    └─ Prepare conditioning tensors (params, BC, time, dt)
    ↓
build_latent_pair_loader
    ├─ Single task: Create dataset
    ├─ Multi-task: ConcatDataset(task_datasets)
    ├─ Cache strategy: PreloadedCache if RAM sufficient, else on-demand
    └─ DataLoader with num_workers, pin_memory, prefetch
    ↓
Training Loop (LatentTrainer)
    ├─ Fetch latent pairs (z0, z1, cond)
    ├─ Forward: operator(z0, dt) → z_pred
    ├─ Loss: compute_operator_loss_bundle(z_pred, z1, ...)
    └─ Backward + EMA update
    ↓
Checkpoints → Evaluation → Leaderboard
```

### Proposed Multi-Dataset Flow (Phase 2-4)

```
PDEBench Suite (10+ tasks)
    ├─ 1D Grid: Burgers, Advection, Diffusion-Sorption
    ├─ 2D Grid: Darcy, NS, Reaction-Diffusion, Shallow Water
    ├─ 3D Grid: Compressible NS
    ├─ Mesh: Darcy2D, ...
    └─ Particles: Advection, ...
    ↓
Task-Specific Loaders (per-task caching)
    ├─ GridLatentPairDataset(burgers1d)
    ├─ GridLatentPairDataset(advection1d)
    ├─ GraphLatentPairDataset(darcy2d_mesh)
    └─ ...
    ↓
ConcatDataset (with curriculum sampling weights)
    ├─ Stage 1: [burgers1d, advection1d] (50:50 mix)
    ├─ Stage 2: + [reaction_diffusion2d] (33:33:33 mix)
    └─ Stage 3: + [darcy2d_mesh] (25:25:25:25 mix)
    ↓
Training Loop (with per-task metrics)
    ├─ Fetch mixed batch
    ├─ Track task IDs in metadata
    ├─ Compute loss per task + aggregate
    └─ Log: {loss_burgers1d, loss_advection1d, loss_aggregate}
    ↓
Multi-Task Checkpoints → Multi-Task Evaluation → Multi-Task Leaderboard
```

## Historical Context (from thoughts/)

- `thoughts/shared/research/2025-11-05-pdebench-data-pipeline.md` — Original research on single-task PDEBench integration
- `thoughts/shared/plans/2025-11-06-pdebench-data-pipeline-expansion.md` — Phase 1-4 plan for multi-dataset support
- `UPT_docs/UPT_Arch_Train_Scaling.md` — UPT staged scaling guidance (S0→S3, capacity tiers)
- `universal_physics_stack_implementation_plan_for_a_coding_agent.md` — Original vision for discretization-agnostic training

## Related Research

- `thoughts/shared/research/2025-10-28-checkpoint-resume-system.md` — Checkpoint orchestration (relevant for multi-stage multi-dataset training)
- `TRAINING_PIPELINE_DOCUMENTATION.md` — Training loop, loss functions, multi-stage progression

## Recommended Next Steps

### Immediate (Next 1-2 Weeks)
1. **Restore `scripts/prepare_data.py` and `scripts/convert_pdebench_multimodal.py`**:
   - Priority: HIGH (blocking mesh/particle unit tests)
   - Reference: Test expectations in `tests/unit/test_mesh_loader.py`, `tests/unit/test_particles_dataset.py`
   - Output: Zarr generation utilities for mesh/particle datasets

2. **Create 3-Task Multi-Dataset Training Config**:
   - Tasks: Burgers1D, Advection1D, Reaction-Diffusion2D (start simple)
   - Config: `configs/train_pdebench_3task.yaml`
   - Test: Run 1 epoch of operator training, verify convergence

3. **Implement Per-Task Metric Logging**:
   - Modify: `scripts/train.py` training loop to track per-task losses
   - WandB: Log `loss_burgers1d`, `loss_advection1d`, etc.
   - Validation: Confirm metrics appear in WandB dashboard

### Short-Term (2-4 Weeks)
4. **Extend TASK_SPECS to Full PDEBench Suite**:
   - Add: Diffusion-Sorption1D, Compressible NS 1D/3D
   - Test: Validate data loading for each new task
   - Documentation: Update `src/ups/data/pdebench.py` docstrings

5. **Implement Curriculum Learning for Multi-Dataset**:
   - Modify: `src/ups/training/loop_train.py` to support dataset mixing schedules
   - Strategy: Start with 1D tasks, progressively add 2D/mesh
   - Evaluation: Compare curriculum vs. flat mixing on convergence speed

6. **Run Hyperparameter Sweeps**:
   - Sweep: `latent.dim` ∈ {32, 48, 64} on 3-task config
   - Sweep: `latent.tokens` ∈ {64, 128, 256}
   - Document: Scaling laws (latent_dim vs. N_tasks)

### Medium-Term (1-2 Months)
7. **Enable Mesh/Particle Training**:
   - Generate: Mesh/particle test datasets via restored `prepare_data.py`
   - Config: `configs/train_multidiscretization.yaml` (grid + mesh + particle)
   - Test: Full pipeline training on heterogeneous discretizations

8. **Multi-Dataset Fast-to-SOTA Pipeline**:
   - Extend: `scripts/run_fast_to_sota.py` to support multi-task evaluation
   - Gates: Per-task NRMSE gates, aggregate conservation checks
   - Leaderboard: Multi-task leaderboard with per-task breakdowns

9. **Production Multi-Dataset Config**:
   - Create: `configs/train_pdebench_golden_multitask.yaml`
   - Validate: 10+ task training, 90%+ GPU utilization, cache efficiency
   - Benchmark: Compare single-task vs. multi-task generalization

## Open Questions

1. **Task Weighting Strategy**: Should multi-task training use uniform sampling or performance-based weighting (e.g., oversample harder tasks)?
   - **Recommendation**: Start uniform, switch to inverse-NRMSE weighting after initial convergence

2. **Cross-Discretization Conditioning**: How to encode discretization type (grid vs. mesh vs. particle) in conditioning tensors?
   - **Recommendation**: Add `discretization_type` embedding (one-hot or learned) to conditioning dict

3. **Multi-Dataset Cache Strategy**: Should we cache multi-task mixed batches or per-task latent pairs?
   - **Recommendation**: Per-task caching (current approach), load & mix in DataLoader

4. **Inverse Loss Scaling**: Do inverse loss weights (λ_inv_enc, λ_inv_dec) need adjustment for multi-dataset training?
   - **Recommendation**: Start with same weights (0.05 max), monitor per-task invertibility metrics

5. **Domain-Adversarial Training**: Should we implement modality alignment (as mentioned in UPT paper) for grid/mesh/particle mixing?
   - **Recommendation**: Defer to Phase 5+ (research extension), not critical for initial multi-dataset training

## Conclusion

**CORRECTED**: The Universal Physics Stack is **fully equipped for multi-dataset scaling**, with **critical infrastructure already implemented**:

✅ **What Works Today**:
1. ✅ Dataset generation utilities (`scripts/prepare_data.py`, `scripts/convert_pdebench_multimodal.py`) — EXIST and functional
2. ✅ Multi-task data loading (`build_latent_pair_loader`) — ConcatDataset support operational
3. ✅ Golden UPT config (`configs/train_burgers_upt_128tokens_pure.yaml`) — 64-dim latent, 128 tokens, proven performance
4. ✅ Latent caching infrastructure — Parallel encoding, RAM preload, 90%+ GPU utilization
5. ✅ UPT inverse losses implemented — Ready for multi-dataset testing

❌ **Remaining Gaps** (Unblock Multi-Dataset Training):
1. ❌ Task specifications incomplete (10/15+ PDEBench tasks) — **30-minute fix**
2. ❌ Multi-task training configs — **1-hour creation** (template from golden config)
3. ❌ Per-task metric logging — **2-3 hour implementation** in `scripts/train.py`
4. ❌ Curriculum learning scheduler — **1-week implementation** (optional, not blocking)

**Recommended Path Forward (UPDATED)**:
- **Day 1** (4-5 hours):
  1. Extend `TASK_SPECS` with missing PDEBench tasks (30 min)
  2. Create `configs/train_pdebench_3task_upt.yaml` (1 hour)
  3. Implement per-task metric logging in `scripts/train.py` (2-3 hours)
  4. Run 1-epoch smoke test on 3-task config (30 min)

- **Week 1** (after Day 1):
  - Launch 3-task multi-dataset training (Burgers1D + Advection1D + Reaction-Diffusion2D)
  - Expected: 4-6 hours on VastAI A100, ~$5-10 cost
  - Validate: Per-task NRMSE breakdown, aggregate metrics, cache efficiency

- **Week 2**:
  - Hyperparameter sweeps: `latent.dim` ∈ {48, 64, 96}, `latent.tokens` ∈ {96, 128, 192}
  - Identify optimal scaling laws (latent_dim vs. N_tasks)

- **Weeks 3-4**:
  - Implement curriculum learning scheduler
  - Scale to 5-task config, then full 10+ task PDEBench suite
  - Validate mesh/particle multi-task training

**Why This Works**:
The existing UPT architecture (latent space evolution, inverse losses, query-based decoding) provides a **modality-agnostic interface** perfectly suited for multi-dataset training. The latent space acts as a **universal interface** where heterogeneous PDEs (1D grids, 2D grids, meshes, particles) are mapped to a common representation, enabling seamless multi-task learning.

**Key Insight from Research**:
> Previous assumption that critical tooling was missing (prepare_data.py, convert_pdebench_multimodal.py) was **incorrect**. These scripts exist and are functional. The **only blockers** are config creation and per-task logging — both straightforward implementations with clear templates.

---

## Actionable Next Steps (START HERE)

### **Immediate (Next 4-5 Hours)**

**Step 1: Extend Task Specifications** (30 minutes)
```bash
# Edit src/ups/data/pdebench.py
# Add lines to TASK_SPECS dict (around line 24-37):
"diffusion_sorption1d": PDEBenchSpec(field_key="data"),
"compressible_ns1d": PDEBenchSpec(field_key="data"),
"compressible_ns2d": PDEBenchSpec(field_key="data"),
"compressible_ns3d": PDEBenchSpec(field_key="data"),
"diffusion_reaction1d": PDEBenchSpec(field_key="data"),

# Test:
pytest tests/unit/test_pdebench.py -v
```

**Step 2: Create 3-Task Config** (1 hour)
```bash
# Copy golden config as template
cp configs/train_burgers_upt_128tokens_pure.yaml configs/train_pdebench_3task_upt.yaml

# Edit configs/train_pdebench_3task_upt.yaml:
# - Change data.task to: [burgers1d, advection1d, reaction_diffusion2d]
# - Change operator.pdet.hidden_dim to: 256 (from 192)
# - Change operator.pdet.depth to: 10 (from 8)
# - Change operator.pdet.num_heads to: 8 (from 6)
# - Change training.batch_size to: 6 (from 8)
# - Change training.accum_steps to: 8 (from 6)
# - Add: training.log_per_task_metrics: true

# Validate config
python scripts/validate_config.py configs/train_pdebench_3task_upt.yaml
```

**Step 3: Implement Per-Task Logging** (2-3 hours)
```bash
# Edit scripts/train.py (around line 614 in train_operator function)
# Add per-task metric extraction and logging after loss computation
# See Section 8, Task 5.1 for code snippet

# Test with 1-epoch smoke test
python scripts/train.py \
  --config configs/train_pdebench_3task_upt.yaml \
  --stage operator \
  --epochs 1
```

**Step 4: Smoke Test Validation** (30 minutes)
```bash
# Verify:
# 1. Training completes without errors
# 2. WandB shows per-task metrics (loss_burgers1d, loss_advection1d, loss_reaction_diffusion2d)
# 3. Latent cache populated for all 3 tasks
# 4. GPU utilization > 80%

# Check WandB dashboard for metrics
```

### **Short-Term (Week 1-2)**

**Launch Full 3-Task Training**:
```bash
# VastAI (recommended)
python scripts/vast_launch.py launch \
  --config configs/train_pdebench_3task_upt.yaml \
  --auto-shutdown

# Or local
python scripts/train.py \
  --config configs/train_pdebench_3task_upt.yaml \
  --stage all
```

**Hyperparameter Sweeps**:
- Create sweep configs based on 3-task template
- Sweep `latent.dim`, `latent.tokens`, `operator.pdet.hidden_dim`
- Document optimal configs in `configs/train_pdebench_3task_golden.yaml`

### **Medium-Term (Month 1-2)**

1. Implement curriculum learning scheduler
2. Scale to 5-task, 10-task configs
3. Validate mesh/particle multi-task training
4. Document scaling laws and best practices

---

**Updated Action Items**:
1. ~~Create GitHub issue: "Restore scripts/prepare_data.py"~~ → **NO LONGER NEEDED** (script exists)
2. **Create `configs/train_pdebench_3task_upt.yaml`** → **HIGH PRIORITY** (1 hour)
3. **Implement per-task metric logging** → **HIGH PRIORITY** (2-3 hours)
4. **Run 1-epoch smoke test** → **VALIDATION** (30 min)
5. Schedule full 3-task training run on VastAI → **Week 1** ($5-10)

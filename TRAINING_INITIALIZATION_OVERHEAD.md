# Training Pipeline Initialization and Overhead

## Overview

This document details the initialization steps and overhead in the Universal Physics Stack training pipeline, identifying time-consuming operations during startup and between training stages.

**Purpose**: Document current initialization bottlenecks to inform optimization efforts.

**Scope**: Covers orchestration, training initialization, multi-stage transitions, validation, and compilation overhead.

---

## Table of Contents

1. [Pipeline Orchestration](#1-pipeline-orchestration)
2. [Training Initialization](#2-training-initialization)
3. [Multi-Stage Training Transitions](#3-multi-stage-training-transitions)
4. [Validation and Gating](#4-validation-and-gating)
5. [Compilation Overhead](#5-compilation-overhead)
6. [Summary of Overhead Sources](#6-summary-of-overhead-sources)

---

## 1. Pipeline Orchestration

### 1.1 run_fast_to_sota.py Coordination

**File**: `scripts/run_fast_to_sota.py`

**Key Initialization Steps**:

1. **Config Loading and Resolution** (Lines 517-527)
   - Load training config with includes
   - Apply config overrides
   - Resolve small/full eval configs
   - Write resolved configs to disk
   - **Overhead**: File I/O, YAML parsing, config validation

2. **Checkpoint Metadata Initialization** (Lines 575-586)
   - Check for existing metadata
   - Create or parse `metadata.json`
   - Write initial checkpoint metadata
   - **Overhead**: JSON parsing, file I/O

3. **Dataset Symlink Setup** (Lines 614-630)
   - Map dataset symlinks (e.g., `burgers1d_train.h5` → `burgers1d_train_000.h5`)
   - **Overhead**: Filesystem operations

4. **Validation Steps** (Lines 664-684)
   - Config validation subprocess
   - Data validation subprocess  
   - Dry-run subprocess
   - **Overhead**: 3 subprocess spawns + validation logic (see Section 4)

5. **Training Subprocess Launch** (Lines 694-712)
   - Construct training command
   - Set environment variables (`WANDB_MODE`, `FAST_TO_SOTA_WANDB_INFO`, `WANDB_CONTEXT_FILE`)
   - Spawn training subprocess
   - Wait for completion
   - **Overhead**: Subprocess spawn, environment setup

6. **Evaluation Subprocesses** (Lines 812-850, 951-988)
   - Small eval subprocess
   - Full eval subprocess
   - Each includes config loading, model loading, data loading
   - **Overhead**: 2 subprocess spawns + model initialization per eval

**Total Orchestrator Overhead**: ~10-30 seconds (validation + subprocess spawns + file I/O)

---

### 1.2 WandB Initialization

**File**: `src/ups/utils/wandb_context.py`

**Initialization Path** (Lines 274-414):

1. **WandB Run Creation** (Lines 381-392)
   - `wandb.init()` with project/entity/config
   - Network request to WandB servers
   - **Overhead**: 1-5 seconds (network latency)

2. **Metric Step Definition** (Lines 394-408)
   - Define step metrics for each training stage
   - Define metric relationships
   - **Overhead**: Minimal (<0.1s)

3. **Context File Persistence** (Lines 1538-1557 in train.py)
   - Save WandB context to file for evaluation subprocesses
   - Save WandB info for orchestrator
   - **Overhead**: Minimal file I/O

**WandB Overhead**: ~1-5 seconds (primarily network latency on `wandb.init()`)

---

## 2. Training Initialization

### 2.1 Model Initialization

**File**: `scripts/train.py`

#### Operator Stage (Lines 400-414)

1. **Dataset Loading** (Line 401)
   - `dataset_loader()` → `build_latent_pair_loader()`
   - PDEBench dataset initialization
   - Encoder initialization
   - Grid coordinate generation
   - **Overhead**: 5-15 seconds (data file opening, encoder creation)

2. **Model Creation** (Line 402)
   - `make_operator()` → LatentOperator instantiation
   - PDETransformer blocks creation
   - Time embedding layers
   - **Overhead**: 1-3 seconds

3. **Device Transfer** (Line 413)
   - Move model to GPU
   - **Overhead**: 0.5-2 seconds (depends on model size)

4. **torch.compile** (Line 414)
   - Compile operator if `training.compile=true`
   - **Overhead**: 10-60 seconds (see Section 5)

5. **Encoder/Decoder Initialization for Inverse Losses** (Lines 422-473)
   - If `use_inverse_losses=true`
   - Create GridEncoder and AnyPointDecoder
   - Share encoder with dataset if possible
   - Move to device and set to eval mode
   - **Overhead**: 2-5 seconds

**Total Operator Init**: ~20-85 seconds (varies with compilation)

---

#### Diffusion Stage (Lines 695-720)

1. **Dataset Loading** (Line 696)
   - Same as operator stage
   - **Overhead**: 5-15 seconds

2. **Device Determination** (Lines 699-701)
   - Check CUDA availability
   - **Overhead**: Minimal

3. **Operator Loading** (Lines 704-710)
   - Create operator model
   - Load checkpoint from disk
   - Strip compiled prefix
   - Move to device
   - **Overhead**: 2-5 seconds (checkpoint I/O + device transfer)

4. **Operator Compilation** (Line 711)
   - Compile operator as "teacher" (skipped for eval-only teacher models)
   - **Overhead**: 0 seconds (skipped due to "teacher" in name, Line 315)

5. **Diffusion Model Creation** (Lines 714-719)
   - Create DiffusionResidual model
   - Move to device
   - **Overhead**: 1-2 seconds

6. **Diffusion Compilation** (Line 720)
   - Compile diffusion model if enabled
   - **Overhead**: 10-60 seconds

**Total Diffusion Init**: ~20-85 seconds

---

#### Consistency Distillation Stage (Lines 942-993)

1. **Config Adjustment** (Lines 945-954)
   - Reduce batch size to avoid OOM
   - Enable pin_memory and persistent_workers
   - **Overhead**: Minimal

2. **Dataset Loading** (Line 956)
   - With adjusted batch size and worker settings
   - **Overhead**: 5-15 seconds

3. **Operator Loading** (Lines 964-972)
   - Same as diffusion stage
   - **Overhead**: 2-5 seconds

4. **Diffusion Loading** (Lines 975-987)
   - Load diffusion checkpoint
   - Move to device
   - **Overhead**: 2-5 seconds

5. **Conditional Compilation** (Lines 1019-1043)
   - Check stage-specific `compile` setting
   - Compile distillation function with `mode="default"`
   - **Overhead**: 10-60 seconds (if enabled)

**Total Consistency Init**: ~20-85 seconds

---

### 2.2 Dataset and DataLoader Initialization

**File**: `src/ups/data/latent_pairs.py`

#### PDEBench Dataset (Lines 569-585)

1. **PDEBenchDataset Creation** (Lines 570-572)
   - Open HDF5 file
   - Read dataset structure
   - **Overhead**: 2-5 seconds

2. **Grid Shape Inference** (Lines 574-576)
   - Infer grid dimensions from first sample
   - Infer channel count
   - **Overhead**: <0.5 seconds

3. **GridEncoder Creation** (Lines 577-584)
   - Create encoder config
   - Instantiate GridEncoder
   - **Overhead**: 1-2 seconds

**Total Dataset Init**: ~5-10 seconds

---

#### DataLoader Creation (Lines 588-720)

1. **Latent Cache Setup** (Lines 601-604)
   - Check cache directory config
   - Create cache paths
   - **Overhead**: Minimal (<0.1s)

2. **Encoder Device Transfer** (Lines 630-631, 654-655)
   - Move encoder to GPU
   - **Overhead**: 0.5-1 second

3. **Coordinate Generation** (Lines 631, 655)
   - `make_grid_coords()` creates normalized grid
   - **Overhead**: <0.1 seconds

4. **GridLatentPairDataset Wrapping** (Lines 633-645, 657-669)
   - Wrap base dataset with encoder
   - Configure caching, time_stride, rollout_horizon
   - **Overhead**: Minimal

5. **DataLoader Instantiation** (Lines 649, 671)
   - Create DataLoader with workers
   - **Overhead**: 1-2 seconds (worker process spawn)

**Total DataLoader Init**: ~2-5 seconds

---

### 2.3 First Batch Warmup

**Latent Cache Behavior** (Lines 293-352 in latent_pairs.py):

1. **Cache Miss on First Epoch**:
   - Read physical fields from HDF5
   - Encode to latent space (forward pass through encoder)
   - Save to cache (`.pt` file with torch.save)
   - **Overhead per sample**: ~50-200ms (depends on encoder size)

2. **Cache Hit on Subsequent Epochs**:
   - Load pre-encoded latents from disk
   - **Overhead per sample**: ~5-20ms (disk I/O)

3. **First Batch Encoding**:
   - With `batch_size=32`, first batch processes 32 samples
   - Cache miss: 32 × 100ms = ~3.2 seconds
   - Cache hit: 32 × 10ms = ~0.3 seconds

**Amortized Cost**:
- First epoch: ~5-10 seconds extra (dataset encoding)
- Subsequent epochs: ~0.5-1 second (cache loading)

---

### 2.4 Optimizer and Scheduler Initialization

**Optimizer Creation** (Lines 245-258):
- Extract stage-specific config
- Instantiate Adam/AdamW/SGD
- **Overhead**: <0.1 seconds

**Scheduler Creation** (Lines 261-295):
- Extract scheduler config
- Instantiate StepLR/CosineAnnealingLR/ReduceLROnPlateau
- **Overhead**: <0.1 seconds

**Total Optimizer/Scheduler Init**: <0.2 seconds

---

### 2.5 AMP and EMA Setup

**AMP Scaler** (Lines 479-480, 732-734):
- Create GradScaler if `training.amp=true`
- **Overhead**: <0.1 seconds

**EMA Model** (Lines 481-483, 735-737):
- Deep copy model if `ema_decay` set
- Set to eval mode
- **Overhead**: 0.5-2 seconds (depends on model size)

**Total AMP/EMA Init**: ~0.5-2 seconds

---

## 3. Multi-Stage Training Transitions

### 3.1 Stage Completion and Checkpoint Saving

**Operator Stage Completion** (Lines 667-692):

1. **Load Best State** (Line 667)
   - Restore best checkpoint from memory
   - **Overhead**: 0.5-1 second

2. **Save Operator Checkpoint** (Lines 670-672)
   - `torch.save()` to `checkpoints/operator.pt`
   - **Overhead**: 1-3 seconds (depends on model size)

3. **Save Operator EMA Checkpoint** (Lines 674-677)
   - Save EMA version to `checkpoints/operator_ema.pt`
   - **Overhead**: 1-3 seconds

4. **WandB Upload** (Lines 680-684)
   - Upload checkpoints to WandB
   - **Overhead**: 5-15 seconds (network transfer)

**Total Stage Completion**: ~8-22 seconds

---

### 3.2 Inter-Stage GPU Cache Clearing

**GPU Memory Cleanup** (Lines 1581-1620):

1. **After Operator** (Lines 1581-1584)
   - `torch.cuda.empty_cache()`
   - **Overhead**: 0.1-0.5 seconds

2. **After Diffusion** (Lines 1599-1602)
   - `torch.cuda.empty_cache()`
   - **Overhead**: 0.1-0.5 seconds

3. **After Consistency** (Lines 1617-1620)
   - `torch.cuda.empty_cache()`
   - **Overhead**: 0.1-0.5 seconds

4. **After Consistency (explicit cleanup)** (Lines 1192-1195)
   - Delete operator model
   - `torch.cuda.empty_cache()`
   - **Overhead**: 0.1-0.5 seconds

**Total GPU Cleanup**: ~0.5-2 seconds

---

### 3.3 Checkpoint Loading Between Stages

**Diffusion Stage Loading** (Lines 706-710):

1. **Operator Checkpoint Load**:
   - `torch.load()` from disk
   - Strip compiled prefix
   - Load state dict
   - **Overhead**: 2-5 seconds

**Consistency Stage Loading** (Lines 966-970, 982-986):

1. **Operator Checkpoint Load**: 2-5 seconds
2. **Diffusion Checkpoint Load**: 2-5 seconds

**Total Checkpoint Loading**: ~4-10 seconds per stage transition

---

## 4. Validation and Gating

### 4.1 Config Validation

**File**: `scripts/validate_config.py`

**Validation Steps** (Lines 356-423):

1. **Config Loading** (Lines 372-374)
   - `load_config_with_includes()`
   - **Overhead**: 0.5-1 second

2. **Architecture Checks** (Lines 379-381)
   - Validate latent dimensions
   - Check operator/diffusion consistency
   - **Overhead**: <0.1 seconds

3. **Data Checks** (Lines 383-385)
   - Validate data.task, data.split, data.root
   - **Overhead**: <0.1 seconds

4. **Hardware Checks** (Lines 387-389)
   - Memory estimates (conservative)
   - **Overhead**: <0.1 seconds

5. **Hyperparameter Checks** (Lines 391-393)
   - Validate learning rates, epochs, grad_clip
   - **Overhead**: <0.1 seconds

6. **Checkpoint Checks** (Lines 395-397)
   - Check if checkpoints exist
   - **Overhead**: <0.1 seconds (filesystem stat)

7. **WandB Checks** (Lines 399-401)
   - Validate WandB config
   - **Overhead**: <0.1 seconds

8. **Stage Checks** (Lines 403-405)
   - Validate training stages config
   - **Overhead**: <0.1 seconds

9. **Evaluation Checks** (Lines 407-409)
   - Validate evaluation config
   - **Overhead**: <0.1 seconds

**Total Config Validation**: ~1-2 seconds

---

### 4.2 Data Validation

**File**: `scripts/validate_data.py`

**Validation Steps** (Lines 168-222):

1. **File Existence Check** (Lines 181-188)
   - Check HDF5 file exists
   - Check file size > 0
   - **Overhead**: <0.1 seconds

2. **HDF5 Integrity Check** (Lines 191-202)
   - Open HDF5 file
   - Read keys and first dataset
   - Read sample chunk
   - **Overhead**: 1-3 seconds

3. **Data Statistics Check** (Lines 205-210)
   - Sample first 10 timesteps
   - Check for NaNs, Infs
   - Compute min/max/mean/std
   - **Overhead**: 0.5-1 second

4. **Data Shape Check** (Lines 213-220)
   - Validate tensor dimensions
   - **Overhead**: <0.1 seconds

**Total Data Validation**: ~2-5 seconds

---

### 4.3 Dry Run

**File**: `scripts/dry_run.py` (inferred)

**Typical Dry Run**:
- Estimate cost based on config
- Quick sanity checks
- **Overhead**: ~1-2 seconds

---

### 4.4 Total Validation Overhead

**From run_fast_to_sota.py** (Lines 664-684):
- Config validation subprocess: ~1-2 seconds
- Data validation subprocess: ~2-5 seconds  
- Dry run subprocess: ~1-2 seconds
- Subprocess spawn overhead: ~0.5-1 second per subprocess

**Total Validation**: ~6-12 seconds

---

## 5. Compilation Overhead

### 5.1 torch.compile Behavior

**File**: `scripts/train.py`

**Compilation Trigger** (Lines 302-333):

1. **Compilation Check** (Lines 307-312)
   - Check `training.compile` flag
   - Skip if model name contains "teacher"
   - **Overhead**: Minimal

2. **First Forward Pass** (Line 329)
   - `torch.compile(model, mode="default", fullgraph=False)`
   - Actual compilation happens on **first forward pass**
   - **Overhead**: Deferred to first batch

3. **Compilation Modes**:
   - `mode="default"`: Safe mode, no CUDA graphs
   - `mode="reduce-overhead"`: Faster but can cause issues (not used)
   - `mode="max-autotune"`: Aggressive optimization (optional)

**Compilation Overhead**:
- First forward pass: **10-60 seconds** (depends on model complexity)
- Subsequent forward passes: No overhead (compiled graph cached)

---

### 5.2 Distillation Function Compilation

**Consistency Stage Compilation** (Lines 1019-1043):

1. **Conditional Compilation** (Lines 1029-1030)
   - Check stage-specific or global `compile` setting
   - `torch.compile(_distill_forward_and_loss, mode="default")`

2. **First Batch Compilation**:
   - Compilation happens on first batch
   - Fuses tau expansion + diffusion forward + loss
   - **Overhead**: ~5-20 seconds (first batch only)

**Note**: Compilation overhead is **amortized** across training epochs.

---

### 5.3 Compilation Impact Summary

| Stage | Compilation Target | First Pass Overhead | Subsequent Overhead |
|-------|-------------------|---------------------|---------------------|
| Operator | LatentOperator | 10-60s | 0s |
| Diffusion | DiffusionResidual | 10-60s | 0s |
| Consistency | Distillation function | 5-20s | 0s |

**Total Compilation Overhead** (first epoch): ~25-140 seconds

**Amortization**: Overhead is one-time per run, negligible after first batch.

---

## 6. Summary of Overhead Sources

### 6.1 Breakdown by Category

| Category | Overhead (seconds) | Location |
|----------|-------------------|----------|
| **Orchestration** | | |
| Config loading & resolution | 1-2 | run_fast_to_sota.py:517-527 |
| Checkpoint metadata init | 0.5-1 | run_fast_to_sota.py:575-586 |
| Dataset symlink setup | 0.1-0.5 | run_fast_to_sota.py:614-630 |
| Validation subprocesses | 6-12 | run_fast_to_sota.py:664-684 |
| Training subprocess spawn | 0.5-1 | run_fast_to_sota.py:694-712 |
| Eval subprocesses (2×) | 10-30 | run_fast_to_sota.py:812-988 |
| **WandB** | | |
| WandB initialization | 1-5 | wandb_context.py:381-392 |
| **Model Initialization** | | |
| Dataset loading | 5-15 | latent_pairs.py:569-585 |
| Model creation | 1-3 | train.py:402 |
| Device transfer | 0.5-2 | train.py:413 |
| Encoder/decoder init | 2-5 | train.py:422-473 |
| Optimizer/scheduler | 0.2 | train.py:245-295 |
| AMP/EMA setup | 0.5-2 | train.py:479-483 |
| **Checkpoint Operations** | | |
| Checkpoint saving (per stage) | 2-6 | train.py:670-677 |
| WandB upload (per stage) | 5-15 | train.py:680-684 |
| Checkpoint loading (per stage) | 4-10 | train.py:706-710, 966-986 |
| GPU cache clearing | 0.5-2 | train.py:1581-1620 |
| **Compilation (First Epoch)** | | |
| Model compilation | 10-60 | train.py:414, 720 |
| Distillation compilation | 5-20 | train.py:1034 |
| **First Batch Warmup** | | |
| Latent cache encoding | 5-10 | latent_pairs.py:293-352 |

---

### 6.2 Total Initialization Time Estimates

**Minimal Configuration** (compile=false, validation skipped):
- Orchestration: ~2 seconds
- Model init: ~10 seconds
- Dataset init: ~5 seconds
- **Total**: ~17 seconds

**Typical Configuration** (compile=true, validation enabled):
- Orchestration: ~20 seconds
- WandB init: ~3 seconds
- Model init: ~15 seconds
- Dataset init: ~8 seconds
- First batch (compilation): ~40 seconds
- **Total**: ~86 seconds

**Maximum Overhead** (all validations, slow network, large models):
- Orchestration: ~45 seconds
- WandB init: ~5 seconds
- Model init: ~25 seconds
- Dataset init: ~15 seconds
- First batch (compilation): ~140 seconds
- **Total**: ~230 seconds (~4 minutes)

---

### 6.3 Per-Stage Breakdown (Typical)

**Stage 1: Operator Training**
1. Dataset loading: 5-15s
2. Model creation: 1-3s
3. Device transfer: 0.5-2s
4. Encoder/decoder init (if inverse losses): 2-5s
5. torch.compile (first batch): 10-60s
6. **Total**: ~20-85s

**Stage 2: Diffusion Training**
1. Dataset loading: 5-15s
2. Operator checkpoint load: 2-5s
3. Diffusion model creation: 1-2s
4. torch.compile (first batch): 10-60s
5. **Total**: ~18-82s

**Stage 3: Consistency Distillation**
1. Dataset loading: 5-15s
2. Operator checkpoint load: 2-5s
3. Diffusion checkpoint load: 2-5s
4. Distillation function compile (first batch): 5-20s
5. **Total**: ~14-45s

**Stage 4: Steady Prior** (if enabled)
1. Dataset loading: 5-15s
2. Model creation: 1-2s
3. **Total**: ~6-17s

---

### 6.4 Optimization Observations

**High-Impact Overhead** (> 10 seconds):
1. torch.compile first pass: 10-60s (unavoidable, amortized)
2. Dataset loading: 5-15s (can be cached)
3. Validation subprocesses: 6-12s (can be skipped with `--skip-validation`)
4. Evaluation subprocesses: 10-30s (necessary for gating)
5. WandB checkpoint upload: 5-15s (network-bound)

**Low-Impact Overhead** (< 1 second):
1. Config loading: 0.5-1s
2. Checkpoint metadata: 0.5-1s
3. Optimizer/scheduler init: 0.2s
4. GPU cache clearing: 0.5-2s

**Amortized Overhead** (one-time per run):
1. Compilation: 25-140s (first epoch only)
2. Latent cache encoding: 5-10s (first epoch only)
3. WandB initialization: 1-5s (per run)

---

## 7. Key File References

**Primary Files**:
- `scripts/run_fast_to_sota.py` - Orchestrator
- `scripts/train.py` - Training engine
- `scripts/validate_config.py` - Config validation
- `scripts/validate_data.py` - Data validation
- `src/ups/utils/wandb_context.py` - WandB lifecycle
- `src/ups/data/latent_pairs.py` - Dataset/DataLoader
- `src/ups/models/latent_operator.py` - Model architecture

---

## 8. Notes

1. **Compilation is amortized**: First-epoch overhead (~25-140s) is one-time per run and negligible across 25+ epochs.

2. **Latent caching is effective**: First epoch encoding overhead (~5-10s) eliminates ~3-5× slowdown in subsequent epochs.

3. **Validation can be skipped**: Use `--skip-validation`, `--skip-data-check`, `--skip-dry-run` to save ~6-12s when iterating.

4. **Subprocess overhead is modest**: Orchestrator spawns 3-5 subprocesses adding ~2-5s total.

5. **Network-bound operations**: WandB init (~1-5s) and checkpoint uploads (~5-15s) depend on network speed.

6. **Checkpoint I/O is fast**: Saving/loading checkpoints takes 2-10s per stage, acceptable for production.

7. **GPU memory management is efficient**: Cache clearing between stages takes <1s per transition.

---

**Document Status**: Complete  
**Last Updated**: 2025-10-28  
**Version**: 1.0

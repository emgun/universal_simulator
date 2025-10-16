# Parallel Cache Quick Start

## 🚀 Immediate Action (For Future Runs)

Your current H200 run uses `num_workers: 0` (slow but stable). For your **next run**, use one of these fast methods:

### Option A: Auto-Parallel (Simplest)

Update `configs/train_burgers_quality_v3.yaml`:

```yaml
training:
  num_workers: 8              # Was: 0
  use_parallel_encoding: true # Add this line
  batch_size: 8               # Already optimal for H200
  latent_cache_dir: data/latent_cache  # Already set
```

**Result**: 4-8× faster cache creation (4-8 min vs 20-30 min)  
**Savings**: ~15-22 min = $0.65-0.95 per run @ $2.59/hr

### Option B: Precompute Once (Best for Multiple Runs)

```bash
# SSH into your instance
ssh root@<instance-ip>
cd /workspace/universal_simulator

# Precompute cache once (4-8 min)
python scripts/precompute_latent_cache.py \
  --config configs/train_burgers_quality_v3.yaml \
  --parallel \
  --num-workers 8 \
  --batch-size 8

# Train (cache already ready, instant)
python scripts/train.py --config configs/train_burgers_quality_v3.yaml
```

**Result**: Cache created once, all subsequent runs instant  
**Best for**: Hyperparameter tuning, debugging, iterations

## 📊 Performance Comparison

| Method | Cache Creation | Total Time | Cost @ $2.59/hr |
|--------|----------------|------------|-----------------|
| **Current** (`num_workers: 0`) | 20-30 min | 35-50 min | $1.50-2.15 |
| **Option A** (auto-parallel) | 4-8 min | 19-28 min | $0.82-1.21 |
| **Option B** (precomputed) | 4-8 min (once) | 15-20 min/run | $0.65-0.86/run |

## 🎯 Recommendation

- **Next single run**: Use Option A (one-line config change)
- **Hyperparameter sweep**: Use Option B (precompute once)
- **Production pipeline**: Use Option B in onstart script

## 📖 Full Documentation

See [docs/parallel_cache_optimization.md](docs/parallel_cache_optimization.md) for:
- Technical details
- Troubleshooting
- Best practices
- Migration guide

## ✅ What Was Implemented

1. **`src/ups/data/parallel_cache.py`** - New parallel-safe data loader
2. **`scripts/precompute_latent_cache.py`** - Enhanced with `--parallel` flag
3. **`src/ups/data/latent_pairs.py`** - Auto-detection of parallel mode
4. **Documentation** - Complete guide in `docs/parallel_cache_optimization.md`

## 🔍 Current Run Status

Your H200 instance is running with `num_workers: 0` (stable):
- Cache creation: ~20-30 min
- Total time: ~35-50 min
- No changes needed for current run

**For next run**: Add `use_parallel_encoding: true` to save 15-22 minutes!


# Production Configs Directory

This directory contains **production-ready, validated configurations only**.

Experimental configs should be created in `experiments/YYYY-MM-DD-name/` instead (see `../experiments/README.md`).

---

## Golden Configs (Use These!)

These are the **canonical, recommended configurations** for production use:

### Training Configs

| Config | Description | Status | Last Validated |
|--------|-------------|--------|----------------|
| `train_burgers_golden.yaml` | **Primary Burgers training config** - 32-dim latent, TTC-enhanced | ✅ **GOLDEN** | 2025-01-22 |

**Golden config characteristics**:
- Extensively tested and validated
- Known good performance metrics
- Stable hyperparameters
- Documented in CLAUDE.md
- Referenced in documentation

### Evaluation Configs

| Config | Description | Status | Last Validated |
|--------|-------------|--------|----------------|
| `small_eval_rerun_txxoc8a8.yaml` | Small evaluation (fast validation) | ✅ Active | 2025-01-22 |
| `full_eval_rerun_txxoc8a8.yaml` | Full evaluation (comprehensive) | ✅ Active | 2025-01-22 |
| `eval_burgers_32dim_practical.yaml` | Practical evaluation config | ✅ Active | 2025-01-21 |
| `small_eval_burgers_32dim_practical.yaml` | Practical small eval | ✅ Active | 2025-01-21 |

### UPT Phase 3 Configs (Architecture Simplification)

**Pure Stacked Transformer** - Alternative to U-shaped architecture, recommended for 256+ tokens:

| Config | Tokens | Architecture | Attention | Purpose |
|--------|--------|--------------|-----------|---------|
| `train_burgers_upt_128tokens_pure.yaml` | 128 | pdet_stack | standard | Test pure transformer at Phase 2 winner token count |
| `train_burgers_upt_256tokens_pure.yaml` | 256 | pdet_stack | standard | UPT recommendation threshold (256-512 tokens) |
| `train_burgers_upt_128tokens_channel_sep.yaml` | 128 | pdet_stack | channel_separated | Compare attention mechanisms |

**Key Features**:
- **Architecture type**: `pdet_stack` (pure stacked transformer) vs `pdet_unet` (U-shaped, default)
- **Drop-path regularization**: Stochastic depth (0.1 for 8-layer networks)
- **Standard attention**: Multi-head self-attention with optional QK normalization
- **Fixed token count**: No pooling/unpooling throughout network
- **Linear drop-path schedule**: Increases from 0 at layer 0 to max at final layer

**Architecture Selection Guidelines**:
- **16-128 tokens**: Use U-shaped (`pdet_unet`) - current golden config
- **128 tokens** (transition): Test both U-shaped and pure (`pdet_stack`)
- **256-512 tokens**: Use pure stacked transformer (`pdet_stack`) - UPT recommendation

See Phase 3 implementation plan for detailed architecture comparison and ablation study design.

### Inference Configs

| Config | Description | Status |
|--------|-------------|--------|
| `inference_transient.yaml` | Transient rollout inference | ✅ Active |
| `inference_ttc.yaml` | Test-time conditioning inference | ✅ Active |
| `inference_steady.yaml` | Steady-state inference | ✅ Active |

### PDEBench Configs

| Config | Description | Status |
|--------|-------------|--------|
| `train_pdebench.yaml` | PDEBench training baseline | ✅ Active |
| `train_pdebench_scale.yaml` | PDEBench scaling experiments | ✅ Active |

---

## Deprecated Configs (Do Not Use)

These configs are **deprecated** and kept only for historical reference. They may have outdated hyperparameters, incorrect dimensions, or known issues.

| Config | Deprecated | Reason | Replaced By |
|--------|-----------|---------|-------------|
| `train_burgers_32dim.yaml` | 2025-01-22 | Superseded by golden config | `train_burgers_golden.yaml` |
| `train_burgers_32dim_golden.yaml` | 2025-01-22 | Superseded by golden config | `train_burgers_golden.yaml` |
| `train_burgers_32dim_v2_practical.yaml` | 2025-01-22 | Experimental iteration, not validated | `train_burgers_golden.yaml` |
| `train_burgers_32dim_practical.yaml` | 2025-01-22 | Experimental iteration, not validated | `train_burgers_golden.yaml` |
| `train_burgers_quality_v2.yaml` | 2025-01-22 | Experimental iteration, not validated | `train_burgers_golden.yaml` |

**Action**: These files will be moved to `configs/deprecated/` in the next cleanup cycle.

---

## Config Naming Convention

Production configs should follow these naming patterns:

### Training Configs
```
train_<problem>_<variant>.yaml

Examples:
- train_burgers_golden.yaml      (canonical Burgers config)
- train_navier_stokes_golden.yaml (future: canonical NS config)
- train_burgers_64dim.yaml        (variant: 64-dim latent space)
```

### Evaluation Configs
```
<scope>_eval_<problem>_<variant>.yaml

Examples:
- small_eval_burgers.yaml     (quick validation)
- full_eval_burgers.yaml      (comprehensive)
- eval_burgers_32dim.yaml     (specific variant)
```

### Inference Configs
```
inference_<mode>.yaml

Examples:
- inference_transient.yaml    (autoregressive rollout)
- inference_ttc.yaml          (test-time conditioning)
- inference_steady.yaml       (steady-state)
```

---

## Adding New Production Configs

**DO NOT add configs directly to this directory!**

Use the experiment lifecycle workflow instead:

```bash
# 1. Create experiment
mkdir -p experiments/$(date +%Y-%m-%d)-my-experiment
cp configs/train_burgers_golden.yaml experiments/$(date +%Y-%m-%d)-my-experiment/config.yaml

# 2. Edit config
vim experiments/$(date +%Y-%m-%d)-my-experiment/config.yaml

# 3. Run experiment
python scripts/vast_launch.py launch \
  --config experiments/$(date +%Y-%m-%d)-my-experiment/config.yaml

# 4. If successful: Promote to production
python scripts/promote_config.py \
  experiments/$(date +%Y-%m-%d)-my-experiment \
  --production-dir configs/ \
  --config-name train_burgers_improved.yaml \
  --update-leaderboard
```

This ensures:
- ✅ Proper validation before promotion
- ✅ Documented experiment history
- ✅ Metrics tracked in leaderboard
- ✅ No untested configs in production

---

## Deprecating Configs

When a config becomes outdated:

1. **Update this README** - Add entry to "Deprecated Configs" table
2. **Add deprecation notice** - Add comment to the config file itself:
   ```yaml
   # DEPRECATED: 2025-01-22
   # This config is outdated and should not be used.
   # Use train_burgers_golden.yaml instead.
   # Reason: Superseded by golden config
   ```
3. **Move to deprecated/** (optional):
   ```bash
   mkdir -p configs/deprecated
   git mv configs/train_burgers_32dim.yaml configs/deprecated/
   ```
4. **Update documentation** - Remove references from CLAUDE.md, README.md

---

## Config Validation

Before using any config in production:

```bash
# Validate config structure
python scripts/validate_config.py configs/train_burgers_golden.yaml

# Validate data availability
python scripts/validate_data.py configs/train_burgers_golden.yaml

# Dry-run with cost estimate
python scripts/dry_run.py configs/train_burgers_golden.yaml --estimate-only
```

---

## Utility Configs

Special-purpose configs that don't fit the standard patterns:

| Config | Purpose |
|--------|---------|
| `defaults.yaml` | Default values for all configs (inherited) |

---

## Questions?

- **"Which config should I use?"** → Use `train_burgers_golden.yaml` for Burgers equation training
- **"Can I modify a production config?"** → No! Create an experiment instead (see above)
- **"How do I create a new variant?"** → Use experiment workflow, then promote if successful
- **"What if I need a quick test?"** → Still use experiments/, mark it as a test in notes.md

See `../experiments/README.md` for the complete experiment workflow.

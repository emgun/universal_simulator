# PyTorch Patterns - Quick Index

This index maps PyTorch patterns to their locations in the codebase.

## Core Files

- **Main Training Script:** `/Users/emerygunselman/Code/universal_simulator/scripts/train.py` (2000+ lines)
  - Contains all training functions: `train_operator`, `train_diffusion`, `train_consistency_distill`, `train_steady_prior`
  - All optimizer, scheduler, and AMP patterns
  
- **Evaluation Script:** `/Users/emerygunselman/Code/universal_simulator/scripts/evaluate.py`
  - Checkpoint loading patterns

- **Optimizer Utilities:** `/Users/emerygunselman/Code/universal_simulator/src/ups/training/`
  - `hybrid_optimizer.py` - Multi-optimizer wrapper
  - `param_groups.py` - Parameter splitting logic
  - `muon_factory.py` - Muon optimizer creation

- **Checkpoint Management:** `/Users/emerygunselman/Code/universal_simulator/src/ups/utils/checkpoint_manager.py`
  - WandB checkpoint downloading and management

---

## Pattern Quick Lookup

### Device Management
- **Runtime detection:** `scripts/train.py:491, 901, 1161, 1423`
- **Aggressive placement:** `scripts/train.py:1083-1093`
- **Batch-level transfers:** `scripts/train.py:640-642`
- **Non-blocking transfers:** `src/ups/data/latent_pairs.py:195-206, 1290-1292`
- **CUDA cache clearing:** `scripts/train.py:780-783, 972-976`

### Optimizers
- **Standard optimizers (Adam, AdamW, SGD):** `scripts/train.py:258-338`
- **Muon+AdamW hybrid:** `scripts/train.py:276-336`
- **Parameter group splitting:** `src/ups/training/param_groups.py:23-59`

### Schedulers
- **Scheduler factory:** `scripts/train.py:341-375`
- **Scheduler stepping:** `scripts/train.py:863-867`

### Mixed Precision
- **AMP availability check:** `scripts/train.py:378-379`
- **GradScaler setup:** `scripts/train.py:559-560, 934-935`
- **Autocast context:** `scripts/train.py:651, 981`
- **Gradient scaling:** `scripts/train.py:806-823, 999-1010, 1330-1340`
- **Spectral loss with autocast disabled:** `scripts/train.py:66-80`

### torch.compile
- **_maybe_compile function:** `scripts/train.py:382-413`
- **State dict prefix stripping:** `scripts/train.py:83-89`
- **torch._dynamo configuration:** `scripts/train.py:32-41`
- **Usage examples:** `scripts/train.py:494, 912, 921`

### Checkpointing
- **State dict saving:** `scripts/train.py:868-878`
- **State dict loading with map_location:** `scripts/train.py:906-911, 1168-1174`
- **EMA state tracking:** `scripts/train.py:557-563, 855-857`
- **Checkpoint manager:** `src/ups/utils/checkpoint_manager.py:36-123`

### Gradient Management
- **Gradient clipping:** `scripts/train.py:815-816, 820-821`
- **Zero grad with set_to_none:** `scripts/train.py:610, 823, 952`
- **EMA update:** `scripts/train.py:438-441, 826`
- **EMA initialization:** `scripts/train.py:430-435`

### Advanced Features
- **Gradient accumulation:** `scripts/train.py:568, 810`
- **Hybrid optimizer wrapper:** `src/ups/training/hybrid_optimizer.py:1-136`
- **Model eval/train modes:** `scripts/train.py:913, 536, 551`
- **Determinism configuration:** `scripts/train.py:97-126`
- **Multiprocessing setup:** `scripts/train.py:44-47`

---

## Pattern Statistics

- **Total lines documented:** 945
- **Files analyzed:** 20+
- **Core training file:** `scripts/train.py` (2000+ lines)
- **Patterns documented:** 11 major categories
- **Code examples:** 60+

---

## Documentation

Full details in: `/Users/emerygunselman/Code/universal_simulator/PYTORCH_PATTERNS.md`

Key Features:
1. Line numbers for easy reference
2. Complete code snippets
3. Explanation of each pattern
4. Cross-file references
5. Usage context

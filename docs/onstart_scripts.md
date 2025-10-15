# VastAI Onstart Script System

## Overview

The onstart script system has been **consolidated** into a single source of truth to eliminate duplication and inconsistencies.

## Architecture

```
scripts/onstart_template.py
  │
  │  Single source of truth for all onstart logic
  │  - Dependency installation (git, rclone, gcc, etc.)
  │  - Repository setup
  │  - Data downloading
  │  - Training launch
  │  - Auto-shutdown
  │
  ├── Used by: scripts/vast_launch.py (automatic)
  └── Used by: scripts/generate_onstart.py (manual)
```

## Key Benefits

✅ **Single Source of Truth**: All onstart logic in one place  
✅ **No Duplication**: Update once, works everywhere  
✅ **Consistent**: Same dependencies and setup across all instances  
✅ **Testable**: Can unit test the template  
✅ **Flexible**: Easy to customize per-instance  

## Usage

### Option 1: Automatic (Recommended)

Use `vast_launch.py` which automatically generates the onstart script:

```bash
python scripts/vast_launch.py launch \
  --gpus "H200_NVL" \
  --min-gpu-ram 96 \
  --max-price 2.50 \
  --datasets "burgers1d_full_v1" \
  --overrides "TRAIN_CONFIG=configs/train_burgers_512dim_v2_pru2jxc4.yaml" \
  --auto-shutdown
```

This will:
1. Generate `.vast/onstart.sh` using the template
2. Launch the instance
3. Upload and execute the onstart script

### Option 2: Manual Generation

Generate a custom onstart script manually:

```bash
# Generate with default settings
python scripts/generate_onstart.py \
  --train-config configs/train_burgers_512dim_v2_pru2jxc4.yaml \
  --datasets burgers1d_full_v1 \
  --auto-shutdown

# Advanced: Full customization
python scripts/generate_onstart.py \
  --train-config configs/train_burgers_32dim_pru2jxc4.yaml \
  --datasets burgers1d_full_v1 \
  --train-stage operator \
  --no-reset-cache \
  --output my_custom_onstart.sh
```

### Option 3: Programmatic

Use the template directly in Python:

```python
from scripts.onstart_template import generate_onstart_script_simple

script = generate_onstart_script_simple(
    train_config="configs/train_burgers_512dim_v2_pru2jxc4.yaml",
    datasets="burgers1d_full_v1",
    auto_shutdown=True
)

with open(".vast/onstart.sh", "w") as f:
    f.write(script)
```

## Template Features

The onstart template automatically includes:

### System Dependencies
- `git` - for repository cloning
- `python3-pip` - for Python packages
- `rclone` - for B2 data transfer
- `build-essential` (gcc) - for torch.compile()

### Environment Setup
- Clone repository from GitHub
- Checkout correct branch
- Install Python dependencies
- Load environment variables from `.env`

### Data Management
- Download test/val datasets via WandB
- Download training data from B2
- Create symlinks for data files

### Training Launch
- Set environment variables (TRAIN_CONFIG, TRAIN_STAGE, RESET_CACHE)
- Execute `scripts/run_remote_scale.sh`
- Optional auto-shutdown after completion

## Configuration Options

See `scripts/onstart_template.py::OnstartConfig` for all options:

```python
@dataclass
class OnstartConfig:
    # Repository
    repo_url: str = "https://github.com/emgun/universal_simulator.git"
    branch: str = "feature/sota_burgers_upgrades"
    workdir: str = "/workspace"
    
    # Training
    train_config: Optional[str] = None
    train_stage: str = "all"
    reset_cache: bool = True
    
    # Data
    datasets: Optional[str] = None
    
    # B2 credentials
    b2_key_id: Optional[str] = None
    # ... (see file for complete list)
    
    # System
    auto_shutdown: bool = False
    install_deps: bool = True
```

## Migration from Old System

### Before (BAD ❌)
Multiple manual scripts with duplicated logic:
- `.vast_onstart_sota.sh` (manual)
- `.vast/onstart.sh` (manual or generated)
- Inline script generation in `vast_launch.py`

**Problem**: Had to update 3 places when changing logic (e.g., adding gcc)

### After (GOOD ✅)
Single template used everywhere:
- `scripts/onstart_template.py` (single source)
- `vast_launch.py` uses template
- `generate_onstart.py` uses template

**Benefit**: Update once in `onstart_template.py`, works everywhere!

## Troubleshooting

### Missing Dependencies

If training fails due to missing system packages, add them to the template:

1. Edit `scripts/onstart_template.py`
2. Add to the "Install dependencies" section:
   ```python
   "command -v yourpackage >/dev/null 2>&1 || (apt-get update && apt-get install -y yourpackage)",
   ```
3. Commit and push to GitHub
4. Future instances will automatically include it!

### Debugging Onstart Scripts

Check the generated script before launching:

```bash
python scripts/generate_onstart.py \
  --train-config configs/train_burgers_512dim_v2_pru2jxc4.yaml \
  --datasets burgers1d_full_v1 \
  --output /tmp/test_onstart.sh

cat /tmp/test_onstart.sh  # Review before using
```

### Instance Logs

Monitor onstart script execution on the instance:

```bash
# Via Vast CLI
vastai ssh <instance_id> cat /workspace/onstart.log

# Via SSH
ssh -p <port> root@<host> "tail -f /tmp/onstart.log"
```

## Best Practices

1. **Always use `vast_launch.py`** for launching instances (automatic template)
2. **Test locally first**: Generate script and review before launching
3. **Keep template simple**: Complex logic should go in separate scripts
4. **Document changes**: Update this file when modifying the template
5. **Use environment variables**: Store credentials in `.env`, not in code

## Examples

### Example 1: Standard 512-dim Training

```bash
python scripts/vast_launch.py launch \
  --gpus "H200_NVL" \
  --max-price 2.50 \
  --datasets "burgers1d_full_v1" \
  --overrides "TRAIN_CONFIG=configs/train_burgers_512dim_v2_pru2jxc4.yaml" \
  --auto-shutdown
```

### Example 2: Resume Training (No Cache Reset)

```bash
python scripts/vast_launch.py launch \
  --gpus "H200_NVL" \
  --max-price 2.50 \
  --datasets "burgers1d_full_v1" \
  --overrides "TRAIN_CONFIG=configs/train_burgers_512dim_v2_pru2jxc4.yaml,RESET_CACHE=0" \
  --auto-shutdown
```

### Example 3: Train Only Operator Stage

```bash
python scripts/vast_launch.py launch \
  --gpus "H200_NVL" \
  --max-price 2.50 \
  --datasets "burgers1d_full_v1" \
  --overrides "TRAIN_CONFIG=configs/train_burgers_512dim_v2_pru2jxc4.yaml,TRAIN_STAGE=operator" \
  --auto-shutdown
```

### Example 4: Custom Manual Script

```bash
# Generate custom script
python scripts/generate_onstart.py \
  --train-config configs/train_burgers_32dim_pru2jxc4.yaml \
  --datasets burgers1d_full_v1 \
  --train-stage diffusion \
  --no-reset-cache \
  --output my_custom_onstart.sh

# Upload to instance
vastai copy <instance_id> my_custom_onstart.sh /workspace/onstart.sh

# Execute manually
vastai ssh <instance_id> "bash /workspace/onstart.sh"
```

## See Also

- `scripts/onstart_template.py` - Template implementation
- `scripts/vast_launch.py` - Automatic instance launcher
- `scripts/generate_onstart.py` - Manual script generator
- `scripts/run_remote_scale.sh` - Training script executed by onstart


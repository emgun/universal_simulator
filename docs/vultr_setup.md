# Vultr GPU Cloud Setup

## Overview

Vultr provides managed GPU cloud infrastructure with high reliability and NVLink/NVSwitch support for multi-GPU training. This guide covers setup for distributed training on Vultr GPU instances.

## 1. Create Vultr Account

Sign up at https://www.vultr.com/ and verify your account.

## 2. Generate API Key

1. Navigate to Account → API
2. Click "Enable API"
3. Copy your API key

## 3. Install Vultr CLI

**macOS**:
```bash
brew install vultr/vultr-cli/vultr-cli
```

**Linux**:
```bash
wget https://github.com/vultr/vultr-cli/releases/latest/download/vultr-cli_linux_amd64.tar.gz
tar -xzf vultr-cli_linux_amd64.tar.gz
sudo mv vultr-cli /usr/local/bin/
```

## 4. Configure Authentication

```bash
export VULTR_API_KEY="your-api-key-here"
echo 'export VULTR_API_KEY="your-api-key-here"' >> ~/.bashrc  # or ~/.zshrc
```

## 5. Verify Setup

```bash
vultr-cli account info
vultr-cli plans list --type vhf --output json
```

## 6. Usage with Universal Simulator

**Note**: The unified cloud launcher (`cloud_launch.py`) is available for Vultr support, but the existing `vast_launch.py` is recommended for production use as it has more mature features. For Vultr-specific deployments, you can use the cloud providers abstraction.

```python
# Example: Using cloud providers abstraction directly
from scripts.cloud_providers import get_provider

provider = get_provider("vultr")
instances = provider.search_instances(num_gpus=2, min_gpu_ram=80)
print(f"Found {len(instances)} available instances")
```

**VastAI (Recommended)**:
```bash
python scripts/vast_launch.py launch \
  --config configs/train_pdebench_2task_baseline_ddp.yaml \
  --auto-shutdown
```

## Provider Selection Guide

**Choose Vultr when:**
- Need high reliability (production workloads)
- Want NVLink/NVSwitch for 8×GPU clusters
- Budget allows $3-4/hr for 2×A100
- Require managed infrastructure with SLAs

**Choose VastAI when:**
- Cost-sensitive experiments ($2-3/hr for 2×A100)
- Okay with variable host reliability (community hosts)
- Need quick spot availability
- Prefer flexibility over guaranteed uptime

## Distributed Training on Vultr

Vultr GPU instances support distributed training via PyTorch DDP (Distributed Data Parallel):

### Multi-GPU Configuration

Your training config should specify the number of GPUs:

```yaml
training:
  num_gpus: 2              # 2×A100 = 160GB total
  batch_size: 8            # Per-GPU batch size
  accum_steps: 6           # Gradient accumulation steps
```

### Available GPU Plans

**2×A100 (160GB total)**:
- Cost: ~$3-4/hr
- Use case: 2-task multi-task training, medium models

**4×A100 (320GB total)**:
- Cost: ~$6-8/hr
- Use case: 11-task PDEBench suite, large models

**8×A100 (640GB total)**:
- Cost: ~$12-16/hr
- Use case: Very large models, maximum throughput
- Features: NVLink/NVSwitch for fast GPU communication

## Troubleshooting

### API Key Not Set

```
Error: VULTR_API_KEY environment variable not set
```

Solution: Export your API key as shown in step 4 above.

### No GPU Plans Available

If `vultr-cli plans list --type vhf` shows no results, GPU plans may not be available in your selected region. Try different regions or contact Vultr support.

### Instance Creation Fails

Check your account limits and billing status. GPU instances require sufficient account balance and may have quota limits.

## Cost Optimization

- Use auto-shutdown to avoid idle costs (already supported in vast_launch.py)
- Consider spot instances for non-critical workloads
- Monitor usage with `vultr-cli instance get <instance_id>`
- Delete instances promptly after training completes

## Additional Resources

- [Vultr API Documentation](https://www.vultr.com/api/)
- [Vultr CLI GitHub](https://github.com/vultr/vultr-cli)
- [Universal Simulator Training Guide](../PRODUCTION_WORKFLOW.md)
- [Distributed Training Plan](../thoughts/shared/plans/2025-11-12-distributed-training-multi-task.md)

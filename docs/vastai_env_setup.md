# VastAI Environment Variable Setup

## One-Time Setup

Set these environment variables **once** on your VastAI account. They'll be available in all future instances:

```bash
# B2 Credentials
vastai create env-var B2_KEY_ID "0043616a62c8bb90000000001"
vastai create env-var B2_APP_KEY "K004VmY71MG0Lk3sG8QM/cAv+S4BzBU"
vastai create env-var B2_S3_ENDPOINT "https://s3.us-west-004.backblazeb2.com"
vastai create env-var B2_S3_REGION "us-west-004"
vastai create env-var B2_BUCKET "pdebench"

# WandB Credentials
vastai create env-var WANDB_API_KEY "ec37eba84558733a8ef56c76e284ab530e94449b"
vastai create env-var WANDB_PROJECT "universal-simulator"
vastai create env-var WANDB_ENTITY "emgun-morpheus-space"
```

## Verify Setup

```bash
vastai show env-vars
```

## Benefits

1. **Security**: No credentials in scripts or git
2. **Convenience**: Set once, use everywhere
3. **Clean**: Scripts are simpler and more portable

## After Setup

Once these are set, you can launch instances without passing credentials:

```bash
python scripts/vast_launch.py launch \
  --overrides "TRAIN_CONFIG=configs/train_burgers_32dim.yaml TRAIN_STAGE=all" \
  --auto-shutdown
```

The onstart script will automatically use the environment variables that VastAI injects into every container.

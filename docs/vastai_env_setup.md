# VastAI Environment Variable Setup

## One-Time Setup

Set these environment variables **once** on your VastAI account. They'll be available in all future instances:

```bash
# B2 Credentials
vastai create env-var B2_KEY_ID "<YOUR_B2_KEY_ID>"
vastai create env-var B2_APP_KEY "<YOUR_B2_APP_KEY>"
vastai create env-var B2_S3_ENDPOINT "<YOUR_B2_ENDPOINT>"
vastai create env-var B2_S3_REGION "<YOUR_B2_REGION>"
vastai create env-var B2_BUCKET "<YOUR_B2_BUCKET>"

# WandB Credentials
vastai create env-var WANDB_API_KEY "<YOUR_WANDB_KEY>"
vastai create env-var WANDB_PROJECT "<YOUR_WANDB_PROJECT>"
vastai create env-var WANDB_ENTITY "<YOUR_WANDB_ENTITY>"
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
  --gpu <PREFERRED_GPU> \
  --config <TRAIN_CONFIG> \
  --auto-shutdown \
  --run-arg=--wandb-run-name=<WANDB_RUN_NAME> \
  --run-arg=--leaderboard-wandb
```

All other arguments (small/full eval configs, tags, redo flags, etc.) can be appended with additional `--run-arg=...` entries. The onstart script pulls credentials from the VastAI env-vars, clones the repo, and executes `scripts/run_fast_to_sota.py` on the remote GPU.

# Vultr Training Integration Plan

## Overview

Provide a first-class alternative to Vast.ai by automating GPU training launches on Vultr. The end state is a reproducible workflow that provisions Vultr GPU instances, attaches high-capacity storage for latent caches, bootstraps the UPS repository, hydrates datasets from Backblaze, and kicks off `scripts/run_fast_to_sota.py` or targeted stage runs with WandB logging parity. All automation must respect the existing Fast-to-SOTA orchestration patterns so teams can switch between Vast.ai and Vultr with a single flag.

## Current State

- `scripts/vast_launch.py` encapsulates Vast.ai provisioning (search → launch → env hydration → training → teardown). No equivalent exists for Vultr.
- `.env` already exposes `VULTR_KEY`, but no tooling consumes it.
- Infrastructure docs (`UPT_docs/UPT_Implementation_Plan.md`, `docs/fast_to_sota_playbook.md`) assume Vast.ai for remote training.
- Training configs expect ≥32 GB VRAM GPUs and up to 1 TB of fast NVMe storage when latent caches are enabled (see `configs/train_burgers_upt_optimized.yaml:3`).
- B2 credentials in `.env` power dataset hydration; the workflow needs to reuse these when running on Vultr.

## Constraints & Success Criteria

- **Security:** Never log `VULTR_KEY` in plaintext; use environment exports only.
- **Hardware Fit:** Must select plans that can run `upt_large` (≥40 GB VRAM) or gracefully fall back when hardware is unavailable.
- **Storage:** Attach ≥1 TB NVMe block volumes to satisfy latent cache and compile artifacts.
- **Automation Parity:** UX mirrors Vast pipeline: one CLI entry point handles provisioning, setup, training, teardown, and log collection.
- **Verification:** CI-friendly dry-run (no instance creation) plus an integration path that validates provisioning against Vultr’s sandbox (requires manual approval).

## Phase 0 — Recon & Abstractions

- [x] Catalogue features of `scripts/vast_launch.py` that must be mirrored (argument parsing, env templating, dataset hydration, fail-safe teardown).
- [x] Document Vultr API endpoints and required fields for:
  - GPU plan discovery (`/v2/plans?type=gpu`)
  - Region availability (`/v2/regions?availability=gpu`)
  - Instance lifecycle (`/v2/instances`)
  - Block storage (`/v2/blocks`)
- [x] Define a shared abstraction layer (`scripts/cloud_launch/common.py`) exposing:
  - `discover_offers(filter_spec)`
  - `provision_instance(spec)`
  - `attach_storage(instance_id, volume_spec)`
  - `run_bootstrap(instance_id, script)`
  - `destroy_instance(instance_id)`
- **Success Criteria:** Recon notes committed to `docs` or `thoughts/shared/research/` and a signed-off interface that both Vast and Vultr launchers can consume.

## Phase 1 — Vultr CLI Implementation

- [x] Create `scripts/vultr_launch.py` mirroring Vast CLI options (`setup-env`, `launch`, `teardown`, `status`, `logs`).
- [x] Implement plan discovery:
  ```python
  curl -s -H "Authorization: Bearer $VULTR_KEY" \
    "https://api.vultr.com/v2/plans?type=gpu"
  ```
  Parse GPU RAM, vCPU count, price, and region compatibility.
- [x] Implement region filtering to co-locate compute with Backblaze (`sjc`, `sea`, `lax` priority).
- [x] Add storage provisioning helpers that:
  - Create 1024 GB block volume
  - Wait for `status == "active"`
  - Attach to instance after creation
- [x] Wire bootstrap script to:
  - Update & install CUDA drivers/NVIDIA toolkit
  - Install Python deps (reuse `scripts/prepare_env.sh` if possible)
  - Mount block storage at `/mnt/cache`
  - Clone repo + checkout specified commit/hash
  - Invoke `scripts/run_fast_to_sota.py` or custom command
- [x] Ensure graceful teardown:
  - Stop training process
  - Detach & delete block volume (optional flag to keep)
  - Destroy instance
- **Success Criteria:** `scripts/vultr_launch.py --help` matches Vast CLI help; dry-run with `--dry-run` prints planned API calls without side effects.

## Phase 2 — Environment & Data Hydration

- [x] Extend bootstrap to export `.env` secrets securely (W&B, Backblaze, B2) via cloud-init or remote SSH.
- [x] Add dataset hydration step using `aws s3 sync` against Backblaze bucket into `/mnt/cache/data`.
- [x] Configure latent cache directories to reside on attached volume (update `configs/*` or pass overrides).
- [x] Integrate health checks:
  - Verify CUDA availability (`nvidia-smi`)
  - Verify disk throughput (simple `dd` benchmark) to ensure NVMe mount
  - Verify dataset presence (`ls data/...`)
- **Success Criteria:** Captured logs confirm environment bootstrap succeeded, dataset sync completed, and health checks passed on first run.

## Phase 3 — Training Workflow Integration

- [ ] Introduce `--cloud-provider {vast,vultr}` flag to `scripts/run_fast_to_sota.py` (or wrapper) to choose launcher.
- [ ] Update automation scripts (`scripts/dry_run.py`, CI docs) to reference Vultr option.
- [ ] Ensure WandB metadata includes `environment=vultr`.
- [ ] Support stage-specific launches (operator only, diffusion only) via CLI parity.
- [ ] Capture artifacts (checkpoints, reports) to Backblaze or Vultr object storage as needed.
- **Success Criteria:** A full operator run (proxy config) completes on Vultr, metrics appear in WandB with correct tags, and artifacts sync as expected.

## Phase 4 — Verification & Documentation

- [ ] Add unit tests that mock Vultr API responses for discovery and provisioning.
- [ ] Provide integration script (`scripts/test_vultr_integration.sh`) that validates credentials and can be gated behind `MANUAL=1`.
- [ ] Update documentation:
  - `docs/runbook.md`
  - `docs/fast_to_sota_playbook.md`
  - New quickstart appendix for Vultr setup (API key IP whitelisting, firewall, SSH).
- [ ] Record cost/performance benchmarks vs Vast (`reports/vultr_vs_vast.md`).
- **Success Criteria:** Tests pass locally; docs reviewed; benchmark report includes cost/hour, setup time, and training throughput comparison.

## Manual Verification Checklist

After automated checks succeed:
1. Confirm API IP allow list includes the runner’s outbound IP before running provisioning commands.
2. Manually inspect the created instance in the Vultr dashboard (GPU type, attached volume, firewall).
3. SSH into the instance to verify `/mnt/cache` mount and dataset presence.
4. Monitor WandB run to ensure metrics stream without interruption.
5. Validate teardown removes instances/volumes (or preserves them when `--keep` set).

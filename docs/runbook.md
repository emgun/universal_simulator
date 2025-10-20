# Operational Runbook

**Purpose:** Step-by-step operational procedures for common tasks.

**Audience:** Operators, DevOps, On-call Engineers

**Last Updated:** 2025-10-16

---

## Table of Contents

1. [Emergency Procedures](#emergency-procedures)
2. [Routine Operations](#routine-operations)
3. [Launch Procedures](#launch-procedures)
4. [Monitoring](#monitoring)
5. [Incident Response](#incident-response)
6. [Maintenance](#maintenance)

---

## Emergency Procedures

### EMERGENCY: Training Run Consuming Excessive Resources

**Severity:** High | **Response Time:** Immediate

**Symptoms:**
- GPU utilization 100% for >8 hours
- Cost exceeding budget
- Instance unresponsive

**Action Steps:**
```bash
# 1. Check instance status
python scripts/monitor_instance.sh <instance_id>

# 2. Stop training gracefully
vastai ssh <instance_id> "pkill -f python"

# 3. Stop instance (don't destroy)
vastai stop instance <instance_id>

# 4. Investigate logs
vastai logs <instance_id> > incident.log

# 5. Analyze what went wrong
python scripts/analyze_run.py <run_id> --output reports/incident_analysis.md
```

**Post-Incident:**
- Review config for issues (infinite loops, wrong epochs)
- Add alerts for similar patterns
- Update runbook with learnings

### EMERGENCY: Multiple Failed Launches

**Severity:** Medium | **Response Time:** 15 minutes

**Symptoms:**
- 3+ consecutive launch failures
- Same error across instances

**Action Steps:**
```bash
# 1. Check if it's a config issue
python scripts/validate_config.py configs/suspected_config.yaml

# 2. Check if it's a data issue
python scripts/validate_data.py configs/suspected_config.yaml

# 3. Check if it's a platform issue
vastai show instances

# 4. Test locally first
python scripts/dry_run.py configs/suspected_config.yaml
```

**Escalation:**
- If config/data valid → Platform issue, contact VastAI support
- If config/data invalid → Fix and redeploy
- If dry-run fails → Code issue, fix before launching

---

## Routine Operations

### OP-001: Launch New Training Run

**Frequency:** As needed | **Duration:** 10 minutes | **Risk:** Low

**Prerequisites:**
- [ ] Config validated
- [ ] Data available
- [ ] Credentials configured
- [ ] Budget approved

**Procedure:**

**Step 1:** Validate Config
```bash
cd /path/to/universal_simulator
python scripts/validate_config.py configs/train_burgers_32dim.yaml

# Expected output: "✅ Config is valid and ready for training!"
```

**Step 2:** Estimate Cost
```bash
python scripts/dry_run.py configs/train_burgers_32dim.yaml --estimate-only

# Note the estimated cost (e.g., "$1.25 on A100")
# Verify within budget
```

**Step 3:** Find Instance
```bash
python scripts/vast_launch.py search -g <PREFERRED_GPU> --max-price <BID_LIMIT>

# Review options, note instance ID
```

**Step 4:** Launch
```bash
# Local quick test (CPU): skip training to validate wiring
python scripts/run_fast_to_sota.py \
  --train-config <TRAIN_CONFIG> \
  --small-eval-config <SMALL_EVAL_CONFIG> \
  --full-eval-config <FULL_EVAL_CONFIG> \
  --skip-training --skip-small-eval --skip-full-eval

# Production launch on Vast (auto-shutdown when finished)
python scripts/vast_launch.py launch \
  --gpu <PREFERRED_GPU> \
  --config <TRAIN_CONFIG> \
  --auto-shutdown \
  --run-arg=--wandb-run-name=<WANDB_RUN_NAME> \
  --run-arg=--small-eval-config=<SMALL_EVAL_CONFIG> \
  --run-arg=--full-eval-config=<FULL_EVAL_CONFIG> \
  --run-arg=--leaderboard-wandb \
  --run-arg=--leaderboard-wandb-project=<WANDB_PROJECT> \
  --run-arg=--leaderboard-wandb-entity=<WANDB_ENTITY> \
  --run-arg=--tag=config=<CONFIG_LABEL>

# Note instance ID and orchestrator run_id printed after launch
```

**Step 5:** Verify Launch
```bash
# Wait 2-3 minutes for startup
python scripts/monitor_instance.sh <instance_id>

# Check GPU utilization is >0%
# Check logs show "Training started"
```

**Step 6:** Document
```bash
# Add to tracking sheet/notion/etc:
# - Instance ID
# - Run ID
# - Config used
# - Expected completion time
# - Estimated cost
```

**Rollback Procedure:**
```bash
# If launch fails:
vastai stop instance <instance_id>
vastai destroy instance <instance_id>
# Investigate issue before relaunching
```

### OP-003: Resume / Re-run a Stage

**Frequency:** As needed | **Risk:** Low

The orchestrator now records checkpoint metadata (`checkpoints/metadata.json`) plus the training W&B run descriptor (`artifacts/wandb_run.json`). Use these flags to resume cleanly:

| Scenario | Flags to pass to `run_fast_to_sota.py` |
|----------|-----------------------------------------|
| Training already finished, just re-evaluate | `--skip-training` (auto-reuses metrics unless you also pass `--redo-small-eval`/`--redo-full-eval`) |
| Force a fresh training pass | `--force-train` |
| Re-run proxy eval and overwrite cached metrics | `--redo-small-eval` |
| Re-run full eval (e.g., leaderboards out of sync) | `--redo-full-eval` |

Example (reuse training, redo full eval only):
```bash
python scripts/run_fast_to_sota.py \
  --train-config <TRAIN_CONFIG> \
  --small-eval-config <SMALL_EVAL_CONFIG> \
  --full-eval-config <FULL_EVAL_CONFIG> \
  --skip-training \
  --redo-full-eval \
  --wandb-sync \
  --leaderboard-wandb
```

If an instance died mid-run, relaunch on Vast with the same config and add `--skip-training` (or `--force-train`) in the `--run-arg` list depending on what completed before failure.

### OP-002: Monitor Active Training

**Frequency:** Every 30 minutes | **Duration:** 5 minutes | **Risk:** Low

**Procedure:**

**Step 1:** Check Instance Status
```bash
vastai show instances | grep -E "running|<your_instance_id>"

# Verify status is "running"
```

**Step 2:** Check GPU Utilization
```bash
python scripts/monitor_instance.sh <instance_id>

# Expected: 70-100% GPU utilization
# If <50% for >10min, investigate
```

**Step 3:** Check WandB
```bash
# Open: https://wandb.ai/<entity>/universal-simulator

# Verify:
# - Loss is decreasing
# - Grad norms are stable
# - No errors in logs
```

**Step 4:** Check Time/Cost
```bash
# Calculate elapsed time
START_TIME="2025-10-16 10:00"
CURRENT_TIME=$(date)
ELAPSED_MIN=$(( ($(date -d "$CURRENT_TIME" +%s) - $(date -d "$START_TIME" +%s)) / 60 ))

# Calculate current cost
GPU_PRICE=1.89  # $/hr
COST=$(echo "scale=2; $GPU_PRICE * $ELAPSED_MIN / 60" | bc)

echo "Elapsed: ${ELAPSED_MIN} min, Cost: $${COST}"
```

**Alerts:**
- GPU util <50% for >10min → Investigate
- Loss not decreasing for >1 hour → Check convergence
- Cost >150% of estimate → Check if stuck

### OP-003: Retrieve Results

**Frequency:** After training completes | **Duration:** 10 minutes | **Risk:** Low

**Procedure:**

**Step 1:** Verify Completion
```bash
# Check WandB for "finished" status
wandb runs --project universal-simulator --state finished | grep <run_id>
```

**Step 2:** Download Checkpoints
```bash
# Checkpoints are auto-uploaded to WandB
wandb artifact get <entity>/universal-simulator/<run_id>-checkpoints:latest \
    --root checkpoints/

# Or download from instance before destroying:
vastai ssh <instance_id> "tar -czf checkpoints.tar.gz checkpoints/"
scp vast@<instance_ip>:~/checkpoints.tar.gz ./
```

**Step 3:** Analyze Results
```bash
python scripts/analyze_run.py <run_id> --output reports/<run_id>_analysis.md

# Review:
# - Final metrics
# - Convergence status
# - Recommendations
```

**Step 4:** Stop Instance
```bash
# If auto-shutdown didn't trigger:
vastai stop instance <instance_id>

# Verify stopped (not destroyed) for audit trail
vastai show instances | grep <instance_id>
```

**Step 5:** Archive
```bash
# Move reports to permanent storage
mkdir -p archive/runs/<run_id>/
mv reports/<run_id>_analysis.md archive/runs/<run_id>/
mv checkpoints/ archive/runs/<run_id>/checkpoints/

# Update tracking sheet with:
# - Final cost
# - Final metrics
# - Link to analysis
```

---

## Launch Procedures

### PROC-001: First Production Launch

**Prerequisites:**
- [ ] All validation checks pass
- [ ] Dry-run completed
- [ ] Stakeholders notified
- [ ] Budget approved

**Critical Path:**

1. **T-60min:** Validate all components
   ```bash
   python scripts/validate_config.py configs/train_burgers_32dim.yaml
   python scripts/validate_data.py configs/train_burgers_32dim.yaml
   python scripts/dry_run.py configs/train_burgers_32dim.yaml
   ```

2. **T-30min:** Find instance
   ```bash
   python scripts/vast_launch.py --search-only --min-gpu-ram 24
   ```

3. **T-15min:** Launch
   ```bash
   python scripts/vast_launch.py \
       --config configs/train_burgers_32dim.yaml \
       --auto-shutdown
   ```

4. **T+0:** Verify launch
   ```bash
   # Monitor for first 10 minutes
   watch -n 30 'python scripts/monitor_instance.sh <instance_id>'
   ```

5. **T+10min:** First checkpoint
   - Verify latent cache created
   - Verify operator training started
   - GPU utilization >70%

6. **T+30min:** Periodic check
   - Loss decreasing
   - No errors in logs

7. **T+completion:** Retrieve and analyze
   - Follow OP-003 procedure

**Abort Criteria:**
- GPU util <50% for >5min in first 15min
- Any Python traceback in logs
- Cost exceeding 120% of estimate

**Go/No-Go Checklist:**
- [ ] Config validated (all 27 checks passed)
- [ ] Data validated (integrity checks passed)
- [ ] Dry-run successful
- [ ] Budget confirmed
- [ ] Monitor ready
- [ ] Rollback plan clear

### PROC-002: Emergency Instance Recreation

**When:** Instance becomes unresponsive or corrupted

**Steps:**

1. **Preserve State**
   ```bash
   # Try to save logs
   vastai logs <old_instance_id> > emergency_logs_$(date +%s).txt
   
   # Try to save checkpoints
   vastai ssh <old_instance_id> "tar -czf /tmp/emergency_ckpt.tar.gz checkpoints/" || true
   ```

2. **Destroy Old Instance**
   ```bash
   vastai destroy instance <old_instance_id>
   ```

3. **Launch New Instance**
   ```bash
   # Same config, will resume from WandB checkpoints if available
   python scripts/vast_launch.py --config <same_config> --auto-shutdown
   ```

4. **Restore Checkpoints** (if saved)
   ```bash
   # If emergency checkpoints saved:
   scp emergency_ckpt.tar.gz vast@<new_instance_ip>:/workspace/universal_simulator/
   vastai ssh <new_instance_id> "cd /workspace/universal_simulator && tar -xzf emergency_ckpt.tar.gz"
   ```

5. **Resume Training**
   ```bash
   # Training script auto-detects checkpoints and resumes
   vastai ssh <new_instance_id> "cd /workspace/universal_simulator && python scripts/train.py <config>"
   ```

---

## Monitoring

### MON-001: Health Check Dashboard

**Check Every:** 30 minutes during active training

**Metrics to Monitor:**

| Metric | Healthy Range | Warning | Critical |
|--------|---------------|---------|----------|
| GPU Utilization | 70-100% | 50-70% | <50% |
| GPU Memory | 50-90% | 90-95% | >95% |
| Operator Loss | Decreasing | Flat for 30min | Increasing |
| Diffusion Loss | <0.01 final | 0.01-0.05 | >0.05 |
| Grad Norm (Op) | 0.01-0.2 | 0.2-1.0 | >1.0 or <0.001 |
| Grad Norm (Diff) | 0.05-0.5 | 0.5-2.0 | >2.0 |
| Cost vs Estimate | 80-120% | 120-150% | >150% |

**Tools:**
```bash
# Quick health check
python scripts/monitor_instance.sh <instance_id>

# Detailed analysis
python scripts/analyze_run.py <run_id>

# WandB dashboard
open https://wandb.ai/<entity>/universal-simulator/runs/<run_id>
```

### MON-002: Cost Tracking

**Daily Review:**
```bash
# List all running instances
vastai show instances | grep running

# Calculate total daily cost
vastai show instances --raw | jq -r '.[] | select(.actual_status=="running") | "\(.id) \(.dph_total)"' | awk '{sum+=$2} END {print "Daily cost: $" sum*24}'
```

**Weekly Report:**
- Total runs launched
- Total cost incurred
- Cost per successful run
- Failed run cost (waste)

---

## Incident Response

### INC-001: Training Failure Mid-Run

**Detection:** Loss spikes, NaN values, or crash

**Response:**

1. **Assess Severity**
   ```bash
   # Check if recoverable
   vastai logs <instance_id> | tail -100 | grep -i "error\|exception\|nan"
   ```

2. **Preserve Evidence**
   ```bash
   # Save logs
   vastai logs <instance_id> > incident_<timestamp>.log
   
   # Download checkpoint
   vastai ssh <instance_id> "tar -czf /tmp/incident_ckpt.tar.gz checkpoints/"
   ```

3. **Stop Instance**
   ```bash
   vastai stop instance <instance_id>
   ```

4. **Analyze Root Cause**
   ```bash
   # Check for common issues:
   grep "CUDA out of memory" incident_<timestamp>.log  # OOM
   grep "dimension mismatch" incident_<timestamp>.log  # Config error
   grep "NaN" incident_<timestamp>.log                  # Numerical instability
   ```

5. **Fix and Relaunch**
   - OOM → Reduce batch size
   - Dimension mismatch → Validate config
   - NaN → Reduce LR, add gradient clipping
   
6. **Document**
   - Add to incident log
   - Update runbook if new issue type

### INC-002: Data Corruption Detected

**Detection:** Validate_data.py fails or training data errors

**Response:**

1. **Quarantine Bad Data**
   ```bash
   mv data/pdebench/suspect_file.h5 data/quarantine/
   ```

2. **Re-download**
   ```bash
   python scripts/validate_data.py --data-root data/pdebench --task burgers1d --split train
   
   # If fails, re-download:
   rclone copy B2:pdebench/full/burgers1d/burgers1d_train_000.h5 data/pdebench/ --progress
   ```

3. **Verify Integrity**
   ```bash
   python scripts/validate_data.py --data-root data/pdebench --task burgers1d --all-splits
   ```

4. **Resume Training**
   ```bash
   # Delete any corrupt cache
   rm -rf data/latent_cache/*
   
   # Relaunch with clean cache
   RESET_CACHE=1 python scripts/train.py configs/train_burgers_32dim.yaml
   ```

---

## Maintenance

### MAINT-001: Weekly Cleanup

**Schedule:** Every Monday | **Duration:** 30 minutes

**Tasks:**

1. **Clean up stopped instances**
   ```bash
   # List stopped instances >7 days old
   vastai show instances | grep stopped
   
   # Destroy after verification
   vastai destroy instance <instance_id>
   ```

2. **Archive old runs**
   ```bash
   # Move runs >30 days old to cold storage
   find archive/runs/ -mtime +30 -type d
   ```

3. **Clean local cache**
   ```bash
   # Clear old latent cache
   find data/latent_cache/ -mtime +14 -delete
   
   # Clear old wandb cache
   wandb artifact cache cleanup 30d
   ```

4. **Update dependencies**
   ```bash
   # Check for security updates
   pip list --outdated | grep -E "torch|wandb|numpy"
   
   # Update if needed (test first!)
   pip install --upgrade <package>
   ```

### MAINT-002: Monthly Review

**Schedule:** First Monday of month | **Duration:** 2 hours

**Tasks:**

1. **Review all incidents from last month**
   - Common failure modes?
   - Need runbook updates?

2. **Review cost efficiency**
   - Average cost per run
   - Failed run costs
   - Optimization opportunities

3. **Review config usage**
   - Which configs most used?
   - Any deprecated configs still in use?

4. **Update documentation**
   - New best practices discovered?
   - Benchmarks need updating?

5. **Test disaster recovery**
   - Can we recreate instance quickly?
   - Are backups accessible?

---

## Appendix

### A. Contact Information

**On-Call Rotation:** (Add your team's contact info)

**Escalation Path:**
1. Check runbook
2. Check troubleshooting in playbook
3. Check GitHub issues
4. Contact team lead

### B. Reference Commands

**Quick Instance Commands:**
```bash
# List all instances
vastai show instances

# SSH to instance
vastai ssh <instance_id>

# Stop instance
vastai stop instance <instance_id>

# Start stopped instance
vastai start instance <instance_id>

# Destroy instance
vastai destroy instance <instance_id>

# View logs
vastai logs <instance_id>
```

**Quick Monitoring:**
```bash
# GPU status
nvidia-smi

# Process list
ps aux | grep python

# Disk usage
df -h

# Network usage
ifstat 1
```

### C. Useful Links

- WandB Dashboard: https://wandb.ai/<entity>/universal-simulator
- VastAI Console: https://cloud.vast.ai/console/
- B2 Dashboard: https://backblaze.com/b2/
- GitHub: https://github.com/emgun/universal_simulator

---

**Last Updated:** 2025-10-16  
**Next Review:** 2025-11-16  
**Maintained By:** ML Engineering Team

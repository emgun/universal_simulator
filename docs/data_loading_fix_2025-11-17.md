# Data Loading Fix Summary (2025-11-17)

## Problem Statement

VastAI training instances were failing during data download with timeout errors:
```
📥 Downloading data for tasks: burgers1d
❌ Download timeout
total 0
```

## Root Cause Analysis

### Issue 1: VastAI Environment Variable Inheritance
**Problem:** VastAI environment variables set via `vastai create env-var` are stored in the container environment (PID 1) but are **NOT** automatically inherited by bash scripts.

**Evidence:**
- Container environment (PID 1): 8 env vars present ✓
- Bash scripts: 0 env vars present ✗

**Impact:** The `setup_vast_data.sh` script tried to use `$B2_KEY_ID` which was undefined, causing rclone to fail silently.

### Issue 2: Incorrect B2 Bucket Path
**Problem:** Script used `pdebench` (lowercase) but actual bucket is `PDEbench` (capital P and D).

**Impact:** rclone couldn't find the files even when credentials were working.

### Issue 3: Inconsistent File Naming
**Problem:** B2 bucket has inconsistent file naming:
- Some files: `burgers1d_train.h5`
- Other files: `burgers1d_train_000.h5`

**Impact:** Single-pattern download failed for files with `_000` suffix.

### Issue 4: Overcomplicated Timeout Loop
**Problem:** Script had a 60-iteration wait loop (5 min timeout) instead of immediate error reporting.

**Impact:** Failed downloads took 5 minutes to report errors, slowing debugging.

## Solution

### Fix 1: Explicit Environment Variable Sourcing
```bash
# Extract env vars from container init process (PID 1)
if [ -z "${B2_KEY_ID:-}" ]; then
  eval $(cat /proc/1/environ | tr '\0' '\n' | grep -E '^(B2_|WANDB_)' | sed 's/^/export /')
fi
```

This reads environment variables directly from the container's init process and exports them to the current bash session.

### Fix 2: Validate All Required Credentials
```bash
for var in B2_KEY_ID B2_APP_KEY B2_S3_ENDPOINT B2_S3_REGION; do
  if [ -z "${!var:-}" ]; then
    echo "❌ ERROR: Required variable $var is not set"
    exit 1
  fi
done
```

Fail fast with clear error messages if credentials are missing.

### Fix 3: Correct Bucket Path and Multi-Pattern Download
```bash
for pattern in "${task}_train.h5" "${task}_train_000.h5"; do
  if rclone copyto "B2TRAIN:PDEbench/full/$task/$pattern" "$target_file" --progress --retries 3 2>/dev/null; then
    # Success
    downloaded=true
    break
  fi
done
```

- Changed `pdebench` → `PDEbench` (case-sensitive bucket name)
- Try multiple file patterns to handle inconsistent naming
- Use `rclone copyto` to normalize output filenames

### Fix 4: Immediate Error Reporting
Removed 60-iteration wait loop, replaced with immediate validation and error reporting.

## Test Results

**Before Fix:**
```
❌ Download timeout
total 0
```

**After Fix:**
```
✓ Loaded credentials from container environment
✓ All B2 credentials validated
  ✓ burgers1d (1.6G)
✓ All data files downloaded successfully
total 1.6G
-rw-r--r-- 1 root root 1.6G Oct  7 22:30 burgers1d_train.h5
```

Download completed successfully in **29 seconds** at **~60 MB/s**.

## Files Modified

1. **scripts/setup_vast_data.sh** - Complete rewrite with proper error handling
   - Added environment variable sourcing from container
   - Added credential validation
   - Fixed bucket path (PDEbench)
   - Added multi-pattern file handling
   - Removed timeout loop, added immediate error reporting

## Commits

1. `ce1be3a` - Fix: Bombproof VastAI data loading with proper env-var handling
2. `0c167bc` - Fix: Correct B2 bucket path and handle inconsistent file naming

## Key Takeaways

1. **VastAI env-vars are not magic** - They must be explicitly sourced from `/proc/1/environ`
2. **Validate early, fail fast** - Check all prerequisites before attempting operations
3. **Be case-sensitive** - Cloud storage paths are case-sensitive (PDEbench vs pdebench)
4. **Handle real-world inconsistencies** - Data in the wild has inconsistent naming
5. **Simple is better** - Removed overcomplicated timeout loop for immediate reporting

## Related Documentation

- VastAI environment setup: `docs/vastai_env_setup.md`
- Production workflow: `PRODUCTION_WORKFLOW.md`
- Data management: `CLAUDE.md` (Data Management section)

## Future Improvements

1. Consider moving to WandB artifacts for all data (train/val/test) to eliminate B2 dependency
2. Add checksum validation for downloaded files
3. Add parallel download support for multi-task training
4. Create a data download monitoring dashboard

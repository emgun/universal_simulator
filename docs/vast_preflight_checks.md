# VastAI Pre-Flight Checks

## Problem

Previously, launching VastAI instances would often fail with "Config file not found" errors when:
1. Config files were created locally but not committed to git
2. Config files had uncommitted changes
3. Config files were committed but not pushed to remote

This caused wasted time and compute costs as instances would fail after setup completes.

## Solution

Added comprehensive pre-flight checks to `scripts/vast_launch.py` that validate config files before launching instances.

## Implementation

The `cmd_launch()` function now includes three validation checks:

### 1. Check if config is tracked in git
```python
result = subprocess.run(
    ["git", "ls-files", "--error-unmatch", str(config_path.relative_to(REPO_ROOT))],
    cwd=REPO_ROOT,
    capture_output=True,
    text=True
)
```

**Warning**: If config is not tracked, warns user and offers to abort.

### 2. Check for uncommitted changes
```python
result = subprocess.run(
    ["git", "diff", "--quiet", str(config_path.relative_to(REPO_ROOT))],
    cwd=REPO_ROOT
)
```

**Warning**: If config has uncommitted changes, warns that VastAI will use the committed version.

### 3. Check if config is pushed to remote
```python
result = subprocess.run(
    ["git", "diff", "--quiet", f"origin/{branch}", "HEAD", "--", str(config_path.relative_to(REPO_ROOT))],
    cwd=REPO_ROOT
)
```

**Warning**: If config has unpushed commits, warns that VastAI will use the older remote version.

## Usage

The checks run automatically before every launch:

```bash
python scripts/vast_launch.py launch --config configs/my_config.yaml
```

Output:
```
🔍 Pre-flight checks...
⚠️  WARNING: Config file has uncommitted changes: configs/my_config.yaml
   VastAI will use the committed version, not your local changes!
   Please run: git commit -am 'Update config' && git push origin feature--UPT
   Continue anyway? [y/N]:
```

## Benefits

1. **Prevents wasted compute**: Catches config issues before launching expensive instances
2. **Clear error messages**: Provides specific instructions on how to fix issues
3. **User control**: Allows users to abort or continue with warnings
4. **Zero false positives**: Only validates the specific config file being used

## Testing

Tested with:
- ✅ Committed and pushed config (passes silently)
- ✅ Uncommitted changes (warns and allows abort)
- ✅ Unpushed commits (warns and allows abort)
- ✅ Untracked file (warns and allows abort)

## Commit

Implemented in commit `3fa4a5b`: "Add pre-flight git checks to vast_launch.py"

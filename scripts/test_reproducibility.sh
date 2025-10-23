#!/bin/bash
# Test reproducibility of training runs with fixed seed
#
# Usage:
#   ./scripts/test_reproducibility.sh [config_path] [num_runs]
#
# Example:
#   ./scripts/test_reproducibility.sh configs/train_burgers_golden.yaml 3

set -e  # Exit on error

CONFIG=${1:-configs/train_burgers_golden.yaml}
NUM_RUNS=${2:-3}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="reproducibility_test_${TIMESTAMP}"

echo "========================================"
echo "REPRODUCIBILITY TEST"
echo "========================================"
echo "Config: $CONFIG"
echo "Runs: $NUM_RUNS"
echo "Output: $OUTPUT_DIR"
echo "========================================"
echo

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Extract seed from config
SEED=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['seed'])" 2>/dev/null || echo "42")
echo "Using seed: $SEED"
echo

# Run training multiple times
for i in $(seq 1 $NUM_RUNS); do
    echo "----------------------------------------"
    echo "RUN $i/$NUM_RUNS"
    echo "----------------------------------------"

    RUN_DIR="$OUTPUT_DIR/run_$i"
    mkdir -p "$RUN_DIR"

    # Run training and capture output
    echo "Starting training (seed=$SEED)..."
    python scripts/train.py \
        --config "$CONFIG" \
        --stage all \
        2>&1 | tee "$RUN_DIR/training.log"

    # Copy checkpoint files
    if [ -d "checkpoints" ]; then
        cp checkpoints/op_latest.ckpt "$RUN_DIR/op_latest.ckpt" 2>/dev/null || true
        cp checkpoints/diff_latest.ckpt "$RUN_DIR/diff_latest.ckpt" 2>/dev/null || true
        cp checkpoints/distill_latest.ckpt "$RUN_DIR/distill_latest.ckpt" 2>/dev/null || true
    fi

    echo "✓ Run $i completed"
    echo
done

echo "========================================"
echo "ALL RUNS COMPLETE"
echo "========================================"
echo

# Extract final metrics from logs
echo "Extracting metrics..."
python3 << 'EOF'
import re
import sys
from pathlib import Path
import json

output_dir = Path(sys.argv[1])
num_runs = int(sys.argv[2])

results = []
for i in range(1, num_runs + 1):
    log_file = output_dir / f"run_{i}" / "training.log"
    if not log_file.exists():
        continue

    log_text = log_file.read_text()

    # Extract operator final loss
    op_match = re.search(r'Operator.*final.*loss.*?(\d+\.\d+e[+-]\d+|\d+\.\d+)', log_text, re.IGNORECASE)
    op_loss = float(op_match.group(1)) if op_match else None

    # Extract diffusion final loss
    diff_match = re.search(r'Diffusion.*final.*loss.*?(\d+\.\d+e[+-]\d+|\d+\.\d+)', log_text, re.IGNORECASE)
    diff_loss = float(diff_match.group(1)) if diff_match else None

    # Extract baseline NRMSE
    nrmse_match = re.search(r'Baseline NRMSE.*?(\d+\.\d+)', log_text, re.IGNORECASE)
    nrmse = float(nrmse_match.group(1)) if nrmse_match else None

    results.append({
        "run": i,
        "operator_loss": op_loss,
        "diffusion_loss": diff_loss,
        "baseline_nrmse": nrmse,
    })

# Save results
results_file = output_dir / "results.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

# Print summary
print("\n=== RESULTS SUMMARY ===\n")
print(f"{'Run':<6} {'Op Loss':<12} {'Diff Loss':<12} {'NRMSE':<10}")
print("-" * 45)

op_losses = []
diff_losses = []
nrmses = []

for r in results:
    op_str = f"{r['operator_loss']:.6f}" if r['operator_loss'] else "N/A"
    diff_str = f"{r['diffusion_loss']:.6f}" if r['diffusion_loss'] else "N/A"
    nrmse_str = f"{r['baseline_nrmse']:.6f}" if r['baseline_nrmse'] else "N/A"

    print(f"{r['run']:<6} {op_str:<12} {diff_str:<12} {nrmse_str:<10}")

    if r['operator_loss']:
        op_losses.append(r['operator_loss'])
    if r['diffusion_loss']:
        diff_losses.append(r['diffusion_loss'])
    if r['baseline_nrmse']:
        nrmses.append(r['baseline_nrmse'])

print("-" * 45)

# Calculate variance
import statistics

if len(op_losses) > 1:
    op_mean = statistics.mean(op_losses)
    op_std = statistics.stdev(op_losses)
    op_cv = (op_std / op_mean) * 100 if op_mean > 0 else 0
    print(f"\nOperator Loss CV: {op_cv:.2f}%")

if len(diff_losses) > 1:
    diff_mean = statistics.mean(diff_losses)
    diff_std = statistics.stdev(diff_losses)
    diff_cv = (diff_std / diff_mean) * 100 if diff_mean > 0 else 0
    print(f"Diffusion Loss CV: {diff_cv:.2f}%")

if len(nrmses) > 1:
    nrmse_mean = statistics.mean(nrmses)
    nrmse_std = statistics.stdev(nrmses)
    nrmse_cv = (nrmse_std / nrmse_mean) * 100 if nrmse_mean > 0 else 0
    print(f"Baseline NRMSE CV: {nrmse_cv:.2f}%")

    if nrmse_cv < 2.0:
        print("\n✅ EXCELLENT: NRMSE variance < 2% (highly reproducible)")
    elif nrmse_cv < 5.0:
        print("\n✅ GOOD: NRMSE variance < 5% (reproducible)")
    elif nrmse_cv < 10.0:
        print("\n⚠️  MODERATE: NRMSE variance < 10% (some variability)")
    else:
        print(f"\n❌ POOR: NRMSE variance = {nrmse_cv:.1f}% (not reproducible)")

print(f"\nResults saved to: {results_file}")

EOF

python3 - "$OUTPUT_DIR" "$NUM_RUNS"

echo
echo "========================================"
echo "TEST COMPLETE"
echo "========================================"
echo "Results directory: $OUTPUT_DIR"
echo

#!/bin/bash
# SOTA Sweep Launch Script
# Launches 6 parallel Vast.ai instances for comprehensive sweep
# Target: nRMSE < 0.035

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  SOTA Sweep Launch - 6 Parallel Runs  ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Baseline: rerun_txxoc8a8 (nRMSE ~0.09)"
echo "Target: nRMSE < 0.035"
echo ""

# Round A: Optimizer sweep (3 runs) - smallest models, fastest
echo -e "${GREEN}[Round A] Launching Optimizer Sweep (3 instances)${NC}"
echo "  - sweep_round_a_lr2e4_w3: LR=2e-4, Warmup=3%, EMA=0.9995"
echo "  - sweep_round_a_lr3e4_w5: LR=3e-4, Warmup=5%, EMA=0.9995"
echo "  - sweep_round_a_lr45e4_w5: LR=4.5e-4, Warmup=5%, EMA=0.9999"
echo ""

# Launch Round A - Run 1
echo -e "${YELLOW}Launching Round A - Run 1 (LR 2e-4)...${NC}"
python scripts/vast_launch.py launch \
  --config configs/sweep_round_a_lr2e4_w3.yaml \
  --gpu RTX_A6000 \
  --disk 128 \
  --auto-shutdown \
  --run-arg='--force-full-eval' \
  --run-arg='--leaderboard-wandb' &

LAUNCH_PID_A1=$!
sleep 10  # Stagger launches to avoid API rate limits

# Launch Round A - Run 2
echo -e "${YELLOW}Launching Round A - Run 2 (LR 3e-4)...${NC}"
python scripts/vast_launch.py launch \
  --config configs/sweep_round_a_lr3e4_w5.yaml \
  --gpu RTX_A6000 \
  --disk 128 \
  --auto-shutdown \
  --run-arg='--force-full-eval' \
  --run-arg='--leaderboard-wandb' &

LAUNCH_PID_A2=$!
sleep 10

# Launch Round A - Run 3
echo -e "${YELLOW}Launching Round A - Run 3 (LR 4.5e-4)...${NC}"
python scripts/vast_launch.py launch \
  --config configs/sweep_round_a_lr45e4_w5.yaml \
  --gpu RTX_A6000 \
  --disk 128 \
  --auto-shutdown \
  --run-arg='--force-full-eval' \
  --run-arg='--leaderboard-wandb' &

LAUNCH_PID_A3=$!
sleep 10

# Round B: Capacity sweep (2 runs) - medium models
echo ""
echo -e "${GREEN}[Round B] Launching Capacity Sweep (2 instances)${NC}"
echo "  - sweep_round_b_capacity_up: Tokens=32, Hidden=96"
echo "  - sweep_round_b_deeper: Depths=[2,2,2], Tokens=24, Hidden=80"
echo ""

# Launch Round B - Run 1
echo -e "${YELLOW}Launching Round B - Run 1 (Capacity Up)...${NC}"
python scripts/vast_launch.py launch \
  --config configs/sweep_round_b_capacity_up.yaml \
  --gpu RTX_A6000 \
  --disk 128 \
  --auto-shutdown \
  --run-arg='--force-full-eval' \
  --run-arg='--leaderboard-wandb' &

LAUNCH_PID_B1=$!
sleep 10

# Launch Round B - Run 2
echo -e "${YELLOW}Launching Round B - Run 2 (Deeper)...${NC}"
python scripts/vast_launch.py launch \
  --config configs/sweep_round_b_deeper.yaml \
  --gpu RTX_A6000 \
  --disk 128 \
  --auto-shutdown \
  --run-arg='--force-full-eval' \
  --run-arg='--leaderboard-wandb' &

LAUNCH_PID_B2=$!
sleep 10

# Aggressive: Large model
echo ""
echo -e "${GREEN}[Aggressive] Launching Large Model (1 instance)${NC}"
echo "  - sweep_aggressive_sota_48gb: Dim=40, Tokens=40, Hidden=112, Extended training"
echo ""

# Launch Aggressive
echo -e "${YELLOW}Launching Aggressive SOTA Push (48GB optimized)...${NC}"
python scripts/vast_launch.py launch \
  --config configs/sweep_aggressive_sota_48gb.yaml \
  --gpu RTX_A6000 \
  --disk 128 \
  --auto-shutdown \
  --run-arg='--force-full-eval' \
  --run-arg='--leaderboard-wandb' &

LAUNCH_PID_AGG=$!

echo ""
echo -e "${GREEN}All 6 launch processes started!${NC}"
echo ""
echo "Launch PIDs:"
echo "  Round A-1: $LAUNCH_PID_A1"
echo "  Round A-2: $LAUNCH_PID_A2"
echo "  Round A-3: $LAUNCH_PID_A3"
echo "  Round B-1: $LAUNCH_PID_B1"
echo "  Round B-2: $LAUNCH_PID_B2"
echo "  Aggressive: $LAUNCH_PID_AGG"
echo ""
echo "Waiting for all launch processes to complete..."
wait

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  All instances launched successfully!  ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Monitor instances: vastai show instances"
echo "  2. Check W&B: https://wandb.ai/emgun-morpheus-space/universal-simulator"
echo "  3. View leaderboard when complete: cat reports/leaderboard.csv"
echo ""
echo "Expected completion: 12-16 hours"
echo "Estimated cost: ~$25-30"

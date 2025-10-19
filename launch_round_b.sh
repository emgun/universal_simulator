#!/bin/bash
# Round B Launch: Capacity Sweep (2 instances)
# Medium VRAM, capacity scaling experiments

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}[Round B] Launching Capacity Sweep (2 instances)${NC}"
echo "Target GPU: RTX A6000 (48GB)"
echo "Expected runtime: 9-11 hours each"
echo "Expected cost: ~$4 per instance = $8 total"
echo ""

# Launch Run 1: Capacity Up (32 tokens, 96 hidden)
echo -e "${YELLOW}[1/2] Launching: Capacity Up (tokens=32, hidden=96)${NC}"
python scripts/vast_launch.py launch \
  --config configs/sweep_round_b_capacity_up.yaml \
  --gpu RTX_A6000 \
  --disk 128 \
  --auto-shutdown \
  --run-arg='--force-full-eval' \
  --run-arg='--leaderboard-wandb'

echo ""
sleep 15

# Launch Run 2: Deeper (depths [2,2,2])
echo -e "${YELLOW}[2/2] Launching: Deeper (depths=[2,2,2], tokens=24, hidden=80)${NC}"
python scripts/vast_launch.py launch \
  --config configs/sweep_round_b_deeper.yaml \
  --gpu RTX_A6000 \
  --disk 128 \
  --auto-shutdown \
  --run-arg='--force-full-eval' \
  --run-arg='--leaderboard-wandb'

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Round B launched successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

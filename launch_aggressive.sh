#!/bin/bash
# Aggressive SOTA Launch: Large model with extended training
# High VRAM, longest runtime

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}[Aggressive] Launching Large SOTA Model${NC}"
echo "Target GPU: RTX A6000 (48GB)"
echo "Config: dim=40, tokens=40, hidden=112, depths=[2,2,2]"
echo "Expected runtime: 14-18 hours"
echo "Expected cost: ~$7"
echo ""

echo -e "${YELLOW}Launching: Aggressive SOTA (48GB optimized)${NC}"
python scripts/vast_launch.py launch \
  --config configs/sweep_aggressive_sota_48gb.yaml \
  --gpu RTX_A6000 \
  --disk 128 \
  --auto-shutdown \
  --run-arg='--force-full-eval' \
  --run-arg='--leaderboard-wandb'

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Aggressive model launched!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

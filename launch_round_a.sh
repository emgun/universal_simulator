#!/bin/bash
# Round A Launch: Optimizer Sweep (3 instances)
# Conservative, lowest VRAM, fastest iteration

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}[Round A] Launching Optimizer Sweep (3 instances)${NC}"
echo "Target GPU: RTX A6000 (48GB)"
echo "Expected runtime: 8-10 hours each"
echo "Expected cost: ~$3 per instance = $9 total"
echo ""

# Launch Run 1: LR 2e-4, Warmup 3%
echo -e "${YELLOW}[1/3] Launching: LR=2e-4, Warmup=3%, EMA=0.9995${NC}"
python scripts/vast_launch.py launch \
  --config configs/sweep_round_a_lr2e4_w3.yaml \
  --gpu RTX_A6000 \
  --disk 128 \
  --auto-shutdown \
  --run-arg='--force-full-eval' \
  --run-arg='--leaderboard-wandb'

echo ""
sleep 15  # Stagger to avoid API rate limits

# Launch Run 2: LR 3e-4, Warmup 5%
echo -e "${YELLOW}[2/3] Launching: LR=3e-4, Warmup=5%, EMA=0.9995${NC}"
python scripts/vast_launch.py launch \
  --config configs/sweep_round_a_lr3e4_w5.yaml \
  --gpu RTX_A6000 \
  --disk 128 \
  --auto-shutdown \
  --run-arg='--force-full-eval' \
  --run-arg='--leaderboard-wandb'

echo ""
sleep 15

# Launch Run 3: LR 4.5e-4, Warmup 5%, EMA 0.9999
echo -e "${YELLOW}[3/3] Launching: LR=4.5e-4, Warmup=5%, EMA=0.9999${NC}"
python scripts/vast_launch.py launch \
  --config configs/sweep_round_a_lr45e4_w5.yaml \
  --gpu RTX_A6000 \
  --disk 128 \
  --auto-shutdown \
  --run-arg='--force-full-eval' \
  --run-arg='--leaderboard-wandb'

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Round A launched successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Monitor: vastai show instances"
echo "W&B: https://wandb.ai/emgun-morpheus-space/universal-simulator"
echo ""

#!/bin/bash
# Launch sweep with explicit offer IDs to avoid instance selection issues

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Launching SOTA Sweep with Explicit Offers${NC}"
echo ""

# Good offers from search (verified, reliable, cheap)
OFFERS=(10755233 24638771 24936690 25105510 24651690)
CONFIGS=(
    "configs/sweep_round_a_lr2e4_w3.yaml"
    "configs/sweep_round_a_lr45e4_w5.yaml"
    "configs/sweep_round_b_capacity_up.yaml"
    "configs/sweep_round_b_deeper.yaml"
    "configs/sweep_aggressive_sota_48gb.yaml"
)

NAMES=(
    "Round-A-1: LR=2e-4"
    "Round-A-3: LR=4.5e-4"
    "Round-B-1: Capacity Up"
    "Round-B-2: Deeper"
    "Aggressive: Large Model"
)

for i in "${!OFFERS[@]}"; do
    OFFER="${OFFERS[$i]}"
    CONFIG="${CONFIGS[$i]}"
    NAME="${NAMES[$i]}"

    echo -e "${YELLOW}[$((i+1))/5] Launching: $NAME${NC}"
    echo "  Offer ID: $OFFER"
    echo "  Config: $CONFIG"

    python scripts/vast_launch.py launch \
        --offer-id "$OFFER" \
        --config "$CONFIG" \
        --auto-shutdown \
        --run-arg='--force-full-eval' \
        --run-arg='--leaderboard-wandb' || echo "  ⚠️  Launch failed for $NAME"

    echo ""
    sleep 5
done

echo -e "${GREEN}All instances launched!${NC}"
echo ""
echo "Monitor at: https://wandb.ai/emgun-morpheus-space/universal-simulator"
echo "Check status: vastai show instances"

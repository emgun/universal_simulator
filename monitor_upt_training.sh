#!/bin/bash
# Monitor UPT Phase 1 Training Run
# Instance ID: 27192839

INSTANCE_ID=27192839

echo "🔍 Monitoring VastAI Instance $INSTANCE_ID"
echo "=========================================="
echo ""

# Show instance status
echo "📊 Instance Status:"
vastai show instances | grep -E "ID|$INSTANCE_ID"
echo ""

# Try to SSH and check logs
echo "📋 Recent Logs (last 50 lines):"
echo "=========================================="
vastai ssh $INSTANCE_ID "tail -50 /workspace/universal_simulator/nohup.out 2>/dev/null || tail -50 /var/log/onstart.log 2>/dev/null || echo 'Logs not available yet (instance still starting up)'"
echo ""

echo "💡 Useful Commands:"
echo "  vastai show instances                          # Check status"
echo "  vastai ssh $INSTANCE_ID                        # SSH into instance"
echo "  vastai ssh $INSTANCE_ID 'tail -f nohup.out'   # Follow logs"
echo "  vastai logs $INSTANCE_ID                       # View container logs"
echo "  vastai destroy instance $INSTANCE_ID           # Manual shutdown"
echo ""

echo "🌐 WandB Dashboard:"
echo "  https://wandb.ai/emgun-morpheus-space/universal-simulator"
echo ""

echo "⏰ Auto-shutdown: Enabled (60 min timeout)"
echo "💰 Cost: $0.37/hr (~$0.25 for 25-30 min run)"

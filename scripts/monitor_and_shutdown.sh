#!/usr/bin/env bash
set -eo pipefail

# Monitor training and shutdown when complete
INSTANCE_ID=${1:-26759442}
CHECK_INTERVAL=${2:-300}  # Check every 5 minutes

echo "Monitoring instance $INSTANCE_ID for training completion..."
echo "Will shutdown automatically when:"
echo "  1. All training stages complete"
echo "  2. Evaluation finishes"
echo "  3. W&B files are synced"
echo ""

while true; do
    echo "[$(date)] Checking instance status..."
    
    # Check if training process is still running
    TRAINING_RUNNING=$(ssh -o StrictHostKeyChecking=no -p 39442 root@ssh9.vast.ai 'ps aux | grep "python.*train.py\|python.*evaluate.py" | grep -v grep | wc -l' 2>/dev/null || echo "0")
    
    if [ "$TRAINING_RUNNING" -eq 0 ]; then
        echo "[$(date)] Training/evaluation processes have stopped. Checking completion status..."
        
        # Check if evaluation completed successfully
        EVAL_COMPLETE=$(ssh -o StrictHostKeyChecking=no -p 39442 root@ssh9.vast.ai 'ls -1 /workspace/universal_simulator/reports/*eval*.json 2>/dev/null | wc -l' 2>/dev/null || echo "0")
        
        if [ "$EVAL_COMPLETE" -gt 0 ]; then
            echo "[$(date)] Evaluation results found. Waiting for W&B sync..."
            
            # Wait for W&B to finish syncing
            sleep 60
            
            # Check W&B sync status
            WANDB_SYNCING=$(ssh -o StrictHostKeyChecking=no -p 39442 root@ssh9.vast.ai 'ps aux | grep wandb | grep -v grep | wc -l' 2>/dev/null || echo "0")
            
            if [ "$WANDB_SYNCING" -eq 0 ]; then
                echo "[$(date)] ✅ Training complete, evaluation done, W&B synced!"
                echo "[$(date)] Downloading final artifacts..."
                
                # Download evaluation results
                mkdir -p reports/h200_run
                scp -P 39442 -o StrictHostKeyChecking=no "root@ssh9.vast.ai:/workspace/universal_simulator/reports/*eval*.json" reports/h200_run/ 2>/dev/null || true
                scp -P 39442 -o StrictHostKeyChecking=no "root@ssh9.vast.ai:/workspace/universal_simulator/reports/*eval*.csv" reports/h200_run/ 2>/dev/null || true
                
                # Download checkpoints
                mkdir -p checkpoints/h200_run
                scp -P 39442 -o StrictHostKeyChecking=no "root@ssh9.vast.ai:/workspace/universal_simulator/checkpoints/scale/*.pt" checkpoints/h200_run/ 2>/dev/null || true
                
                echo "[$(date)] Final artifacts downloaded to local reports/h200_run and checkpoints/h200_run"
                echo "[$(date)] 🛑 Shutting down instance $INSTANCE_ID..."
                
                vastai destroy instance "$INSTANCE_ID"
                
                echo "[$(date)] ✅ Instance shutdown complete!"
                exit 0
            else
                echo "[$(date)] W&B still syncing, waiting..."
            fi
        else
            echo "[$(date)] ⚠️ Training stopped but no evaluation results found. Checking logs..."
            ssh -o StrictHostKeyChecking=no -p 39442 root@ssh9.vast.ai 'tail -n 50 /workspace/universal_simulator/run_fast.log' || true
        fi
    else
        echo "[$(date)] Training/evaluation still running ($TRAINING_RUNNING processes active)"
        
        # Show recent progress
        ssh -o StrictHostKeyChecking=no -p 39442 root@ssh9.vast.ai 'tail -n 5 /workspace/universal_simulator/run_fast.log 2>/dev/null | grep -E "(epoch|loss|Evaluating)" | tail -3' 2>/dev/null || true
    fi
    
    echo "[$(date)] Next check in $CHECK_INTERVAL seconds..."
    echo "---"
    sleep "$CHECK_INTERVAL"
done






#!/usr/bin/env bash
set -euo pipefail

# Monitor a Vast.ai instance and tail logs once running
# Usage: ./scripts/monitor_instance.sh [INSTANCE_ID]

INSTANCE_ID="${1:-}"

if [ -z "$INSTANCE_ID" ]; then
  echo "Usage: $0 <instance_id>"
  echo ""
  echo "Finding most recent instance..."
  INSTANCE_ID=$(vastai show instances --raw | jq -r '.[0].id // empty')
  if [ -z "$INSTANCE_ID" ]; then
    echo "No instances found"
    exit 1
  fi
  echo "Using instance: $INSTANCE_ID"
fi

echo "=== Monitoring Vast.ai Instance $INSTANCE_ID ==="
echo ""

# Wait for instance to start
echo "Waiting for instance to start..."
for i in {1..60}; do
  STATUS=$(vastai show instances --raw | jq -r ".[] | select(.id==$INSTANCE_ID) | .actual_status // .cur_state" || echo "unknown")

  if [ "$STATUS" = "running" ]; then
    echo "✓ Instance is running!"
    break
  elif [ "$STATUS" = "exited" ] || [ "$STATUS" = "stopped" ]; then
    echo "⚠️  Instance has stopped/exited"
    exit 1
  fi

  echo "  Status: $STATUS (attempt $i/60)"
  sleep 10
done

if [ "$STATUS" != "running" ]; then
  echo "Instance did not start within 10 minutes"
  exit 1
fi

# Get SSH connection info
INFO=$(vastai show instances --raw | jq -r ".[] | select(.id==$INSTANCE_ID) | {ssh_host, ssh_port} | @json")
SSH_HOST=$(echo "$INFO" | jq -r .ssh_host)
SSH_PORT=$(echo "$INFO" | jq -r .ssh_port)

echo ""
echo "Instance Details:"
echo "  ID: $INSTANCE_ID"
echo "  SSH: ssh://root@$SSH_HOST:$SSH_PORT"
echo ""

# Wait a bit for onstart script to begin
echo "Waiting for onstart script to initialize (30s)..."
sleep 30

# Tail logs
echo ""
echo "=== Tailing Logs (Ctrl-C to stop) ==="
echo ""

while true; do
  vastai logs "$INSTANCE_ID" --tail 50 2>/dev/null || {
    echo "Failed to fetch logs, retrying in 20s..."
    sleep 20
    continue
  }
  sleep 20
done

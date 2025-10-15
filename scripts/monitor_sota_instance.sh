#!/usr/bin/env bash
set -euo pipefail

# Monitor SOTA evaluation instance
INSTANCE_ID=${1:-26798823}

echo "Monitoring instance $INSTANCE_ID"
echo "Press Ctrl+C to stop monitoring"
echo ""

check_count=0
max_checks=120  # 10 minutes (5 sec intervals)

while [ $check_count -lt $max_checks ]; do
  status=$(vastai show instances --raw 2>/dev/null | jq -r ".[] | select(.id==$INSTANCE_ID) | .actual_status // empty")

  if [ -z "$status" ]; then
    echo "[$(date +%H:%M:%S)] Instance not found or destroyed"
    exit 1
  fi

  echo -n "[$(date +%H:%M:%S)] Status: $status"

  if [ "$status" = "running" ]; then
    echo " ✓"
    echo ""
    echo "Instance is running! Fetching logs..."
    sleep 5
    vastai logs "$INSTANCE_ID" --tail 100
    exit 0
  elif [ "$status" = "exited" ]; then
    echo " ✗ (exited)"
    echo ""
    echo "Instance exited. Fetching final logs..."
    vastai logs "$INSTANCE_ID" --tail 200
    exit 1
  else
    echo " (waiting...)"
  fi

  sleep 5
  check_count=$((check_count + 1))
done

echo ""
echo "Timeout waiting for instance to start"
exit 1

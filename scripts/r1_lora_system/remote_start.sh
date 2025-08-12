#!/usr/bin/env bash
set -euo pipefail

echo "=== START $(date) ==="
echo "=== KILL OLD ==="
pkill -f private_train.py || true
sleep 2

echo "=== CLEAN LOG ==="
rm -f training.log || true

echo "=== CLEAN __pycache__ ==="
find ~ -name __pycache__ -type d -exec rm -rf {} + || true

echo "=== SOURCE ENV ==="
if [ -f "~/private_env.sh" ]; then
  # shellcheck disable=SC1090
  source ~/private_env.sh
fi

echo "=== START TRAIN ==="
nohup python3 private_train.py > training.log 2>&1 &
pid=$!
echo "PID: $pid"

sleep 12
echo "=== LAST LOG (120) ==="
tail -n 120 training.log || true

echo "=== GPU ==="
nvidia-smi | sed -n '1,15p' || true


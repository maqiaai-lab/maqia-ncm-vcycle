#!/usr/bin/env bash
# LM1B arm: residual diagnostic + V-cycle vs no-restriction, one seed at a time.
# Usage: experiments/run_lm1b.sh [size] [seeds] [levels]
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
size=${1:-small}
seeds=${2:-"42 0 1"}
levels=${3:-"3,6"}
mkdir -p results/logs
for seed in $seeds; do
  echo "[$(date +%H:%M:%S)] lm1b $size seed $seed"
  "$PY" experiments/lm1b_diagnostic.py --size "$size" --n 256 --seed "$seed" --vcycle "$levels" \
    > "results/logs/lm1b_${size}_s${seed}.log" 2>&1 || echo "FAILED: lm1b seed $seed"
done
echo "[$(date +%H:%M:%S)] done"

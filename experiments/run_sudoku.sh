#!/usr/bin/env bash
# Run the Sudoku pipeline one job at a time (concurrent jobs oversubscribe the CPU threads).
# Usage: experiments/run_sudoku.sh [phase ...]   phases: 0 0m 1 1b 2 3   (default: 0 1 1b 2 3)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
mkdir -p results/logs
phases=${*:-"0 1 1b 2 3"}
run() { echo "[$(date +%H:%M:%S)] $*"; "$PY" "$@" > "results/logs/$LOG.log" 2>&1 || echo "FAILED: $* (see results/logs/$LOG.log)"; }
for ph in $phases; do
  case $ph in
    0)  for d in hard medium easy; do LOG=phase0_$d run experiments/phase0_baseline.py --difficulty $d; done ;;
    1)  for d in hard medium easy; do LOG=phase1_$d run experiments/phase1_fixed_point.py --difficulty $d; done ;;
    1b) for d in hard medium easy; do LOG=phase1b_$d run experiments/phase1b_localization.py --difficulty $d; done ;;
    0m) for d in medium easy; do LOG=phase0_$d run experiments/phase0_baseline.py --difficulty $d; done ;;
    2)  LOG=phase2_hard_n1000 run experiments/phase2_vcycle.py --difficulty hard --n 1000 --cycles 3 ;;
    3)  for seed in 0 1 2; do
          LOG=phase3_hard_T1_s$seed run experiments/phase3_controls.py --difficulty hard --n 2000 --rounds 8 \
            --temp 1.0 --levels 1,3,5,7 --seed $seed --tag hard_n2000_T1.0_s$seed
        done ;;
  esac
done
echo "[$(date +%H:%M:%S)] done"

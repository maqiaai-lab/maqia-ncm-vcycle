#!/usr/bin/env bash
# GPU box: once judges.sh is done, run the jobs the laptop handed over (quality checks on all seeds), then the tables
# and charts, so the laptop only copies results back. From the repo root on the box:
#   nohup bash experiments/box/after_judges.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../.."
. .venv/bin/activate
LOG=results/logs/box_queue.log
stamp() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
until grep -qE "judges done|queue stopped" "$LOG"; do sleep 20; done
stamp "start lm1b_quality_medium"
if python experiments/lm1b_quality.py --size medium --no_mauve \
     --variants "baseline,vcycle|m=6,no-restrict|m=6,vcycle|m=8,no-restrict|m=8" > results/logs/lm1b_quality_medium.log 2>&1; then
  stamp "ok lm1b_quality_medium"
else
  stamp "FAILED lm1b_quality_medium"
fi
stamp "start analyze"
if python experiments/analyze.py > results/logs/analyze.log 2>&1; then stamp "ok analyze"; else stamp "FAILED analyze"; fi
stamp "all done"

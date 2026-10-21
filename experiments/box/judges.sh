#!/usr/bin/env bash
# GPU box, judge only (the laptop ran everything else): after experiments/box/setup.sh and a push of the finished
# results, from the repo root on the box:
#   MAX_MINUTES=120 nohup bash experiments/box/judges.sh > /dev/null 2>&1 &
# MAX_MINUTES schedules a hard shutdown (the instance's shutdown behavior is "terminate"). Progress:
# results/logs/box_queue.log. The comparisons are fixed in advance: one cycle from level m = 6 (LM1B) or m = 5
# (XSum, WMT14); the human-vs-NCM sanity check comes first.
set -uo pipefail
cd "$(dirname "$0")/../.."
. .venv/bin/activate
LOG=results/logs/box_queue.log
mkdir -p results/logs
stamp() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
run() {
  local name=$1; shift
  stamp "start $name"
  if python "$@" > "results/logs/$name.log" 2>&1; then stamp "ok $name"; return 0; fi
  stamp "FAILED $name"; return 1
}
if [ -n "${MAX_MINUTES:-}" ]; then
  sudo -n shutdown -h "+$MAX_MINUTES" > /dev/null 2>&1 && stamp "hard shutdown scheduled in $MAX_MINUTES minutes" \
    || stamp "could not schedule a shutdown"
fi
run box_check_scoring experiments/check_gpu_scoring.py --size small --seed 42 || { stamp "queue stopped"; exit 1; }
run judge_lm1b_human experiments/judge.py --task lm1b --size medium --a human --b baseline --bs 16
run judge_lm1b_vc_nr experiments/judge.py --task lm1b --size medium --a "vcycle|m=6" --b "no-restrict|m=6" --bs 16
run judge_lm1b_vc_base experiments/judge.py --task lm1b --size medium --a "vcycle|m=6" --b baseline --bs 16
run judge_xsum experiments/judge.py --task xsum --a "vcycle|m=5|c=1" --b baseline --n 2000 --bs 16
run judge_wmt14 experiments/judge.py --task wmt14 --a "vcycle|m=5|c=1" --b baseline --bs 16
stamp "judges done"

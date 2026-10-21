#!/usr/bin/env bash
# The job queue, on the GPU box or the laptop. Start it detached, from the repo root:
#   MAX_MINUTES=180 nohup bash experiments/box/queue.sh > /dev/null 2>&1 &            (GPU box)
#   SERIAL=1 SKIP_CHECK=1 SKIP_JUDGE=1 XSUM_BS=16 bash experiments/box/queue.sh      (laptop, 6 GB)
# Every job is skipped when its output is already complete, and the long jobs resume mid-way, so a run started on
# one machine continues on another after experiments/box/push.sh (results travel with the code).
# MAX_MINUTES schedules a hard shutdown, so a hung job cannot keep the machine billing (set the instance's shutdown
# behavior to "terminate" to end billing completely). Progress: results/logs/box_queue.log; one log per job.
# Two lanes, XSum and LM1B (more seeds, MAUVE, quality checks), run side by side (each job needs < 10 GB of GPU
# memory) unless SERIAL=1; the LLM-judge jobs (16 GB) and analyze.py follow. The scoring check runs first and stops
# everything if it fails. Comparisons are fixed in advance: one cycle from level m = 6 (LM1B) or m = 5 (XSum, WMT14)
# is what the judge sees.
set -uo pipefail
cd "$(dirname "$0")/../.."
if [ -f .venv/bin/activate ]; then . .venv/bin/activate; else . .venv/Scripts/activate; fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}
LOG=results/logs/box_queue.log
mkdir -p results/logs
stamp() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
run() {  # name, script and arguments
  local name=$1; shift
  stamp "start $name"
  if python "$@" > "results/logs/$name.log" 2>&1; then stamp "ok $name"; return 0; fi
  stamp "FAILED $name"; return 1
}
complete() {  # JSON file, Python expression over j that is true once the job's output is complete
  python - "$1" "$2" <<'PY'
import json, sys
try:
    j = json.load(open(sys.argv[1]))
    sys.exit(0 if eval(sys.argv[2], {"j": j, "len": len}) else 1)
except Exception:
    sys.exit(1)
PY
}
step() {  # name, output JSON, completeness expression, script and arguments
  local name=$1 out=$2 expr=$3; shift 3
  if complete "$out" "$expr"; then stamp "skip $name (complete)"; return 0; fi
  run "$name" "$@"
}
if [ -n "${MAX_MINUTES:-}" ]; then
  (sudo -n shutdown -h "+$MAX_MINUTES" || shutdown -h "+$MAX_MINUTES") > /dev/null 2>&1 \
    && stamp "hard shutdown scheduled in $MAX_MINUTES minutes" || stamp "could not schedule a shutdown (no permission)"
fi

# 1. This machine must score exactly as the committed runs did (decoders are checked against generate() in each script).
if [ -z "${SKIP_CHECK:-}" ]; then
  run box_check_scoring experiments/check_gpu_scoring.py --size small --seed 42 || { stamp "queue stopped"; exit 1; }
fi

lane_xsum() {  # XSum, the full test set (11,334 articles): levels 3 and 5, 3 cycles each; raw-restriction ablation
  step xsum_vcycle results/xsum_vcycle.json "'vcycle|m=5|c=3' in j['variants'] and 'examples' in j" \
    experiments/cond_vcycle.py --task xsum --levels 3,5 --cycles 3 --bs "${XSUM_BS:-32}"
  step xsum_vcycle_raw results/xsum_vcycle_raw.json "'vcycle|m=5|c=1' in j['variants'] and 'examples' in j" \
    experiments/cond_vcycle.py --task xsum --levels 5 --cycles 1 --raw_restrict --tag _raw --bs "${XSUM_BS:-32}"
}
lane_lm1b() {  # LM1B-medium: 17 more seeds (20 in all: 5,120 samples per variant), MAUVE, quality checks
  for seed in $(seq 2 18); do
    step "lm1b_medium_s$seed" "results/lm1b_medium_s$seed.json" "len(j.get('vcycle', {})) >= 4" \
      experiments/lm1b_diagnostic.py --size medium --n 256 --seed "$seed" --vcycle 6,8
  done
  if ! complete results/lm1b_mauve_medium.json "j.get('n_pooled') == 5120 and 'half_size' in j"; then
    rm -f results/cache/lm1b_mauve_feats_medium.pt  # features of an earlier, smaller run; recomputed for all seeds
  fi
  step lm1b_mauve_medium results/lm1b_mauve_medium.json "j.get('n_pooled') == 5120 and 'half_size' in j" \
    experiments/lm1b_mauve.py --size medium --boot 20 --variants "baseline,vcycle|m=6,no-restrict|m=6"
  step lm1b_quality_medium results/lm1b_quality_medium.json "len(j['seeds']) == 20 and len(j.get('ref2_ratios', {})) >= 4" \
    experiments/lm1b_quality.py --size medium --no_mauve \
    --variants "baseline,vcycle|m=6,no-restrict|m=6,vcycle|m=8,no-restrict|m=8"
}
if [ -n "${SERIAL:-}" ]; then
  lane_xsum
  lane_lm1b
else
  lane_xsum & p1=$!
  lane_lm1b & p2=$!
  wait $p1 $p2
fi
if [ -n "${SKIP_JUDGE:-}" ]; then stamp "stopping before the judge (SKIP_JUDGE)"; exit 0; fi

# 2. LLM judge (Qwen2.5-7B-Instruct, on this machine). First the sanity check: human text vs NCM's single pass.
jdone="'mean_preference_for_a' in j"
step judge_lm1b_human results/judge_lm1b_medium_human_vs_baseline.json "$jdone" \
  experiments/judge.py --task lm1b --size medium --a human --b baseline --bs 16
step judge_lm1b_vc_nr results/judge_lm1b_medium_vcyclem6_vs_norestrictm6.json "$jdone" \
  experiments/judge.py --task lm1b --size medium --a "vcycle|m=6" --b "no-restrict|m=6" --bs 16
step judge_lm1b_vc_base results/judge_lm1b_medium_vcyclem6_vs_baseline.json "$jdone" \
  experiments/judge.py --task lm1b --size medium --a "vcycle|m=6" --b baseline --bs 16
step judge_xsum results/judge_xsum_vcyclem5c1_vs_baseline.json "$jdone" \
  experiments/judge.py --task xsum --a "vcycle|m=5|c=1" --b baseline --n 2000 --bs 16
step judge_wmt14 results/judge_wmt14_vcyclem5c1_vs_baseline.json "$jdone" \
  experiments/judge.py --task wmt14 --a "vcycle|m=5|c=1" --b baseline --bs 16

# 3. Tables and charts here too, so the laptop only copies results back and commits.
run analyze experiments/analyze.py
stamp "queue done"

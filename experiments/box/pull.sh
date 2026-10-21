#!/usr/bin/env bash
# GPU box -> laptop: bring results/ back (the box started from the laptop's copy, so it is a superset): tables, charts
# and SUMMARY.md included (analyze.py runs on the box). Then commit on the laptop.
#   experiments/box/pull.sh <host> <port>   (BOX_USER=ubuntu on AWS, root on Vast; key: KEY=~/.ssh/id_ed25519_vast)
set -euo pipefail
cd "$(dirname "$0")/../.."
HOST=$1 PORT=$2 KEY=${KEY:-$HOME/.ssh/id_ed25519_vast} BOX_USER=${BOX_USER:-root}
ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=accept-new "$BOX_USER@$HOST" "cd ~/ncm && tar czf - ./results" \
  | tar xzf - -C .
echo "pulled results/ from $BOX_USER@$HOST:~/ncm"

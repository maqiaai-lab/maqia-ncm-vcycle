#!/usr/bin/env bash
# Laptop -> GPU box: copy the code, the results and the small data files (whose pickles were checked with
# pickletools) over SSH. Nothing secret goes to the box: no .git, no tokens; checkpoints are downloaded there.
#   experiments/box/push.sh <host> <port>   (BOX_USER=ubuntu on AWS, root on Vast; key: KEY=~/.ssh/id_ed25519_vast)
set -euo pipefail
cd "$(dirname "$0")/../.."
HOST=$1 PORT=$2 KEY=${KEY:-$HOME/.ssh/id_ed25519_vast} BOX_USER=${BOX_USER:-root}
tar czf - --exclude=./.venv --exclude=./.git --exclude=./third_party --exclude=./artifacts \
    --exclude='./results/cache/lm1b_mauve_feats_*' --exclude='./results/cache/*_smoke.pt' --exclude='./results/*_smoke.json' . \
  | ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=accept-new "$BOX_USER@$HOST" "mkdir -p ~/ncm && tar xzf - -C ~/ncm"
echo "pushed to $BOX_USER@$HOST:~/ncm"

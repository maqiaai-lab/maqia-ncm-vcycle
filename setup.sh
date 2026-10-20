#!/usr/bin/env bash
# Fetch the NCM code (pinned), the Sudoku checkpoints and the Sudoku test boards.
set -euo pipefail
cd "$(dirname "$0")"
NCM_COMMIT=c70274273f489abc73832b2b98584a8483b6b203

if [ ! -d third_party/ncm ]; then
  git clone https://github.com/HazyResearch/ncm.git third_party/ncm
fi
git -C third_party/ncm checkout -q "$NCM_COMMIT"

pip install -r requirements.txt
hf download hazyresearch/ncm-checkpoints --include "sudoku/*" --local-dir artifacts
hf download hazyresearch/ncm-datasets --repo-type dataset --include "sudoku/*" --local-dir data

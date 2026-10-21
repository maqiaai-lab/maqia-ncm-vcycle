#!/usr/bin/env bash
# One-time setup on a rented GPU box (Linux, NVIDIA driver supporting CUDA >= 12.8; AWS Deep Learning AMI or a Vast
# PyTorch image), from the repo root:
#   bash experiments/box/setup.sh
# Code, results and the small data files arrive from the laptop with experiments/box/push.sh; this script adds the
# environment, the pinned NCM code and the released checkpoints. No credentials are needed or stored here.
set -euo pipefail
cd "$(dirname "$0")/../.."

SUDO=$([ "$(id -u)" = 0 ] && echo "" || echo sudo)
python3 -m venv .venv || { $SUDO apt-get update -qq && $SUDO apt-get install -y -qq python3-venv && python3 -m venv .venv; }
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cu128
pip install -q -r requirements.txt mauve-text rouge-score sacrebleu

NCM_COMMIT=c70274273f489abc73832b2b98584a8483b6b203
[ -d third_party/ncm ] || git clone -q https://github.com/HazyResearch/ncm.git third_party/ncm
git -C third_party/ncm checkout -q "$NCM_COMMIT"

# Checkpoints (explicit file names: --include with several patterns failed in an earlier run).
hf download hazyresearch/ncm-checkpoints lm1b/tokenizer.pt lm1b/small/generator.pt lm1b/small/config.yaml \
  lm1b/medium/generator.pt lm1b/medium/config.yaml xsum/generator.pt xsum/tokenizer.pt xsum/config.yaml \
  wmt14/config.yaml --local-dir artifacts
# The judge (15 GB) and the second scorer: their repositories hold only safetensors, so a plain download wastes
# nothing. GPT-2 large and small download on first use (their repositories also hold TF, Flax and ONNX weights).
hf download Qwen/Qwen2.5-7B-Instruct > /dev/null
hf download Qwen/Qwen2.5-1.5B > /dev/null

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"
echo "setup done"

#!/usr/bin/env bash
set -euo pipefail

# Execute from the project root on AutoDL.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/autodl-tmp/uv-cache}"
export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"

python -m pip install -U uv
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.12 .venv
fi
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
uv pip install -r requirements.txt
python -c 'import torch, vllm, openai, datasets; print("torch", torch.__version__, "vllm", vllm.__version__, "CUDA available", torch.cuda.is_available())'

echo 'Environment ready. Run: source .venv/bin/activate'

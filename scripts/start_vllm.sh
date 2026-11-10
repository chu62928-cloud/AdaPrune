#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/Qwen2.5-Math-7B-Instruct}"
if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "Missing model config: $MODEL_DIR/config.json" >&2
  exit 1
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Use low initial memory fraction on a 32GiB GPU, especially when other processes exist.
args=(
  serve "$MODEL_DIR"
  --served-model-name qwen-math-7b
  --host 127.0.0.1
  --port 8000
  --dtype float16
  --max-model-len "${MAX_MODEL_LEN:-4096}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.65}"
  --max-num-seqs "${MAX_NUM_SEQS:-4}"
)
if [[ "${ENFORCE_EAGER:-0}" == 1 ]]; then
  args+=(--enforce-eager)
fi
exec vllm "${args[@]}"

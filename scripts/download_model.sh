#!/usr/bin/env bash
set -euo pipefail
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/Qwen2.5-Math-7B-Instruct}"
mkdir -p "$MODEL_DIR" "$HF_HOME"
echo "HF_HOME=$HF_HOME"
echo "MODEL_DIR=$MODEL_DIR"
hf download Qwen/Qwen2.5-Math-7B-Instruct --local-dir "$MODEL_DIR"
echo "Model files downloaded; inspect with: du -sh '$MODEL_DIR'"

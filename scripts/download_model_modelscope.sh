#!/usr/bin/env bash
set -euo pipefail
# Alternative only when the Hugging Face download is unavailable.
# Check https://modelscope.cn for the model repository availability.
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/Qwen2.5-Math-7B-Instruct}"
python -m pip install modelscope
python - "$MODEL_DIR" <<'PY'
import sys
from modelscope import snapshot_download
model_dir = snapshot_download(model_id="Qwen/Qwen2.5-Math-7B-Instruct", local_dir=sys.argv[1])
print("Downloaded to:", model_dir)
PY

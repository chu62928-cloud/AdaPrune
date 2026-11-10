#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4
mkdir -p outputs
python -m dp.prepare_data --outdir data
python -m dp.baseline --input data/aime_dev.jsonl \
  --output outputs/smoke_n8.jsonl \
  --n 8 --limit 3 --concurrency 2 --max-tokens 1024 \
  --seed 2026 --logprobs --top-logprobs 5
python -m dp.evaluate --input outputs/smoke_n8.jsonl --ks 1,4,8 \
  --output outputs/smoke_metrics.json
python -m dp.staged_replay --input outputs/smoke_n8.jsonl \
  --stage1-keep 4 --stage2-keep 2 \
  --stage1-tokens 128 --stage2-tokens 256 \
  --window 32 --delta 0.08 --output outputs/smoke_replay.json

#!/usr/bin/env bash
set -euo pipefail
curl -fsS http://127.0.0.1:8000/v1/models | python -m json.tool
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-math-7b","messages":[{"role":"system","content":"Solve the math question and put the integer in \\boxed{}."},{"role":"user","content":"What is 19 + 23?"}],"max_tokens":128,"temperature":0.0}' \
  | python -m json.tool

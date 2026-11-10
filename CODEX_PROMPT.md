# 给 Codex 的执行指令（可复制）

你现在负责在我的 AutoDL 单卡环境中，完成 `dynamic_pruning_autodl` 项目的第一阶段复现和验收。已经附带现有项目文件，请**先阅读 README.md、dp 目录、scripts 目录和 tests 目录，不要从零覆盖现有代码**。

## 我的环境

- AutoDL：NVIDIA GeForce RTX 4080 SUPER，实际 31.48 GiB，SM 8.9。
- NVIDIA Driver 595.71.05；基础环境 Python 的 PyTorch 2.7.0+cu128；驱动报告 CUDA 13.2。
- `/root/autodl-tmp` 有 271GB 可用空间，整个磁盘占用 93%。
- 上次 `nvidia-smi` 有 6 个未知 Python 进程、GPU 使用率 98%、显存约 6.4GB 已被占用。**任何时候不要未经确认直接 kill 进程或删除原有文件。**
- 项目目录设为 `/root/autodl-tmp/dynamic_pruning_autodl`，模型目录为 `/root/autodl-tmp/models/Qwen2.5-Math-7B-Instruct`。

## 第一阶段任务（按顺序执行）

1. `nvidia-smi`、`df -h /root/autodl-tmp`、`python --version`；确认是否有其他 GPU 任务。如果有未知进程，报告 PID/使用情况，**不要自行结束**。
2. `export OMP_NUM_THREADS=4`；用 `bash scripts/setup_autodl.sh` 创建隔离 `.venv`。任何 pip/vLLM 版本冲突只在 `.venv` 内修复，不能破坏 AutoDL 基础 PyTorch 环境。
3. `source .venv/bin/activate && python -m pytest -q`；修复测试失败。
4. 使用 `bash scripts/download_model.sh` 下载官方 `Qwen/Qwen2.5-Math-7B-Instruct`；如 HF 网络不通，尝试验证对应 ModelScope 仓库并下载（允许请求用户确认网络/代理配置，不能自动改动系统代理）。确认 config、tokenizer、safetensors 文件完整。
5. `python -m dp.prepare_data --outdir data`；验证 dev=60、test=30，以及 AIME 答案为 0..999 整数。
6. 在 GPU 空闲且显存允许时执行 `bash scripts/start_vllm.sh`；使用 `bash scripts/check_server.sh` 做 API 测试。默认显存比例 0.65、max-model-len=4096、max-num-seqs=4，如果 OOM，首先检查已有进程、再保守调整配置。
7. 运行 `bash scripts/run_smoke.sh`（3 题 × 8 轨迹），确认 JSONL 完整、API 错误数量为零、logprobs 非空；产生 `outputs/smoke_metrics.json` 和 `outputs/smoke_replay.json`。
8. 上述通过后再跑 `outputs/dev_n8.jsonl`（开发集 60 题 × 8 轨迹）。不要贸然运行 60×32×2048 的大实验。先报告运行成本和显存余量，再决定扩展。
9. 所有生成结果都要保存参数、随机种子、模型标识及版本。记录 GPU 显存峰值、总 Token 和有效答案率。

## 对现有实现的检查重点

- 答案提取只允许有明确最终答案标志的 0..999 整数；不允许从推理正文盲取最后一个数字。
- `dp/entropy.py` 当前是 Top-K 条件熵近似，而不是完整词表熵，输出指标要明确标注近似。
- `dp/staged_replay.py` 只是**离线反事实回放**；所有轨迹在离线选择前已经生成，所以**不能宣称计算、KV Cache 或时间节省**。
- 所有 entropy selection 只能使用对应 checkpoint 的前缀，不允许把完整答案是否正确或后续 Token 熵用于早期筛选。
- Best-of-N 是同一 N 候选池的前 K 投票，不是独立测量的静态 Best-of-K 延迟；要对真实延迟对比，应分别执行。
- 动态剪枝阶段 2 的 Python Sandbox 与在线请求 abort 尚未实现，不要把离线 entropy-only 筛选称为海报完整复现。

## 每个阶段交付

- 修改的文件与理由。
- 实际运行命令，命令输出的关键行。
- `pytest` 结果。
- 如果执行了 GPU 推理，输出 1 个模型回答样例和一份 `outputs/` 汇总。
- 所有未完成项及原因（网络、GPU 占用、CUDA、兼容性）。
- 严禁虚构 GPU 验证、Benchmark 分数或显存优化结果。

## 第二阶段（只有第一阶段验收通过后才开始设计）

设计真正的在线增量 Rollout、vLLM 动态中止、轨迹状态管理与沙箱验证。**先提交设计说明、接口协议和测试计划**，不要直接部署不受信任的代码执行服务。

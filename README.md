# 动态剪枝 LLM 数学推理复现（AutoDL RTX 4080 SUPER 32GB）

> 本仓库第一阶段实现：数据准备、vLLM Best-of-N 采样、AIME 整数答案评测、Top-K 近似熵、**离线** 32→16→8（或 8→4→2）前缀筛选回放。
>
> **尚未实现**：在线提前终止请求、Python Sandbox、训练或 GRPO。离线回放所有轨迹已经完整生成，**绝对不能声称节省了 GPU 时间、KV Cache 或训练计算**。

## A. 当前机器信息（用户实测，2026-11-10）

- NVIDIA GeForce RTX 4080 SUPER；Compute Capability 8.9；显存 31.48 GiB。
- Driver 595.71.05；PyTorch 2.7.0+cu128；驱动所报告 CUDA 13.2（不要求二者数字相同）。
- `/root/autodl-tmp` 剩余 271 GB，足够一份 7B FP16 模型及实验文件，但整体磁盘占用已 93%。
- 当时 GPU 使用率 98%，有 6 个 Python 进程合计占用约 6.4 GB 显存。**先确认进程是否自己的，勿杀不属于自己的进程。**
- 如果出现 `libgomp: Invalid value for environment variable OMP_NUM_THREADS`，执行 `export OMP_NUM_THREADS=4`。

查看现有任务：

```bash
nvidia-smi
ps -o user,pid,etime,args -p 760405,760408,760409,760410,760411,760412
```

这些 PID 仅为你那次截图中的 PID，重启实例后可能变化。

## B. 推荐目录

AutoDL 工作目录为 `/root/autodl-tmp/AdaPrune`。恢复已有复现实验时，先查看 `PROJECT_MEMORY.md` 和服务器上的 `outputs/`。

```bash
cd /root/autodl-tmp/AdaPrune
pwd
ls dp scripts tests
```

模型权重存放 `/root/autodl-tmp/models/Qwen2.5-Math-7B-Instruct`，HF 缓存存放 `/root/autodl-tmp/hf-cache`。

## C. 环境安装（AutoDL Terminal）

**强烈建议创建独立虚拟环境。** 已有 torch 2.7.0+cu128 是基础环境中的版本；安装新版 vLLM 可能改变 PyTorch 版本，因此不应在基础环境直接覆盖。

```bash
cd /root/autodl-tmp/AdaPrune
export OMP_NUM_THREADS=4
mkdir -p outputs
python -m venv outputs/uv-bootstrap
source outputs/uv-bootstrap/bin/activate
bash scripts/setup_autodl.sh
source .venv/bin/activate
python -m pytest -q
```

引导环境用于安装 `uv`，推理依赖安装到单独的 Python 3.12 环境 `.venv`。安装中断后可以再次运行脚本。vLLM 会选择与其轮子兼容的 CUDA 和 PyTorch 组合。

如果 PyPI 下载缓慢，可在运行安装脚本前为当前会话设置 `PIP_INDEX_URL` 和 `UV_DEFAULT_INDEX`，例如 `https://mirrors.aliyun.com/pypi/simple/`。安装完成后记录版本：

```bash
python - <<'PY'
import torch, vllm
print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
print('vllm:', vllm.__version__)
print('gpu:', torch.cuda.get_device_name(0))
PY
uv pip freeze > outputs/installed_packages.txt
```

## D. 模型下载

第一选择 Hugging Face 官方仓库：<https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct>。

```bash
source .venv/bin/activate
bash scripts/download_model.sh
ls -lh /root/autodl-tmp/models/Qwen2.5-Math-7B-Instruct/*.safetensors
du -sh /root/autodl-tmp/models/Qwen2.5-Math-7B-Instruct
```

如果 Hugging Face 不通，Qwen 官方文档推荐中国大陆用户考虑 ModelScope；可以尝试：

```bash
(source outputs/uv-bootstrap/bin/activate && bash scripts/download_model_modelscope.sh)
```

替代下载仅在对应的 ModelScope 仓库可访问时有效。下载后必须检查 `config.json`、Tokenizer 文件和 `*.safetensors` 均存在，不应把下载失败的空目录当作模型。一次只选择一个下载渠道，避免重复占用磁盘空间。

## E. 数据集下载

公开数据集共 90 道 AIME 2022–2024 题目，原标识 `AI-MO/aimo-validation-aime` 重定向到 [project-numina/aimo-validation-aime](https://huggingface.co/datasets/project-numina/aimo-validation-aime)。

```bash
python -m dp.prepare_data --outdir data
wc -l data/*.jsonl
head -n 1 data/aime_dev.jsonl | cut -c 1-200
```

如果 AutoDL 直连 Hugging Face 超时，可以在当前会话导出 `HF_ENDPOINT=https://hf-mirror.com` 和 `HF_HUB_DISABLE_XET=1`。先验证数据文件的哈希和年份切分，再运行冒烟测试。

预期：`data/aime_dev.jsonl` 60 道（2022–2023），`data/aime_2024.jsonl` 30 道（2024）。

**实验纪律**：

- 开发集用于调参数，例如 `window`、`delta`、`max_tokens`。
- 2024 留作最后测试，不要根据其正确率反复调整阈值。
- 这个公开验证集和海报未指明版本的 “AIMO Benchmark” **不能视为同一测试集**；数值不可直接对照。
- 该数据集来源于公开竞赛题，预训练模型可能接触过相关题目；严格泛化评测仍需其他公开或私有未见题。

## F. 启动本机模型服务（终端 A）

确保显卡已空闲或预留足够显存。

```bash
cd /root/autodl-tmp/AdaPrune
source .venv/bin/activate
export OMP_NUM_THREADS=4
export GPU_MEMORY_UTILIZATION=0.65
export MAX_MODEL_LEN=4096
export MAX_NUM_SEQS=4
bash scripts/start_vllm.sh 2>&1 | tee outputs/vllm.log
```

服务绑定 `127.0.0.1:8000`，只在 AutoDL 实例内部访问；别把无鉴权服务公开到公网。

若启动时显存不足：先检查其他占用，再将 `GPU_MEMORY_UTILIZATION=0.60`、`MAX_NUM_SEQS=2` 或 `MAX_MODEL_LEN=3072` 重试；也可尝试 `ENFORCE_EAGER=1`（通常牺牲一定吞吐、减少 CUDA Graph 相关开销）。如果 7B FP16 权重本身已不适应剩余显存，先换 1.5B 模型用于功能测试，**不要假设调整一个参数就能解决所有 OOM**。

## G. 检查服务器（终端 B）

```bash
cd /root/autodl-tmp/AdaPrune
source .venv/bin/activate
bash scripts/check_server.sh
```

**验收标准**：`/v1/models` 返回 `qwen-math-7b`，简单的 `19+23` 数学题能完成推理，并且返回 `choices`。

如果此步骤失败，不要往下做 32 路实验。

## H. 跑 Best-of-N 基线

### H1. 最小端到端 smoke test（3 道题、8 条轨迹）

```bash
bash scripts/run_smoke.sh
```

该脚本一次性完成数据下载、生成、准确率评测和离线回放。它只应在输出文件尚未存在时直接运行。重新执行相同生成配置，请向 `dp.baseline` 手动加 `--resume`，或者改一个 `--output`；不要直接覆盖已保存的实验结果。

对应的手动命令：

```bash
python -m dp.baseline \
  --input data/aime_dev.jsonl \
  --output outputs/smoke_n8.jsonl \
  --n 8 --limit 3 --concurrency 2 --max-tokens 1024 \
  --seed 2026 --logprobs --top-logprobs 5

python -m dp.evaluate --input outputs/smoke_n8.jsonl \
  --ks 1,4,8 --output outputs/smoke_metrics.json

python -m dp.staged_replay --input outputs/smoke_n8.jsonl \
  --stage1-keep 4 --stage2-keep 2 \
  --stage1-tokens 128 --stage2-tokens 256 \
  --window 32 --delta 0.08 \
  --output outputs/smoke_replay.json
```

**验收标准**：

- `outputs/smoke_n8.jsonl` 有 3 行，每行有 8 条 `rollouts`；`text`、`answer`、`completion_tokens`、`finish_reason` 可读。
- `outputs/smoke_metrics.json` 有 Best-of-1、4、8 的 `accuracy` 和 `generated_tokens`。
- `outputs/smoke_replay.json` 有阶段筛选 ID 和准确率；不需要先追求高准确率。
- 若 `--logprobs` 不被当前 vLLM 支持，先不带它完成基础 Best-of-N；**熵分析必须等 logprobs 正常再运行**，不要假造熵。

### H2. 扩展到开发集（60 道题、每题 8 路）

```bash
python -m dp.baseline \
  --input data/aime_dev.jsonl \
  --output outputs/dev_n8.jsonl \
  --n 8 --limit 0 --concurrency 2 \
  --temperature 0.7 --top-p 0.95 --max-tokens 2048 \
  --seed 2026 --logprobs --top-logprobs 5

python -m dp.evaluate --input outputs/dev_n8.jsonl \
  --ks 1,4,8 --output outputs/dev_n8_metrics.json
```

这里 `--limit 0` 表示处理所有输入问题。若中途中断，再运行原命令并添加 `--resume`，已完成的题目会被跳过。确保 `--n`、种子、模型、采样和日志配置保持一致。

### H3. 32 路采样与 32→16→8 离线筛选

确认 H2 正常、内存和时间预算可接受后再执行：

```bash
python -m dp.baseline \
  --input data/aime_dev.jsonl \
  --output outputs/dev_n32.jsonl \
  --n 32 --limit 0 --concurrency 2 \
  --temperature 0.7 --top-p 0.95 --max-tokens 2048 \
  --seed 2026 --logprobs --top-logprobs 5

python -m dp.evaluate --input outputs/dev_n32.jsonl \
  --ks 1,4,8,16,32 --output outputs/dev_n32_metrics.json

python -m dp.staged_replay --input outputs/dev_n32.jsonl \
  --stage1-keep 16 --stage2-keep 8 \
  --stage1-tokens 256 --stage2-tokens 512 \
  --window 32 --delta 0.08 \
  --output outputs/dev_n32_replay.json
```

**明确限制**：这是两阶段 `entropy-only` 离线筛选；没有 Stage 3 Python Sandbox。第二阶段要替换成在线动态请求中止 + 沙箱验证后，才有资格评估计算效率。

## I. 已有代码解读

| 文件 | 已实现的功能 |
|---|---|
| `dp/prepare_data.py` | Hugging Face 数据集下载、年份切分、答案标准化 |
| `dp/baseline.py` | AsyncOpenAI 并发请求、Best-of-N、seed、logprobs、JSONL 断点续跑 |
| `dp/answers.py` | 提取 `\\boxed{}` 中的 AIME 整数答案、确定性多数投票 |
| `dp/entropy.py` | Top-K 条件熵、分窗均值、连续上升触发检测 |
| `dp/evaluate.py` | 同一 N 候选池的前缀 1/4/8/16/32 路准确率统计 |
| `dp/analyze.py` | 固定前缀熵的单次剪枝离线分析 |
| `dp/staged_replay.py` | 固定前缀的两阶段模拟筛选、随机保留对照、熵加权投票 |
| `tests/test_core.py` | 答案解析、熵、投票、分割和离线选择的基础单测 |

### 熵的科学性限制

`dp/entropy.py` 计算的是返回的 Top-K 候选重新归一化后的**条件熵（nats）**，并非完整词表熵。未知尾部概率被排除，不能把它与原海报中的完整 `H_t` 直接划等号。

`delta=0.08` 和 `window=32` 是可运行的**初始超参数**，并非已从作者处确认的官方值。只有经过开发集阈值搜索和误剪分析后，才可能对其有效性作结论。

### 基线的科学性限制

从同一份 `Best-of-32` 候选中抽取前 1/4/8/16 条来计算准确率，便于节约实验开销和保证候选一致，但**不是分别运行 Best-of-1/4/8/16 的实际延迟测量**。若对比显存、真实壁钟时间，必须单独运行各组、保持推理参数一致并多次重复。

## J. 记录资源

新开终端，边跑边观察：

```bash
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw \
  --format=csv -l 5 | tee outputs/gpu_trace.csv
```

程序完成后记录 vLLM 日志、环境版本、总 Token、准确率。实际 Token 的总和能通过 `evaluate.py` 计算；**没有在线剪枝时不应声称 GPU savings**。

## K. 后续阶段（待 Codex 开发）

1. 实现基于 vLLM AsyncLLM 或可中止流式请求的**在线**动态终止，不在每个 checkpoint 重算已经生成的前缀。
2. 在 checkpoint 分段推理时正确维护 KV Cache 和继续生成上下文，避免拼接造成 Token 不一致或额外 Prefill。
3. 实现状态隔离、无网络、限制 CPU / 内存 / 时间的 Python Sandbox，独立会话之间不能共享状态。
4. 实现 32→16（Temporal Entropy）→8（Sandbox Verification）→加权投票，明确错误轨迹选择和固定保留预算如何交互。
5. 对比最佳静态 N、Random Pruning、Entropy-only、Sandbox-only 和完整动态剪枝；报告正确率、总生成 Token、端到端延迟、吞吐、KV Cache 峰值。
6. 再考虑与 GRPO / PPO 训练交互（目前仓库完全不涉及模型训练）。

## L. 官方参考

- Qwen2.5-Math-7B-Instruct：<https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct>
- AIME 数据：<https://huggingface.co/datasets/AI-MO/aimo-validation-aime>
- vLLM 安装：<https://docs.vllm.ai/en/latest/getting_started/installation/gpu/>
- vLLM Serve 参数：<https://docs.vllm.ai/en/stable/cli/serve/>
- vLLM Chat API：<https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/>
- Hugging Face 模型下载：<https://huggingface.co/docs/huggingface_hub/main/guides/download>

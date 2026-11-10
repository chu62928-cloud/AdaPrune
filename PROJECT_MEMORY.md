# AdaPrune 项目状态

## 最后核验

- 2026-11-10 14:34（北京时间），AutoDL `motion-data` 和本地工作区。

## 目标

- 在 AutoDL 复现第一阶段，验收隔离环境、Qwen 数学模型、AIME 切分、vLLM 服务、Best-of-8 与离线剪枝回放。

## 当前状态

- 已核验。服务器工作目录 `/root/autodl-tmp/AdaPrune`，远端 `origin` 指向 [AdaPrune](https://github.com/chu62928-cloud/AdaPrune)。原始项目已推送到 `main`（`e351a89`）。
- 已核验。RTX 4080 SUPER，计算能力 8.9，显存 32,760 MiB，驱动 595.71.05，基础 PyTorch 2.7.0+cu128。初始检查时 GPU 空闲。
- 进行中。隔离环境安装和四个权重分片下载。服务和冒烟测试等待先决条件。

## 验证与证据

- 本机及服务器独立预检均执行 `python -m pytest -q`，各通过 6 项。服务器日志 `outputs/pytest_isolated_preflight_retry.log`。
- `python -m dp.prepare_data --outdir data` 生成开发集 60 题、2024 年测试集 30 题。90 个 ID 唯一，答案全部是 0 至 999 的规范整数。来源版本 `13f9e12f613e720c2a2b2f345dd04b998a29494d`。日志 `outputs/prepare_data_retry.log` 和 `outputs/aime_validation.log`。
- 服务器直连 Hugging Face 超时。模型使用 ModelScope 的 Qwen 官方仓库，四个权重分片的清单哈希与 Hugging Face 版本 `ef9926d75ab1d54532f6a30dd5e760355eb9aa4d` 一致。下载后的本地哈希待核验。

## 实验约束

- 保留服务器已有 GPU 任务及基础 PyTorch。安装和重试使用项目专属环境。
- 2024 年测试集留作最终评估。开发集冒烟测试使用前三题、每题八条轨迹、种子 2026、每条上限 1024 个词元。
- 回放根据检查点前缀的 Top-K 条件熵筛选已生成轨迹。在线中止、沙箱和资源节省属于后续工作。完整 Best-of-32 实验暂缓。

## 关键路径

- `.venv` 为推理环境，`outputs/uv-bootstrap` 为依赖引导和模型下载环境。
- 模型 `/root/autodl-tmp/models/Qwen2.5-Math-7B-Instruct`。日志、原始生成、依赖版本和显存数据保存在服务器项目的 `outputs/`。

## 后台任务

- `adaprune-setup` 写入 `outputs/setup_retry2.log`，完成时生成 `outputs/setup.done`。
- `adaprune-download` 写入 `outputs/download_parallel.log`，完成时生成 `outputs/model_download.done`。
- `adaprune-smoke` 写入 `outputs/serve_smoke.log`，等待环境、模型和数据就绪后校验模型，检查 GPU 与端口，再启动服务和冒烟测试。

## 续跑

1. 执行 `ssh motion-data 'cd /root/autodl-tmp/AdaPrune; tmux ls; tail -20 outputs/setup_retry2.log; tail -12 outputs/serve_smoke.log'`。
2. 检查模型校验、服务探测和冒烟结果。用实测词元吞吐和剩余显存估算开发集运行时间，再决定是否扩展。

## 待解决

- 待核验推理环境实际版本、模型权重、服务响应和冒烟结果。

## 更新记录

- 2026-11-10。导入原项目，建立远端，完成数据切分与独立预检。

# DataForge

**面向 LLM 训练数据合成与筛选的高并发异步流水线。**

[![CI](https://github.com/kero-ly/dataforge/actions/workflows/ci.yml/badge.svg)](https://github.com/kero-ly/dataforge/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/dataforge)](https://pypi.org/project/dataforge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

[English README](README.md)

---

## 目录

- [DataForge 是什么？](#dataforge-是什么)
- [效果示例](#效果示例)
- [安装](#安装)
- [快速开始](#快速开始)
- [工作原理](#工作原理)
- [CLI 命令](#cli-命令)
- [支持的后端](#支持的后端)
- [配置参数说明](#配置参数说明)
- [核心功能](#核心功能)
- [性能基准](#性能基准)
- [文档](#文档)
- [参与贡献](#参与贡献)
- [引用](#引用)
- [许可证](#许可证)

---

## DataForge 是什么？

大规模构建 SFT/RLHF 数据集，首先是一个基础设施问题，其次才是数据问题。朴素的异步脚本会悄悄触达速率限制、在没有检查点的情况下崩溃、生成格式错误的输出，并且没有任何审计记录。当记录数超过 1 万条时，这些问题会相互叠加——在进度 80% 时崩溃，意味着损失数小时的 API 费用，并且需要从头开始。

DataForge 是一个生产级异步流水线，自动处理速率限制、重试、检查点和质量过滤——让你专注于生成*什么*数据，而不是*如何*生成。无论你使用云端 API 还是本地 vLLM 服务器，它的工作方式完全一致。

**适用人群：** 运行 WizardLM 风格指令进化的研究人员和工程师、从种子语料库构建 SFT 数据集的团队、需要在大规模（1K–100K+ 条记录）下过滤和评分合成数据的场景，或者需要跨模型提供商比较数据质量的用户。

---

## 效果示例

以下是 DataForge 使用 `evol-instruct`（深度为 2）对种子指令的处理结果：

**输入**（`seeds.jsonl`）：
```json
{"id": "q1", "instruction": "什么是机器学习？"}
```

**输出**（`output.jsonl`），经过两轮 WizardLM 风格变异后：
```json
{
  "id": "q1",
  "seed_data": {"instruction": "什么是机器学习？"},
  "synthetic_data": {
    "instruction": "你正在为一家有严格延迟要求（p99 < 50ms）的金融科技公司设计生产级机器学习系统。请解释机器学习的工作原理，涵盖：(1) 特征与预测之间的数学关系，(2) 模型如何泛化到训练数据之外，(3) 在低延迟环境中部署时出现的两个具体故障模式。"
  },
  "score": 4.5,
  "status": "COMPLETED"
}
```

EvolInstruct 应用 WizardLM 风格的变异——添加约束、深化复杂度、具体化上下文——使指令更难、更丰富。`depth` 字段控制每条记录应用的变异轮数。

---

## 安装

```bash
pip install dataforge
```

**要求：** Python 3.10+，云端 API 模式无需 GPU。

本地 LLM 推理（vLLM / Ollama）：
```bash
pip install vllm   # 需要支持 CUDA 的 GPU
```

从源码安装：
```bash
git clone https://github.com/kero-ly/dataforge.git
cd dataforge
pip install -e ".[dev]"
```

---

## 快速开始

### 立即试用（无需 API Key）

最快体验 DataForge 的方式——无需任何凭证。

**第一步**，创建 `seeds.jsonl`：
```json
{"id": "q1", "instruction": "什么是机器学习？"}
{"id": "q2", "instruction": "解释 Transformer 中的注意力机制。"}
{"id": "q3", "instruction": "什么是过拟合，如何防止？"}
```

**第二步**，创建 `config.yaml`：
```yaml
name: quickstart
source:
  type: jsonl
  path: ./seeds.jsonl
pipeline:
  - step: generate
    strategy: evol-instruct
    depth: 2
sink:
  path: ./output.jsonl
concurrency: 4
```

**第三步**，运行：
```bash
python -m dataforge run config.yaml --backend fake
```

预期输出：
```
Pipeline completed: 3/3 records
  Completed: 3  Rejected: 0  Failed: 0
  Elapsed: 0.1s  Throughput: 30.0 rec/s
```

结果写入 `output.jsonl`，每条记录包含 `seed_data`、`synthetic_data`、`score` 和 `status`。

---

### 使用云端 API（OpenAI / DeepSeek）

**第一步**，使用上面的 `seeds.jsonl`。

**第二步**，创建 `config.yaml`：
```yaml
name: cloud-pipeline
source:
  type: jsonl
  path: ./seeds.jsonl
pipeline:
  - step: generate
    strategy: evol-instruct
    depth: 3
    mutation_types: [constraints, deepen, concretize]
    llm:
      provider: openai
      model: gpt-4o-mini
      api_key: ${OPENAI_API_KEY}   # 从环境变量读取
      rpm_limit: 60
      tpm_limit: 100000
      generation_kwargs:
        temperature: 0.7
        max_tokens: 1024
  - step: evaluate
    evaluator: llm-judge
    criteria: helpfulness
    threshold: 4.0
    llm:
      provider: openai
      model: gpt-4o-mini
      api_key: ${OPENAI_API_KEY}
sink:
  path: ./output.jsonl
  checkpoint_dir: ./.dataforge_runs   # 崩溃安全：用同一命令重启即可续跑
  dead_letter_path: ./failed.jsonl    # 被拒绝的记录保存在此
concurrency: 20
```

**第三步**，运行：
```bash
export OPENAI_API_KEY=sk-...
dataforge run config.yaml
```

> **DeepSeek：** 添加 `base_url: https://api.deepseek.com/v1` 并使用 `api_key: ${DEEPSEEK_API_KEY}`，其他配置不变。

如果运行中断，用完全相同的命令重启——DataForge 会读取检查点并自动跳过已完成的记录。

---

### 使用本地 vLLM 服务器

**第一步**，使用上面的 `seeds.jsonl`。

**第二步**，创建 `config.yaml`：
```yaml
name: local-pipeline
source:
  type: jsonl
  path: ./seeds.jsonl
pipeline:
  - step: generate
    strategy: evol-instruct
    depth: 3
    mutation_types: [constraints, deepen, concretize]
    llm:
      provider: vllm
      model: Qwen/Qwen2.5-7B-Instruct
      base_url: http://localhost:8000/v1
      concurrency: 50
  - step: evaluate
    evaluator: regex-filter
    blacklist_patterns: ["I cannot", "I'm sorry", "As an AI"]
sink:
  path: ./output.jsonl
  checkpoint_dir: ./.dataforge_runs
concurrency: 50
```

**第三步**，启动 vLLM 并运行：
```bash
# 终端 1
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# 终端 2
dataforge run config.yaml
```

---

## 工作原理

```
seeds.jsonl → [生产者] → asyncio.Queue → [Worker ×N] → output.jsonl
                                               │
                                      strategy.apply()    ← 进化指令
                                               │
                                     [evaluator 链]       ← 过滤/评分
                                               │
                                  checkpoint.commit()     ← WAL 追加写入
```

`N` 个 Worker 从有界队列中拉取记录。每个 Worker 运行策略（EvolInstruct 对指令进行 `depth` 轮变异），然后将结果传递给评估器链（RegexFilter 拒绝黑名单模式；LLMJudge 评分 1–5 分，低于阈值则拒绝）。只有通过所有评估器的记录才会写入输出 JSONL。每次写入后，记录 ID 被追加到预写日志——用同一命令重启，零记录丢失。

双桶限速器同时对每个 LLM 客户端强制执行 RPM 和 TPM 限制，防止朴素异步代码导致的静默配额违规。

---

## CLI 命令

| 命令 | 说明 |
|------|------|
| `dataforge run config.yaml` | 运行合成流水线 |
| `dataforge run config.yaml --dry-run` | 验证配置但不运行 |
| `dataforge assess output.jsonl` | 生成数据质量报告（JSON / HTML） |
| `dataforge benchmark config.yaml` | 运行 MT-Bench / IF-Eval 精简评测 |
| `dataforge status .dataforge_runs` | 查看检查点进度 |
| `dataforge inspect output.jsonl` | 打印输出统计信息 |

---

## 支持的后端

| 提供商 | `provider` 值 | 说明 |
|--------|--------------|------|
| OpenAI | `openai` | GPT-4o、GPT-4o-mini、o1 等 |
| Anthropic | `anthropic` | Claude 3.x，通过 OpenAI 兼容接口 |
| DeepSeek | `openai` | 设置 `base_url: https://api.deepseek.com/v1` |
| vLLM | `vllm` | 本地部署（Qwen、Llama、Mistral、Phi…） |
| Ollama | `vllm` | 设置 `base_url: http://localhost:11434/v1` |
| 阿里百炼 | `bailian` | DashScope API |

只需修改 `config.yaml` 中的两行即可切换提供商——无需改动代码。

---

## 配置参数说明

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `source.path` | — | 输入路径（JSONL / CSV / Parquet） |
| `sink.path` | — | 输出 JSONL 路径 |
| `sink.checkpoint_dir` | `.dataforge_runs` | WAL 检查点目录 |
| `sink.dead_letter_path` | `null` | 被拒绝/失败记录的保存路径 |
| `concurrency` | `50` | 最大并发 LLM 请求数 |
| `mode` | `streaming` | `streaming`（大数据集）或 `burst`（内存可容纳） |
| `pipeline[].step` | — | `generate` 或 `evaluate` |
| `pipeline[].strategy` | `evol-instruct` | `evol-instruct` · `paraphrase` · `seed-to-qa` · `self-play` |
| `pipeline[].depth` | `3` | EvolInstruct 变异轮数 |
| `pipeline[].mutation_types` | 全部 | `constraints` · `deepen` · `concretize` |
| `pipeline[].evaluator` | — | `regex-filter` · `llm-judge` · `length-filter` |
| `llm.provider` | — | 参见[支持的后端](#支持的后端) |
| `llm.rpm_limit` | `60` | 每分钟请求数限制 |
| `llm.tpm_limit` | `100000` | 每分钟 token 数限制 |
| `llm.api_key` | — | 支持 `${ENV_VAR}` 语法 |
| `llm.generation_kwargs` | `{}` | 传递给 LLM：`temperature`、`max_tokens` 等 |

---

## 核心功能

| 功能 | 说明 |
|------|------|
| 异步优先 | 单有界队列 + N-worker 模式，全程非阻塞 I/O |
| 双维度限速 | RPM + TPM 令牌桶，持续补充 |
| WAL 检查点 | 崩溃安全；支持 SQLite 和 Redis 后端 |
| EvolInstruct | WizardLM 风格变异：约束 / 深化 / 具体化 |
| LLMJudge | 1–5 分评估器，正则优先 + 数字提取兜底 |
| RegexFilter | 黑名单模式匹配 + 可选 JSON Schema 验证 |
| 分布式 | 支持 Ray Actor、Dask、分片三种后端 |
| 统一路由 | 一份配置同时适配 OpenAI、Anthropic、DeepSeek、vLLM、Ollama |
| 数据质量 CLI | `dataforge assess` 生成结构化质量报告 |

---

## 性能基准

所有实验默认使用 Qwen2.5-7B-Instruct，统计检验采用双侧 t 检验。

### 吞吐量

| 基准 | DataForge | 提升 |
|------|-----------|------|
| 串行（1 worker） | 2,419 条/分 | **48×** |
| Naive Async | 2,419 vs 2,330 条/分 | **+3.82%**，p=0.013，95% CI=[44.3, 133.6] |
| Distilabel | — | **+152%** 吞吐量 |

### 容错能力（30% 随机故障注入）

| 方法 | 完成率 |
|------|--------|
| DataForge | **99.9%** |
| Naive Async（无重试） | 70.4% |
| **优势** | **+29 pp** |

3 次独立实验一致（7B 和 14B 模型均验证）。

### 云 API 限速合规（qwen-plus，RPM=120）

| 方法 | 完成率 |
|------|--------|
| DataForge 双桶限速 | **89.2%**（446/500） |
| 无限速器 | 40.0%（200/500） |
| Naive Async | 0.0%（0/500，vLLM 崩溃） |

### 分布式扩展（RayActor，4× workers）

| Worker 数 | 吞吐量 | 效率 |
|-----------|--------|------|
| 1× | 1,482 rpm | — |
| 2× | 2,798 rpm | 94.4% |
| 4× | **5,361 rpm** | **90.5%** |

---

## 文档

- [架构概览](docs/plans/structure.md)
- [配置参考](docs/plans/api_interface.md)
- [贡献指南](CONTRIBUTING.md)
- [更新日志](CHANGELOG.md)

---

## 参与贡献

欢迎贡献！请查阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解环境搭建、编码规范和 PR 流程。

---

## 引用

如果你在研究中使用了 DataForge，请引用：

```bibtex
@software{dataforge2026,
  title  = {DataForge: High-Concurrency Async Pipeline for LLM Training Data Synthesis},
  year   = {2026},
  url    = {https://github.com/kero-ly/dataforge}
}
```

---

## 许可证

MIT © [DataForge Contributors](https://github.com/kero-ly/dataforge/graphs/contributors)

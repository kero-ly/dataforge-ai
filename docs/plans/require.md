# 📄 DataForge 核心需求设计文档 (PRD & SRS)

**项目代号:** DataForge
**文档版本:** v1.0
**文档状态:** MVP 规划中
**核心定位:** 专为大模型时代打造的高并发、端到端“数据合成与提纯”中间件。

---

## 一、 产品概述 (Introduction)

### 1.1 产品愿景

DataForge 致力于成为 AI 数据处理领域的“炼油厂”。通过将前沿的生成算法（如指令演进、多智能体博弈）与极度稳健的工程架构（高并发调度、自动容错）结合，为 AI 开发者和企业提供低门槛、高吞吐的高质量训练数据流管线。

### 1.2 目标用户与核心痛点

* **AI 算法工程师 / 研究员：** 缺乏高质量的领域微调数据，手动编写并发脚本耗时且极易因 API 限制而崩溃。
* **企业 AI 业务线负责人：** 内部私有数据无法出域，亟需使用本地化部署的开源模型（如基于 vLLM）自动化构建 RAG 测试集或微调数据集，且要求数据来源可追溯、质量可量化。

---

## 二、 核心功能需求 (Functional Requirements)

DataForge 的核心功能模块划分为五大子系统：输入输出（I/O）、模型路由（Routing）、合成策略（Synthesis）、评估提纯（Evaluation）与 容错调度（Scheduling）。

### 2.1 数据输入与输出模块 (I/O Subsystem)

* **FR-101 多格式支持：** 支持 `.jsonl`, `.csv`, `.parquet` 格式的数据流式读取与写入。
* **FR-102 极简数据映射：** 允许用户通过简单的字段映射配置（Mapping），将原始数据结构转化为系统内部的标准化 Payload。
* **FR-103 元数据（Metadata）追加：** 输出的每一条数据必须强制附带元数据，包括：生成时间戳、耗时、模型版本、消耗 Token 数、以及所经过的 Pipeline 节点记录（实现数据血缘追溯）。

### 2.2 统一模型路由模块 (Model Routing)

* **FR-201 异构模型接入：** 提供统一接口兼容 OpenAI, Anthropic, Gemini 等云端 API，同时原生支持 vLLM, Ollama, SGLang 等本地推理引擎。
* **FR-202 动态故障转移 (Failover)：** 允许为每个节点配置备用模型（Fallback Model）。当主模型遭遇 `5xx` 错误或连续超时，自动无缝切换至备用模型，保证任务不中断。

### 2.3 数据合成策略模块 (Synthesis Strategies)

* **FR-301 内置 Evol-Instruct (指令演进)：** 系统预置深度演进（加约束、深化推理）与广度演进（话题突变）算子，支持对单一种子数据进行指定深度的 $N$ 轮裂变。
* **FR-302 思维链注入 (CoT Injection)：** 强制生成带有思考过程的数据，原生支持提取 `<think>...</think>` 等特殊标签结构。
* **FR-303 结构化输出约束 (Structured Output)：** 结合 JSON Schema，强制大模型输出合规的数据结构，内置针对非标准 JSON 输出的正则表达式修复器。

### 2.4 数据评估与提纯模块 (Evaluation & Curation)

* **FR-401 LLM 裁判 (LLM-as-a-Judge)：** 支持配置独立的评审模型，基于自定义的维度（如“事实准确性”、“逻辑严密性”）对生成数据进行 1-5 分量化打分。
* **FR-402 阈值过滤 (Threshold Filtering)：** 允许设置硬性阈值（例如：得分 $< 4.0$ 的数据自动进入废弃池或死信队列）。
* **FR-403 规则引擎 (Rule-based Filter)：** 支持基于正则表达式、敏感词库、文本长度等硬性规则的极速预过滤。

### 2.5 并发调度与容错模块 (Scheduling & Resiliency)

* **FR-501 智能令牌桶限流 (Token-Bucket Rate Limiting)：** 根据各大 API 的 RPM（每分钟请求）和 TPM（每分钟 Token）限制，动态调节并发协程数。
* **FR-502 指数退避重试 (Exponential Backoff)：** 捕获 `429` 错误并执行带有随机抖动（Jitter）的延迟重试。
* **FR-503 检查点与断点续传 (Checkpointing)：** 采用预写式日志（WAL）设计，每完成一个 Chunk 立即落盘。重启任务时自动比对已完成的 ID，跳过重复工作。

---

## 三、 非功能性需求 (Non-Functional Requirements)

### 3.1 性能与吞吐量 (Performance)

* 单机环境下，调度器本身的开销极低。支持同时挂起 10,000+ 个异步协程而不发生内存溢出（OOM）。
* IO 读写完全异步化，不得阻塞主事件循环（Event Loop）。

### 3.2 可靠性 (Reliability)

* 面对断网、进程被强杀（如 `SIGKILL`）等极端情况，重启后数据丢失率必须为 $0$。最多仅丢失当前正在内存中请求的那几个批次，已返回的数据必须 100% 持久化。

### 3.3 可扩展性 (Extensibility)

* **插件化设计：** 所有的合成策略（Strategy）和评估器（Evaluator）必须继承自基础基类，开发者可以通过几行 Python 代码轻松注册自定义的逻辑组件。

---

## 四、 开发者体验 (DX & Usability)

为了降低使用门槛，DataForge 提供两种交互范式：

### 4.1 极简的 Python SDK

面向高阶开发者，提供如搭积木般的链式调用体验（上文已展示过）。

### 4.2 声明式的 YAML 配置 (Data as Code)

面向不熟悉异步编程的算法研究员或数据产品经理，提供通过单一 `.yaml` 文件定义整个工作流的能力。

```yaml
# experiment_v1.yaml
name: "medical-reasoning-dataset"
source:
  type: "jsonl"
  path: "./data/medical_seeds.jsonl"

pipeline:
  - step: "generate"
    strategy: "evol-instruct"
    depth: 2
    llm:
      provider: "vllm"
      model: "Qwen/Qwen2.5-7B-Instruct"
      base_url: "http://localhost:8000"
      concurrency: 50
    
  - step: "evaluate"
    evaluator: "llm-judge"
    criteria: "medical-accuracy"
    threshold: 4.0
    llm:
      provider: "openai"
      model: "gpt-4o-mini"

sink:
  path: "./data/output/high_quality_medical.jsonl"
  checkpoint_dir: "./checkpoints/exp_v1"

```

用户只需在终端运行：`dataforge run experiment_v1.yaml` 即可启动。

---

## 五、 版本迭代计划 (Roadmap)

| 版本阶段 | 核心目标 | 交付功能 |
| --- | --- | --- |
| **v0.1 (Alpha)** | 跑通底层并发基建 | 异步调度器、统一模型路由接口、基础检查点功能、极简 Python SDK。 |
| **v0.5 (Beta)** | 引入前沿合成算法 | Evol-Instruct 模块、支持 JSON 强制校验与自修复逻辑、命令行 UI 进度条。 |
| **v1.0 (MVP)** | 闭环与开源发布 | 引入 LLM 裁判评估器、YAML 配置文件解析、完善的文档库。正式对外开源。 |
| **v1.5 (Pro)** | 高阶数据提纯 | 引入基于向量的语义去重模块、支持复杂的多智能体（Self-Play）对抗生成模板。 |

---

基于我们在前面确定的商业战略、进阶业务场景以及详细的需求规格（SRS），这份**《DataForge 系统架构设计文档 (System Architecture Document, SAD)》**将从更高的工程视角，把“概念”转化为研发团队可以直接落地的“蓝图”。

这份文档采用了类似 C4 模型（Context, Containers, Components, Code）的架构表达方式，确保系统的模块化、高可用和可扩展性。

---

# 🏗️ DataForge 系统架构设计文档 (v1.0 MVP)

## 一、 架构设计原则 (Design Principles)

1. **面向异步与非阻塞 (Async-First)：** 全局采用 Python `asyncio` 架构。所有 I/O 操作（API 请求、磁盘读写）均不阻塞主线程，实现单机万级协程的极致并发。
2. **职责分离与依赖注入 (IoC & Decoupling)：** 管道调度逻辑与底层 LLM 请求完全解耦。LLM Client 仅作为一个提供 `generate()` 方法的接口被注入到策略层，方便未来无限横向扩展模型厂商。
3. **“无常”假设下的高可用 (Design for Failure)：** 假设大模型 API 会随时限流、返回错误格式、甚至宕机。系统必须具备极强的韧性（Resiliency），通过预写式日志（WAL）保证数据零丢失。
4. **管道化与插件化 (Pipeline & Plugin)：** 采用有向无环图（DAG）的流式处理思想，每一个合成和清洗步骤（Step）都是独立的插件，用户可自由插拔。

---

## 二、 系统上下文视图 (System Context View)

定义 DataForge 系统与其周围生态环境的交互边界：

* **内部系统 (DataForge Core)：** 承担核心的数据读取、调度、策略生成、评估打分和持久化。
* **上游外部系统 (外部依赖)：**
* **云端 LLM 供应商：** OpenAI, Anthropic, DeepSeek API（提供强大但不稳定的在线推理算力）。
* **本地/私有推理集群：** 基于 vLLM, SGLang, Ollama 部署的本地大模型（提供高吞吐、低延迟、高隐私的算力）。


* **下游外部系统 (输出目标)：**
* **存储层：** 本地文件系统 (JSONL/Parquet)，对象存储 (S3/OSS)。
* **生态集成：** Hugging Face Hub（一键发布数据集），Argilla/Label Studio（导出边界数据进行人工微调）。



---

## 三、 核心容器与组件架构 (Container & Component Architecture)

系统自上而下分为四层：**接口层、管道调度层、执行引擎层、核心组件层**。

### 3.1 接口层 (Interface Layer)

* **SDK CLI & Parser：** 负责解析用户的 YAML 配置文件，将其转化为系统内部可执行的 DAG（有向无环图）配置对象。
* **Pipeline API：** 供高级 Python 开发者直接实例化的链式调用接口。

### 3.2 管道调度层 (Pipeline Orchestrator Layer)

这是整个数据流转的“交通指挥中心”。

* **DAG Manager：** 管理任务的依赖关系（例如：必须先经过 `Evol-Instruct` 节点，才能进入 `LLM-Judge` 节点）。
* **State & Checkpoint Manager：** 负责断点续传。每次启动时，读取 `checkpoint.jsonl`，比对输入源的 Hash/ID，将已完成的任务剔除，仅将未完成的 `TaskPayload` 放入队列。

### 3.3 执行引擎层 (Execution Engine Layer) 🌟

这是 DataForge 的核心护城河。

* **Async Task Queue (`asyncio.Queue`)：** 承载待处理任务的高性能内存队列。
* **Unified Rate Limiter (统一限流器)：** 维持双重令牌桶。精确计算并控制 TPM（Tokens Per Minute）和 RPM（Requests Per Minute）。
* **Resiliency Engine (韧性引擎)：** * 负责捕获异常（如 `HTTP 429`, `Timeout`）。
* 执行指数退避算法：$T_{wait} = \min(T_{max}, 2^c \cdot T_{base} + \text{jitter})$ （其中 $c$ 为重试次数）。
* 超过最大重试次数的任务，转移至 **Dead Letter Queue (死信队列)**。



### 3.4 核心组件层 (Core Components Layer)

* **LLM Clients (模型客户端)：** 抽象基类 `BaseLLMClient`，派生出 `OpenAIClient`, `vLLMClient` 等，实现请求参数的统一映射。
* **Strategies (策略组件)：** 如 `EvolInstruct`, `SelfPlay`。内部包含 Prompt 模板（Jinja2）和生成逻辑。
* **Evaluators (评估组件)：** 如 `LLMJudgeEvaluator`, `RegexFilter`。
* **Output Validators (输出校验器)：** 基于 Pydantic 的强制格式校验。若 LLM 输出非法 JSON，触发 Feedback 机制，将错误信息喂给大模型重试。

---

## 四、 关键数据结构定义 (Core Data Structures)

为了满足企业级“数据血缘追踪”的需求，系统内流转的标准数据结构 `TaskPayload` 极其重要。它必须贯穿生成到落盘的全生命周期。

```json
// TaskPayload Schema (流转与落盘的标准 JSONL 结构)
{
  "task_id": "uuid-v4",
  "status": "COMPLETED", // PENDING, GENERATING, EVALUATING, FAILED
  "source_data": {
    "instruction": "原生态的简单问题"
  },
  "synthetic_data": {
    "instruction": "经过深度演进后的复杂问题",
    "response": "包含 <think> 标签的高质量思考与回答"
  },
  "evaluation": {
    "score": 4.5,
    "reason": "回答逻辑严密，且遵循了 JSON 输出规范。",
    "evaluator_model": "gpt-4o-mini"
  },
  "metadata": {
    "generator_model": "Qwen2.5-7B-Instruct",
    "strategy_used": "evol_instruct_depth_2",
    "total_tokens_used": 1450,
    "retries": 1,
    "timestamp": "2026-03-05T19:20:00Z"
  }
}

```

---

## 五、 核心处理时序图 (Core Sequence Diagram)

以一条种子数据经过“演进生成 -> 格式校验 -> 质量打分”的 Happy Path（包含一次格式修正重试）为例：

1. **DAG Manager** 从输入源读取一条 `TaskPayload`，推入**执行引擎**。
2. **执行引擎** 向**令牌桶 (Rate Limiter)** 申请 Token 额度。获取额度后，调用 **Strategy (Evol-Instruct)**。
3. **Strategy** 组装 Prompt，调用 **LLM Client (vLLM)** 发起异步生成请求。
4. **LLM Client** 返回结果。交由 **Output Validator** 进行 JSON 校验。
5. *(异常分支)* 如果 Validator 发现缺少字段，触发重试：将报错上下文发回给 **LLM Client** 要求修正。
6. **LLM Client** 返回修正后的正确 JSON。
7. **执行引擎** 将生成结果传递给下游的 **Evaluator (LLM-Judge)** 打分。
8. 打分结果 $\ge$ 设定阈值，更新 `TaskPayload` 状态为 `COMPLETED`。
9. **State Manager** 将最终结果追加写入 (Append) `output.jsonl`，并更新终端进度条。

---

## 六、 部署与扩展架构 (Deployment & Scaling Architecture)

考虑到未来向企业版（DataForge Enterprise）演进，架构需具备平滑扩容能力：

* **v1.0 开源版（Local / Single Node）：**
* 用户在一台配备单张或多张 GPU 的工作站上运行。
* DataForge Python SDK 进程与本地部署的 vLLM 进程通过同一台机器的 HTTP/UDS 通信，实现极低延迟。


* **v2.0 商业版展望（Distributed / Cloud Native）：**
* 引入 Redis 作为全局状态管理和分布式锁。
* 将 `Async Task Queue` 升级为 RabbitMQ 或 Kafka。
* DataForge Worker 可以通过 Kubernetes 进行横向水平扩容 (HPA)，支持多租户、PB 级别的数据合成调度。



---
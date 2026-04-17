这份《DataForge Python SDK API 接口设计文档》将详细定义开发者在实际编写代码时，如何与我们的系统进行交互。

在设计这套 API 时，我们的核心哲学是：**“像搭乐高积木一样组合（Composability），像写 Keras 一样优雅（Elegance），并在底层屏蔽所有高并发的脏活累活。”**

---

# 💻 DataForge Python SDK API 接口设计文档 (v1.0)

## 一、 设计规范与基础约定

1. **Async-First (异步优先):** 所有涉及网络 I/O 的核心方法一律使用 `async def`，基于 `asyncio` 提供原生的高并发支持。
2. **强类型约束 (Type Hinting):** 全面拥抱 Python 3.10+ 的 Type Hints，并使用 `Pydantic` 进行内部数据结构的严格校验。
3. **流式处理 (Streaming by default):** 数据读取和落盘默认采用流式生成器 (Generators)，确保处理 10GB 的 JSONL 文件也不会撑爆内存。

---

## 二、 核心数据模型 (Data Models)

所有在管道中流转的数据必须是 `DataRecord` 实例。这保证了数据结构的统一和血缘的可追溯性。

### `dataforge.schema.DataRecord`

基于 Pydantic BaseModel 构建的标准数据包。

* **属性 (Attributes):**
* `id` *(str)*: 数据的唯一标识符 (UUID)。
* `seed_data` *(dict)*: 原始输入的种子数据。
* `synthetic_data` *(dict | None)*: 生成器产出的合成数据。
* `score` *(float | None)*: 评估器给出的综合评分。
* `status` *(str)*: 当前状态，枚举值 `["PENDING", "GENERATED", "REJECTED", "COMPLETED", "FAILED"]`。
* `metadata` *(dict)*: 元数据（包含重试次数、消耗的 Token 数、使用的模型版本等）。



---

## 三、 模型客户端接口 (LLM Clients)

将各大模型厂商的 API 差异抹平，对外提供统一的异步生成接口。

### 抽象基类 `dataforge.clients.BaseLLMClient`

所有具体的模型客户端都必须继承此基类并实现 `generate` 方法。

### `dataforge.clients.OpenAIClient` / `vLLMClient` / `AnthropicClient`

* **初始化参数 (Init Params):**
* `model` *(str)*: 模型名称，如 `"Qwen/Qwen2.5-7B-Instruct"`。
* `api_key` *(str | None)*: API 秘钥（本地 vLLM 可留空）。
* `base_url` *(str | None)*: 接口地址。
* `rpm_limit` *(int)*: 每分钟最大请求数限制（触发内置令牌桶）。
* `tpm_limit` *(int)*: 每分钟最大 Token 数限制。
* `generation_kwargs` *(dict)*: 默认生成参数，如 `{"temperature": 0.7, "max_tokens": 2048}`。


* **核心方法 (Methods):**
* `async def generate(self, prompt: str | list[dict], **kwargs) -> str`: 发起异步请求并返回生成的文本。引擎层会自动在此方法外层包裹重试与限流装饰器。



---

## 四、 策略生成器接口 (Strategies)

定义如何将“种子数据”变异、演进为“高质量合成数据”。

### 抽象基类 `dataforge.strategies.BaseStrategy`

* **核心方法:** `async def apply(self, record: DataRecord) -> DataRecord`

### `dataforge.strategies.EvolInstruct` (指令演进策略)

将简单指令演进为极其复杂的推理指令。

* **初始化参数:**
* `llm` *(BaseLLMClient)*: 绑定的生成大模型。
* `depth` *(int)*: 演进的轮数（深度），默认 `3`。
* `mutation_types` *(list[str])*: 允许的演进方向，如 `["constraints", "deepen", "concretize"]`。
* `require_reasoning` *(bool)*: 是否强制要求模型输出 `<think>` 思考过程，默认 `False`。



### `dataforge.strategies.SelfPlay` (多智能体博弈策略)

* **初始化参数:**
* `llm_a` *(BaseLLMClient)*: 扮演正方的模型。
* `llm_b` *(BaseLLMClient)*: 扮演反方的模型。
* `turns` *(int)*: 对话的轮数。



---

## 五、 评估与过滤器接口 (Evaluators)

对生成后的数据进行质量把控和清洗。

### 抽象基类 `dataforge.evaluators.BaseEvaluator`

* **核心方法:** `async def evaluate(self, record: DataRecord) -> bool`: 返回 `True` 表示通过，`False` 表示被过滤。

### `dataforge.evaluators.LLMJudge` (大模型裁判)

使用强大的模型（如 GPT-4o）对生成内容打分。

* **初始化参数:**
* `llm` *(BaseLLMClient)*: 绑定的裁判模型。
* `criteria` *(str | list[str])*: 评分维度，支持内置字符串（如 `"helpfulness"`, `"factuality"`）或自定义 Prompt。
* `threshold` *(float)*: 录取及格线（例如 `4.0` / 5.0 分）。



### `dataforge.evaluators.RegexFilter` (正则过滤器)

极速且零成本的安全/格式过滤。

* **初始化参数:**
* `blacklist_patterns` *(list[str])*: 触发拒绝的正则表达式列表。
* `require_json` *(bool)*: 是否强制检查输出包含有效的 JSON 块。



---

## 六、 核心编排器 (The Pipeline Orchestrator)

将 Client、Strategy 和 Evaluator 串联起来，并接管所有的并发、容错和断点续传。

### `dataforge.Pipeline`

* **初始化参数 (Init Params):**
* `strategy` *(BaseStrategy)*: 必须指定一个主生成策略。
* `evaluators` *(list[BaseEvaluator] | None)*: 可选的评估器列表，按顺序串行执行。
* `checkpoint_dir` *(str)*: 断点续传和运行日志保存目录，默认 `"./.dataforge_runs"`。
* `max_retries` *(int)*: 模型输出不合规时的最大重试次数，默认 `3`。


* **核心方法 (Methods):**
* `async def run(self, input_path: str, output_path: str, concurrency: int = 50)`
* **功能:** 启动端到端的数据处理流水线。
* **参数:**
* `input_path`: 种子数据路径 (`.jsonl`)。
* `output_path`: 合成后的高质量数据输出路径 (`.jsonl`)。
* `concurrency`: 最大并发协程数。底层限流器会确保即使此处设置 `1000`，也不会突破 API 厂商的硬限制。







---

## 七、 实战代码示例 (Hello World)

通过上述 API，用户只需短短几行代码，就能拉起一个**工业级的高并发容错数据管线**：

```python
import asyncio
from dataforge import Pipeline
from dataforge.clients import vLLMClient, OpenAIClient
from dataforge.strategies import EvolInstruct
from dataforge.evaluators import LLMJudge, RegexFilter

async def main():
    # 1. 实例化底层的模型端点
    # 使用本地私有算力做“生成” (省钱、无隐私风险)
    worker_llm = vLLMClient(
        model="Qwen/Qwen2.5-7B-Instruct",
        base_url="http://localhost:8000/v1",
        rpm_limit=1000  # 本地机器性能够强，限制调高
    )
    
    # 使用云端最强模型做“裁判” (花刀刃上的钱)
    judge_llm = OpenAIClient(
        model="gpt-4o-mini",
        api_key="sk-...",
        rpm_limit=500,
        tpm_limit=200000
    )

    # 2. 组装 Pipeline 积木
    pipeline = Pipeline(
        # 核心策略：经过 2 轮变异的指令演进
        strategy=EvolInstruct(llm=worker_llm, depth=2),
        
        # 质量网关：先过正则规则，再过 LLM 裁判
        evaluators=[
            RegexFilter(blacklist_patterns=[r"作为AI", r"我是一个人工智能"]),
            LLMJudge(llm=judge_llm, criteria="logical_reasoning", threshold=4.0)
        ],
        checkpoint_dir="./experiments/run_001"
    )

    # 3. 启动高并发引擎！
    print("🚀 DataForge 引擎启动...")
    await pipeline.run(
        input_path="seed_prompts.jsonl",
        output_path="high_quality_dataset.jsonl",
        concurrency=100  # 自动挂起 100 个并发任务
    )

if __name__ == "__main__":
    asyncio.run(main())

```

---
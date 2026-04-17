import asyncio
import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Protocol

from dataforge import Pipeline, EvolInstruct, RegexFilter
from dataforge.clients import OpenAIClient
from dataforge.evaluators import BaseEvaluator
from dataforge.schema import DataRecord
from dataforge.strategies import BaseStrategy


class LLMProtocol(Protocol):
    async def generate(self, prompt: str) -> str:
        ...


class FakeLLM:
    """用于本地演示的伪 LLM，避免依赖外部 API。"""

    async def generate(self, prompt: str) -> str:
        base = prompt.replace("\n", " ").strip()
        if "简短" in base:
            return "过短输出"
        return (
            "请结合背景信息、关键约束和逐步推理过程，给出完整解答："
            f"{base}，并说明可复用的方法。"
        )


class MyStrategy(BaseStrategy):
    def __init__(self, llm: LLMProtocol):
        self.llm = llm

    async def apply(self, record: DataRecord) -> DataRecord:
        instruction = record.seed_data["instruction"]
        prompt = f"将以下问题扩展为更复杂的推理题：{instruction}"
        response = await self.llm.generate(prompt)
        record.synthetic_data = {"instruction": response}
        return record


class LengthFilter(BaseEvaluator):
    def __init__(self, min_len: int = 50):
        self.min_len = min_len

    async def evaluate(self, record: DataRecord) -> bool:
        text = (record.synthetic_data or {}).get("instruction", "")
        return len(text) >= self.min_len


def load_env_file(env_file: str) -> None:
    path = Path(env_file)
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def write_seed_data(path: Path) -> None:
    seeds = [
        {"id": "seed-001", "instruction": "解释一下 Transformer 的注意力机制"},
        {"id": "seed-002", "instruction": "简短回答：什么是梯度下降"},
        {"id": "seed-003", "instruction": "设计一个高并发 API 的限流方案"},
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in seeds:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def print_output(output_path: Path) -> int:
    if not output_path.exists():
        print("未生成输出文件。")
        return 0

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    print(f"\n完成记录数: {len(records)}")
    for rec in records:
        text = rec.get("synthetic_data", {}).get("instruction", "")
        preview = text[:60] + ("..." if len(text) > 60 else "")
        score = rec.get("score")
        score_str = f" score={score}" if score is not None else ""
        print(f"- id={rec['id']} status={rec['status']}{score_str} text={preview}")
    return len(records)


async def main() -> None:
    parser = argparse.ArgumentParser(description="DataForge 端到端演示")
    parser.add_argument(
        "--backend",
        choices=["fake", "fake-v05", "deepseek"],
        default="deepseek",
        help=(
            "fake=v0.1 自定义策略演示；"
            "fake-v05=v0.5 内置 EvolInstruct+RegexFilter 演示（无需 API Key）；"
            "deepseek=真实 OpenAI 协议接口"
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="并发 worker 数",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="环境变量文件路径（默认: .env）",
    )
    args = parser.parse_args()
    load_env_file(args.env_file)

    base_dir = Path("runs/e2e_demo")
    input_path = base_dir / "seed_prompts.jsonl"
    output_path = base_dir / "high_quality_dataset.jsonl"
    checkpoint_dir = base_dir / "checkpoint"

    write_seed_data(input_path)
    if output_path.exists():
        output_path.unlink()
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    if args.backend == "deepseek":
        # DeepSeek 提供 OpenAI 兼容接口，直接复用 OpenAIClient。
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "未检测到 DEEPSEEK_API_KEY。请先执行：\n"
                "export DEEPSEEK_API_KEY='你的 key'\n"
                "然后重试：python main.py --backend deepseek"
            )
        llm: LLMProtocol = OpenAIClient(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            rpm_limit=60,
            generation_kwargs={"temperature": 0.7, "max_tokens": 256},
        )
        pipeline = Pipeline(
            strategy=MyStrategy(llm=llm),
            evaluators=[LengthFilter(min_len=50)],
            checkpoint_dir=str(checkpoint_dir),
            max_retries=3,
        )

    elif args.backend == "fake-v05":
        # v0.5 Beta 演示：使用内置 EvolInstruct 策略 + RegexFilter 评估器
        # 无需任何 API Key，完全离线运行。
        print("[ v0.5 Beta 演示 ] 使用 EvolInstruct (depth=1) + RegexFilter")
        print("  策略：对种子指令进行一轮指令演进（constraints 变异）")
        print('  过滤：拦截包含\u201c过短输出\u201d的低质量结果\n')
        llm = FakeLLM()
        pipeline = Pipeline(
            strategy=EvolInstruct(
                llm=llm,
                depth=1,
                mutation_types=["constraints"],
            ),
            evaluators=[
                RegexFilter(blacklist_patterns=[r"过短输出"]),
            ],
            checkpoint_dir=str(checkpoint_dir),
            max_retries=3,
        )

    else:  # fake (v0.1 style)
        llm = FakeLLM()
        pipeline = Pipeline(
            strategy=MyStrategy(llm=llm),
            evaluators=[LengthFilter(min_len=50)],
            checkpoint_dir=str(checkpoint_dir),
            max_retries=3,
        )

    await pipeline.run(
        input_path=str(input_path),
        output_path=str(output_path),
        concurrency=args.concurrency,
    )
    completed = print_output(output_path)

    if completed == 0 and args.backend == "deepseek":
        print(
            "\n[提示] DeepSeek 模式下结果为 0。请检查："
            "DEEPSEEK_API_KEY 是否正确、额度是否充足、"
            "DEEPSEEK_BASE_URL/DEEPSEEK_MODEL 是否配置正确。"
        )


if __name__ == "__main__":
    asyncio.run(main())

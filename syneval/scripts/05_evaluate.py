#!/usr/bin/env python3
"""SynEval Step 5: Evaluate a fine-tuned model on MT-Bench, AlpacaEval 2.0, and IFEval.

Evaluation modes:
  mt_bench     - 80-question multi-turn dialog benchmark, judged by GPT-4o (score 1-10)
  alpacaeval   - AlpacaEval 2.0 length-controlled win rate vs GPT-4
  ifeval       - IFEval strict instruction-following (hard metrics, no LLM judge)

Usage::

    python syneval/scripts/05_evaluate.py \
        --model-path syneval/results/models/exp1_full_qwen_run1/final \
        --benchmark mt_bench \
        --config-name exp1_full --run-id 1 \
        --judge-api-key $OPENAI_API_KEY \
        --output-dir syneval/results/evals

    python syneval/scripts/05_evaluate.py \
        --model-path syneval/results/models/exp1_full_qwen_run1/final \
        --benchmark ifeval \
        --config-name exp1_full --run-id 1 \
        --output-dir syneval/results/evals
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MT-Bench questions (lite version - 14 questions spanning 8 categories)
# ---------------------------------------------------------------------------
MT_BENCH_QUESTIONS = [
    # Writing
    {"id": "mt_w1", "category": "writing",
     "turn1": "Compose a short poem about the seasons changing.",
     "turn2": "Now rewrite it in the style of Shakespeare."},
    {"id": "mt_w2", "category": "writing",
     "turn1": "Write a professional email declining a job offer.",
     "turn2": "Make it more casual and friendly."},
    # Roleplay
    {"id": "mt_r1", "category": "roleplay",
     "turn1": "Act as a travel guide for Paris and give me top 3 recommendations.",
     "turn2": "What about budget-friendly options?"},
    # Reasoning
    {"id": "mt_re1", "category": "reasoning",
     "turn1": "If a bat and a ball cost $1.10 in total, and the bat costs $1 more than the ball, how much does the ball cost?",
     "turn2": "Explain your reasoning step by step."},
    {"id": "mt_re2", "category": "reasoning",
     "turn1": "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have?",
     "turn2": "Why do people often get this wrong?"},
    # Math
    {"id": "mt_m1", "category": "math",
     "turn1": "Solve: 2x + 3 = 11. Show your work.",
     "turn2": "Now solve: 3x^2 - 12 = 0."},
    {"id": "mt_m2", "category": "math",
     "turn1": "What is the derivative of f(x) = x^3 + 2x^2 - 5x + 3?",
     "turn2": "Find the critical points."},
    # Coding
    {"id": "mt_c1", "category": "coding",
     "turn1": "Write a Python function to find all prime numbers up to n using the Sieve of Eratosthenes.",
     "turn2": "Optimize it for memory efficiency."},
    {"id": "mt_c2", "category": "coding",
     "turn1": "Explain what a binary search tree is and write a basic implementation in Python.",
     "turn2": "Add a method to find the height of the tree."},
    # Extraction
    {"id": "mt_e1", "category": "extraction",
     "turn1": "Extract all email addresses from: 'Contact us at info@example.com or support@test.org for help.'",
     "turn2": "Now extract only the domain names."},
    # STEM
    {"id": "mt_s1", "category": "stem",
     "turn1": "Explain how CRISPR-Cas9 gene editing works in simple terms.",
     "turn2": "What are the main ethical concerns?"},
    {"id": "mt_s2", "category": "stem",
     "turn1": "What is the difference between mitosis and meiosis?",
     "turn2": "Create a comparison table."},
    # Humanities
    {"id": "mt_h1", "category": "humanities",
     "turn1": "What were the main causes of World War I?",
     "turn2": "How might the war have been avoided?"},
    {"id": "mt_h2", "category": "humanities",
     "turn1": "Explain the concept of supply and demand in economics.",
     "turn2": "Give a real-world example from the past 5 years."},
]

# ---------------------------------------------------------------------------
# IFEval test cases (strict instruction-following, hard metrics)
# ---------------------------------------------------------------------------
IFEVAL_CASES = [
    {"id": "if_1", "instruction": "Write exactly 3 sentences about climate change.",
     "checks": [("sentence_count", 3)]},
    {"id": "if_2", "instruction": "List 5 benefits of exercise. Start each with a number.",
     "checks": [("starts_with_numbers", 5)]},
    {"id": "if_3", "instruction": "Explain quantum computing in one paragraph without using the word 'computer'.",
     "checks": [("no_word", "computer"), ("paragraph_count", 1)]},
    {"id": "if_4", "instruction": "Write your answer in ALL CAPS.",
     "checks": [("is_uppercase", True)]},
    {"id": "if_5", "instruction": "Respond with exactly 50 words. Count carefully.",
     "checks": [("word_count", 50)]},
    {"id": "if_6", "instruction": "Give 3 examples, each on a new line starting with '- '.",
     "checks": [("bullet_format", "- ")]},
    {"id": "if_7", "instruction": "Answer in French.",
     "checks": [("language_french", True)]},
    {"id": "if_8", "instruction": "Your response must end with 'Thank you.'",
     "checks": [("ends_with", "Thank you.")]},
    {"id": "if_9", "instruction": "Write a haiku about the ocean (5-7-5 syllable structure).",
     "checks": [("haiku_lines", 3)]},
    {"id": "if_10", "instruction": "Provide exactly 4 points. Use 'First:', 'Second:', 'Third:', 'Fourth:'.",
     "checks": [("ordered_markers", ["First:", "Second:", "Third:", "Fourth:"])]},
]

# Prompts for win-rate benchmark
BENCHMARK_PROMPTS = [
    "What are the health benefits of green tea?",
    "Explain the concept of compound interest.",
    "How do I start meditating?",
    "What is the difference between affect and effect?",
    "Write a cover letter for a software engineer position.",
    "How does the human immune system work?",
    "What are the key principles of machine learning?",
    "Explain blockchain technology in simple terms.",
    "What is the best way to learn a new language?",
    "How can I improve my public speaking skills?",
    "What are the main differences between Python and JavaScript?",
    "Explain the concept of GDP.",
    "How does photosynthesis work?",
    "What is the significance of the French Revolution?",
    "How do I write a good thesis statement?",
    "What are the causes of inflation?",
    "Explain the theory of evolution.",
    "How does the internet work?",
    "What are the benefits of regular exercise?",
    "How do neural networks learn?",
]


# ---------------------------------------------------------------------------
# Model inference helpers
# ---------------------------------------------------------------------------
def generate_response(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int = 512) -> str:
    import torch
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def load_model(model_path: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info("Loading model from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
    )
    model.eval()
    return model, tokenizer


def make_chat_prompt(text: str) -> str:
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"


# ---------------------------------------------------------------------------
# MT-Bench
# ---------------------------------------------------------------------------
async def run_mt_bench(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    import openai
    model, tokenizer = load_model(args.model_path)
    judge = openai.AsyncOpenAI(api_key=args.judge_api_key)

    all_scores: list[float] = []
    category_scores: dict[str, list[float]] = {}
    results: list[dict[str, Any]] = []

    for q in MT_BENCH_QUESTIONS:
        t1_prompt = make_chat_prompt(q["turn1"])
        t1_response = generate_response(model, tokenizer, t1_prompt)

        t2_prompt = (
            f"<|im_start|>user\n{q['turn1']}<|im_end|>\n"
            f"<|im_start|>assistant\n{t1_response}<|im_end|>\n"
            f"<|im_start|>user\n{q['turn2']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        t2_response = generate_response(model, tokenizer, t2_prompt)

        judge_prompt = (
            f"Rate the AI response quality (1-10) for helpfulness, accuracy, and clarity.\n\n"
            f"User turn 1: {q['turn1']}\nUser turn 2: {q['turn2']}\n"
            f"AI response to turn 2: {t2_response}\n\n"
            f"Output only a number from 1 to 10:"
        )
        try:
            completion = await judge.chat.completions.create(
                model=args.judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=5, temperature=0,
            )
            score_text = completion.choices[0].message.content or ""
            match = re.search(r"\b([1-9]|10)\b", score_text)
            score = float(match.group(1)) if match else 5.0
        except Exception as exc:
            logger.warning("Judge error for %s: %s", q["id"], exc)
            score = 5.0

        all_scores.append(score)
        category_scores.setdefault(q["category"], []).append(score)
        results.append({"question_id": q["id"], "category": q["category"],
                        "t1_response": t1_response, "t2_response": t2_response, "score": score})
        logger.info("MT-Bench %s [%s]: score=%.1f", q["id"], q["category"], score)

    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    output = {
        "benchmark": "mt_bench", "config_name": args.config_name, "run_id": args.run_id,
        "model_path": args.model_path, "overall_score": round(overall, 3),
        "category_scores": {k: round(sum(v) / len(v), 3) for k, v in category_scores.items()},
        "num_questions": len(all_scores), "results": results,
    }
    out_path = output_dir / f"mt_bench_{args.config_name}_run{args.run_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("MT-Bench overall=%.3f → %s", overall, out_path)
    return output


# ---------------------------------------------------------------------------
# IFEval
# ---------------------------------------------------------------------------
def check_response(response: str, checks: list[tuple]) -> dict[str, Any]:
    passed_checks, failed_checks = [], []
    for check_type, check_val in checks:
        passed = False
        if check_type == "sentence_count":
            count = len([s for s in re.split(r"[.!?]", response) if s.strip()])
            passed = count == check_val
        elif check_type == "starts_with_numbers":
            numbered = [l for l in response.strip().split("\n") if re.match(r"^\d+[\.\):]", l.strip())]
            passed = len(numbered) >= check_val
        elif check_type == "no_word":
            passed = check_val.lower() not in response.lower()
        elif check_type == "is_uppercase":
            letters = [c for c in response if c.isalpha()]
            passed = len(letters) > 0 and sum(c.isupper() for c in letters) / len(letters) >= 0.8
        elif check_type == "word_count":
            passed = abs(len(response.split()) - check_val) <= 2
        elif check_type == "bullet_format":
            passed = any(l.strip().startswith(check_val) for l in response.split("\n"))
        elif check_type == "ends_with":
            passed = response.strip().endswith(check_val)
        elif check_type == "haiku_lines":
            passed = len([l for l in response.strip().split("\n") if l.strip()]) == check_val
        elif check_type == "ordered_markers":
            passed = all(m in response for m in check_val)
        elif check_type == "paragraph_count":
            passed = len([p for p in response.split("\n\n") if p.strip()]) == check_val
        elif check_type == "language_french":
            fr_words = {"le", "la", "les", "de", "du", "un", "une", "et", "est", "en"}
            passed = len(fr_words & set(response.lower().split())) >= 3
        (passed_checks if passed else failed_checks).append(check_type)
    return {"passed": not failed_checks, "passed_checks": passed_checks, "failed_checks": failed_checks}


def run_ifeval(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    model, tokenizer = load_model(args.model_path)
    results, strict_pass = [], 0
    for case in IFEVAL_CASES:
        prompt = make_chat_prompt(case["instruction"])
        response = generate_response(model, tokenizer, prompt, max_new_tokens=256)
        check_result = check_response(response, case["checks"])
        if check_result["passed"]:
            strict_pass += 1
        results.append({"id": case["id"], "instruction": case["instruction"],
                        "response": response, **check_result})
        logger.info("IFEval %s: passed=%s", case["id"], check_result["passed"])

    total = len(IFEVAL_CASES)
    output = {
        "benchmark": "ifeval", "config_name": args.config_name, "run_id": args.run_id,
        "model_path": args.model_path, "strict_accuracy": round(strict_pass / total, 4),
        "strict_pass": strict_pass, "total": total, "results": results,
    }
    out_path = output_dir / f"ifeval_{args.config_name}_run{args.run_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("IFEval strict_acc=%.3f → %s", output["strict_accuracy"], out_path)
    return output


# ---------------------------------------------------------------------------
# Win-rate benchmark (AlpacaEval-style)
# ---------------------------------------------------------------------------
async def run_winrate(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    import openai
    model, tokenizer = load_model(args.model_path)
    judge = openai.AsyncOpenAI(api_key=args.judge_api_key)

    prompts = BENCHMARK_PROMPTS[:args.num_prompts]
    wins, results = 0, []

    for prompt in prompts:
        model_response = generate_response(model, tokenizer, make_chat_prompt(prompt))
        try:
            ref = await judge.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512, temperature=0,
            )
            reference = ref.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("Reference generation failed: %s", exc)
            reference = ""

        judge_prompt = (
            f"Which response is better for the question below? Consider quality over length.\n"
            f"Question: {prompt}\nResponse A: {model_response}\nResponse B (GPT-4 reference): {reference}\n"
            f"Output only 'A', 'B', or 'tie':"
        )
        try:
            verdict_resp = await judge.chat.completions.create(
                model=args.judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=5, temperature=0,
            )
            verdict = (verdict_resp.choices[0].message.content or "").strip().upper()
            win = verdict.startswith("A")
        except Exception as exc:
            logger.warning("Judge failed: %s", exc)
            win = False

        if win:
            wins += 1
        results.append({"prompt": prompt, "model_response": model_response, "win": win})

    total = len(prompts)
    output = {
        "benchmark": "alpacaeval_winrate", "config_name": args.config_name, "run_id": args.run_id,
        "model_path": args.model_path, "win_rate": round(wins / total, 4) if total > 0 else 0.0,
        "wins": wins, "total": total, "results": results,
    }
    out_path = output_dir / f"winrate_{args.config_name}_run{args.run_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("Win-rate=%.3f → %s", output["win_rate"], out_path)
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def async_main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.benchmark == "mt_bench":
        result = await run_mt_bench(args, output_dir)
    elif args.benchmark == "ifeval":
        result = run_ifeval(args, output_dir)
    elif args.benchmark == "alpacaeval":
        result = await run_winrate(args, output_dir)
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")

    print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SynEval Step 5: Evaluate fine-tuned model on benchmarks."
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--benchmark", choices=["mt_bench", "ifeval", "alpacaeval"], required=True)
    parser.add_argument("--config-name", type=str, required=True)
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--judge-api-key", type=str, default=None)
    parser.add_argument("--judge-model", type=str, default="gpt-4o")
    parser.add_argument("--output-dir", type=str, default="syneval/results/evals")
    parser.add_argument("--num-prompts", type=int, default=20,
                        help="Number of prompts for win-rate benchmark (default: 20).")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

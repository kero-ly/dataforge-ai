from __future__ import annotations

import json
import re
import time
from pathlib import Path

from dataforge.benchmark.schema import BenchmarkCase, BenchmarkCaseResult, BenchmarkTaskSummary
from dataforge.registry import register_benchmark


@register_benchmark("mt_bench_lite_v1")
class MTBenchLiteTask:
    name = "mt_bench_lite_v1"
    version = "v1"

    def load_cases(self) -> list[BenchmarkCase]:
        path = Path(__file__).resolve().parents[1] / "data" / "mt_bench_lite_v1.jsonl"
        cases: list[BenchmarkCase] = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    cases.append(BenchmarkCase.model_validate_json(line))
        return cases

    async def run_case(self, case: BenchmarkCase, candidate: object, judge: object | None) -> BenchmarkCaseResult:
        started = time.monotonic()
        response = await candidate.generate(case.prompt)
        if judge is None:
            raise ValueError("mt_bench_lite_v1 requires a judge client")
        judge_prompt = (
            "Rate the following response on a scale of 1-10. Only output the number.\n\n"
            f"Question: {case.prompt}\n\nResponse: {response}\n"
        )
        judge_text = await judge.score(judge_prompt)
        match = re.findall(r"\b(\d+(?:\.\d+)?)\b", judge_text)
        raw_score = float(match[-1]) if match else 0.0
        raw_score = min(10.0, max(0.0, raw_score))
        return BenchmarkCaseResult(
            case_id=case.id,
            category=case.category,
            prompt=case.prompt,
            response=response,
            raw_score=raw_score,
            normalized_score=round(raw_score * 10.0, 2),
            passed=raw_score >= 6.0,
            details={"judge_response": judge_text},
            duration_ms=round((time.monotonic() - started) * 1000.0, 3),
        )

    def summarize(self, case_results: list[BenchmarkCaseResult]) -> BenchmarkTaskSummary:
        category_scores: dict[str, list[float]] = {}
        errors = 0
        successes = 0
        for result in case_results:
            if result.error:
                errors += 1
            else:
                successes += 1
            if result.normalized_score is not None:
                category_scores.setdefault(result.category, []).append(result.normalized_score)
        averaged = {
            category: round(sum(scores) / len(scores), 2)
            for category, scores in category_scores.items()
            if scores
        }
        flat = [score for scores in category_scores.values() for score in scores]
        overall = round(sum(flat) / len(flat), 2) if flat else 0.0
        return BenchmarkTaskSummary(
            task=self.name,
            category_scores=averaged,
            overall_score=overall,
            success_rate=(successes / len(case_results)) if case_results else 0.0,
            num_cases=len(case_results),
            num_errors=errors,
        )

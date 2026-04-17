from __future__ import annotations

import json
import re
import time
from pathlib import Path

from dataforge.benchmark.schema import BenchmarkCase, BenchmarkCaseResult, BenchmarkTaskSummary
from dataforge.registry import register_benchmark


def _run_check(response: str, check: dict[str, object]) -> bool:
    kind = str(check.get("type"))
    value = check.get("value")
    if kind == "contains":
        return str(value).lower() in response.lower()
    if kind == "not_contains":
        return str(value).lower() not in response.lower()
    if kind == "max_words":
        return len(response.split()) <= int(value)
    if kind == "min_list_items":
        items = [line for line in response.splitlines() if re.match(r"^\s*(?:[-*]|\d+\.)\s+", line)]
        return len(items) >= int(value)
    if kind == "json_parseable":
        try:
            json.loads(response)
            return True
        except json.JSONDecodeError:
            return False
    if kind == "starts_with":
        return response.strip().lower().startswith(str(value).lower())
    return False


@register_benchmark("if_eval_lite_v1")
class IFEvalLiteTask:
    name = "if_eval_lite_v1"
    version = "v1"

    def load_cases(self) -> list[BenchmarkCase]:
        path = Path(__file__).resolve().parents[1] / "data" / "if_eval_lite_v1.jsonl"
        cases: list[BenchmarkCase] = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    cases.append(BenchmarkCase.model_validate_json(line))
        return cases

    async def run_case(self, case: BenchmarkCase, candidate: object, judge: object | None) -> BenchmarkCaseResult:
        del judge
        started = time.monotonic()
        response = await candidate.generate(case.prompt)
        checks = list(case.metadata.get("checks", []))
        passed = sum(1 for check in checks if _run_check(response, check))
        total = len(checks)
        score = (passed / total * 100.0) if total else 0.0
        return BenchmarkCaseResult(
            case_id=case.id,
            category=case.category,
            prompt=case.prompt,
            response=response,
            raw_score=score,
            normalized_score=round(score, 2),
            passed=(passed == total) if total else False,
            details={"checks_total": total, "checks_passed": passed},
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
        return BenchmarkTaskSummary(
            task=self.name,
            category_scores=averaged,
            overall_score=round(sum(flat) / len(flat), 2) if flat else 0.0,
            success_rate=(successes / len(case_results)) if case_results else 0.0,
            num_cases=len(case_results),
            num_errors=errors,
        )

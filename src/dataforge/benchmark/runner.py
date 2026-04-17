from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import dataforge.benchmark.tasks  # noqa: F401

from dataforge.benchmark.adapters import LLMCandidateAdapter, LLMJudgeAdapter
from dataforge.benchmark.reporter import write_benchmark_report
from dataforge.benchmark.schema import BenchmarkCaseResult, BenchmarkRunSummary
from dataforge.registry import get_benchmark


class BenchmarkRunner:
    def __init__(self, config: Any) -> None:
        self.config = config
        self.candidate = LLMCandidateAdapter(config._candidate_client)
        self.judge = (
            LLMJudgeAdapter(config._judge_client)
            if hasattr(config, "_judge_client")
            else None
        )

    async def run(self) -> tuple[Path, BenchmarkRunSummary]:
        task_results: dict[str, list[BenchmarkCaseResult]] = {}
        task_summaries = []
        warnings: list[str] = []
        weights = self._resolved_weights()
        for task_config in self.config.tasks:
            registered = get_benchmark(task_config.name)
            task = registered() if isinstance(registered, type) else registered
            cases = task.load_cases()
            results = await self._run_task(task, cases)
            summary = task.summarize(results)
            task_results[task.name] = results
            task_summaries.append(summary)
            if summary.num_cases and (summary.num_errors / summary.num_cases) > 0.25:
                warnings.append(
                    f"{task.name} error rate exceeded 25% ({summary.num_errors}/{summary.num_cases})"
                )
        if warnings:
            raise RuntimeError("; ".join(warnings))

        weighted_scores = {
            summary.task: round((summary.overall_score or 0.0) * weights[summary.task], 4)
            for summary in task_summaries
        }
        overall = round(sum(weighted_scores.values()), 2) if weighted_scores else None
        summary = BenchmarkRunSummary(
            benchmark=self.config.name,
            candidate_name=self.config.candidate.model,
            task_summaries=task_summaries,
            overall_score=overall,
            weighted_scores=weighted_scores,
            warnings=[],
        )
        output_dir = write_benchmark_report(
            run_name=self.config.name,
            benchmark_name=self.config.name,
            candidate_name=self.config.candidate.model,
            summary=summary,
            task_results=task_results,
            output_dir=self.config.output.dir,
            output_formats=list(self.config.output.formats),
            config_snapshot=self.config.model_dump(mode="json"),
        )
        return output_dir, summary

    async def _run_task(self, task: object, cases: list[object]) -> list[BenchmarkCaseResult]:
        semaphore = asyncio.Semaphore(max(1, int(getattr(self.config.candidate, "concurrency", 8))))
        results: list[BenchmarkCaseResult] = []

        async def _run_case(case: object) -> None:
            async with semaphore:
                attempt = 0
                while True:
                    try:
                        result = await task.run_case(case, self.candidate, self.judge)
                        results.append(result)
                        return
                    except Exception as exc:
                        attempt += 1
                        if attempt > 2:
                            results.append(
                                BenchmarkCaseResult(
                                    case_id=case.id,
                                    category=case.category,
                                    prompt=case.prompt,
                                    response="",
                                    raw_score=0.0,
                                    normalized_score=0.0,
                                    passed=False,
                                    error=str(exc),
                                )
                            )
                            return

        await asyncio.gather(*[_run_case(case) for case in cases])
        results.sort(key=lambda item: item.case_id)
        return results

    def _resolved_weights(self) -> dict[str, float]:
        specified = {task.name: task.weight for task in self.config.tasks if task.weight is not None}
        if len(specified) == len(self.config.tasks):
            total = sum(float(weight) for weight in specified.values())
            return {name: float(weight) / total for name, weight in specified.items()}
        remaining = [task for task in self.config.tasks if task.weight is None]
        specified_total = sum(float(weight) for weight in specified.values())
        remaining_total = max(0.0, 1.0 - specified_total)
        default = (remaining_total / len(remaining)) if remaining else 0.0
        weights = {
            task.name: float(task.weight) if task.weight is not None else default
            for task in self.config.tasks
        }
        total = sum(weights.values()) or 1.0
        return {name: value / total for name, value in weights.items()}

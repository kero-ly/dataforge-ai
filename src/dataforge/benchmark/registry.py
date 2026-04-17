from __future__ import annotations

from typing import Protocol

from dataforge.benchmark.schema import BenchmarkCase, BenchmarkCaseResult, BenchmarkTaskSummary


class BenchmarkTask(Protocol):
    name: str
    version: str

    def load_cases(self) -> list[BenchmarkCase]: ...

    async def run_case(self, case: BenchmarkCase, candidate: object, judge: object | None) -> BenchmarkCaseResult: ...

    def summarize(self, case_results: list[BenchmarkCaseResult]) -> BenchmarkTaskSummary: ...

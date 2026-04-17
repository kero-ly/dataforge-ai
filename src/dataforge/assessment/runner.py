from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Any

import dataforge.assessment.suite  # noqa: F401

from dataforge.assessment.aggregator import build_dataset_summary
from dataforge.assessment.normalizer import normalize_row
from dataforge.assessment.reporter import write_assessment_report
from dataforge.assessment.schema import RecordAssessment
from dataforge.registry import get_assessment_suite


class AssessmentRunner:
    def __init__(self, config: Any) -> None:
        self.config = config
        registered = get_assessment_suite(config.suite.name)
        self.suite = registered() if isinstance(registered, type) else registered
        self.evaluators = self.suite.build_evaluators(config)

    async def run(self) -> tuple[Path, Any]:
        source_path = Path(self.config.source.path)
        rows = self._load_rows(source_path)
        total_records = len(rows)
        sampled = self._sample_rows(rows)
        if getattr(self.config, "reference_corpus", None) and self.config.reference_corpus.enabled:
            object.__setattr__(
                self.config,
                "_reference_responses",
                self._load_reference_responses(Path(self.config.reference_corpus.path)),
            )
        records = await self._assess_rows(sampled, str(source_path))
        dataset_metrics = await self.suite.compute_dataset_metrics(records, self.config)
        warnings = self._build_warnings(records)
        overall_score = self.suite.aggregate_score(records, dataset_metrics)
        summary = build_dataset_summary(
            suite_name=self.config.suite.name,
            total_records=total_records,
            sampled_records=len(sampled),
            sample_seed=self.config.suite.sample_seed,
            overall_quality_score=overall_score,
            records=records,
            dataset_metrics=dataset_metrics,
            warnings=warnings,
        )
        output_path = write_assessment_report(
            run_name=self.config.name,
            suite_name=self.config.suite.name,
            source_path=self.config.source.path,
            summary=summary,
            records=records,
            output_dir=self.config.output.dir,
            output_formats=list(self.config.output.formats),
            persist_record_results=self.config.output.persist_record_results,
            config_snapshot=self.config.model_dump(mode="json"),
        )
        return output_path, summary

    def _load_rows(self, source_path: Path) -> list[tuple[int, dict[str, Any]]]:
        rows: list[tuple[int, dict[str, Any]]] = []
        with open(source_path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                rows.append((line_number, json.loads(line)))
        return rows

    def _sample_rows(self, rows: list[tuple[int, dict[str, Any]]]) -> list[tuple[int, dict[str, Any]]]:
        sample_size = min(self.config.suite.sample_size, len(rows))
        if sample_size >= len(rows):
            return rows
        rng = random.Random(self.config.suite.sample_seed)
        indices = sorted(rng.sample(range(len(rows)), sample_size))
        return [rows[idx] for idx in indices]

    async def _assess_rows(
        self,
        rows: list[tuple[int, dict[str, Any]]],
        source_path: str,
    ) -> list[RecordAssessment]:
        semaphore = asyncio.Semaphore(min(20, getattr(self.config, "max_concurrency", 20)))
        assessed: list[RecordAssessment] = []

        async def _assess_one(line_number: int, raw: dict[str, Any]) -> None:
            async with semaphore:
                record = normalize_row(
                    raw,
                    line_number=line_number,
                    source_path=source_path,
                    source_format=self.config.source.format,
                )
                results = []
                for evaluator in self.evaluators:
                    results.append(await evaluator.assess(record))
                numeric_scores = [result.score for result in results if result.score is not None]
                assessed.append(
                    RecordAssessment(
                        record_id=record.id,
                        line_number=line_number,
                        source_path=source_path,
                        normalized_record=record,
                        results=results,
                        aggregate_score=(
                            round(sum(numeric_scores) / len(numeric_scores), 4)
                            if numeric_scores
                            else None
                        ),
                        passed_all_required=all(result.passed for result in results),
                        tags=[],
                    )
                )

        await asyncio.gather(*[_assess_one(line_number, raw) for line_number, raw in rows])
        assessed.sort(key=lambda item: item.line_number)
        return assessed

    def _load_reference_responses(self, path: Path) -> list[str]:
        responses: list[str] = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                if "synthetic_data" in raw:
                    responses.append(str((raw.get("synthetic_data") or {}).get("response", "")))
                else:
                    responses.append(str(raw.get("response", raw.get("output", ""))))
        return [response for response in responses if response.strip()]

    def _build_warnings(self, records: list[RecordAssessment]) -> list[str]:
        warnings: list[str] = []
        by_evaluator: dict[str, list[bool]] = {}
        for record in records:
            for result in record.results:
                by_evaluator.setdefault(result.evaluator, []).append(result.passed)
        for evaluator, values in sorted(by_evaluator.items()):
            failure_rate = 1.0 - (sum(values) / len(values))
            if failure_rate > 0.25:
                warnings.append(f"{evaluator} failure rate exceeded 25% ({failure_rate:.1%})")
        return warnings

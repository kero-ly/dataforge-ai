from __future__ import annotations

from typing import Any

from dataforge.schema import DataRecord

SUPPORTED_SOURCE_FORMATS = {
    "auto",
    "dataforge_jsonl",
    "instruction_response_jsonl",
}


def detect_source_format(raw: dict[str, Any]) -> str:
    if "seed_data" in raw or "synthetic_data" in raw:
        return "dataforge_jsonl"
    if "instruction" in raw and ("response" in raw or "output" in raw):
        return "instruction_response_jsonl"
    raise ValueError("Unable to detect source format for assessment input row")


def normalize_row(
    raw: dict[str, Any],
    *,
    line_number: int,
    source_path: str,
    source_format: str = "auto",
) -> DataRecord:
    if source_format not in SUPPORTED_SOURCE_FORMATS:
        raise ValueError(
            f"Unsupported source format {source_format!r}. "
            f"Expected one of {sorted(SUPPORTED_SOURCE_FORMATS)}."
        )

    resolved_format = detect_source_format(raw) if source_format == "auto" else source_format

    if resolved_format == "dataforge_jsonl":
        record = DataRecord.model_validate(raw)
        if not record.id:
            record.id = f"{source_path}:{line_number}"
        return record

    if resolved_format == "instruction_response_jsonl":
        return DataRecord(
            id=str(raw.get("id") or f"{source_path}:{line_number}"),
            seed_data={"instruction": raw.get("instruction", "")},
            synthetic_data={"response": raw.get("response", raw.get("output", ""))},
            metadata={"source_format": resolved_format},
        )

    raise ValueError(f"Unhandled source format: {resolved_format}")

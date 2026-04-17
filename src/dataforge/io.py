# src/dataforge/io.py
"""Multi-format I/O readers and writers for pipeline data."""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def read_jsonl(path: str) -> list[dict[str, Any]]:
    """Read records from a JSONL file."""
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_csv(path: str) -> list[dict[str, Any]]:
    """Read records from a CSV file. Each row becomes a dict keyed by column headers."""
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))
    return records


def read_parquet(path: str) -> list[dict[str, Any]]:
    """Read records from a Parquet file. Requires pyarrow."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "pyarrow is required for Parquet support. "
            "Install it with: pip install pyarrow"
        ) from None

    table = pq.read_table(path)
    return table.to_pylist()


def read_records(path: str, fmt: str) -> list[dict[str, Any]]:
    """Read records from a file in the specified format.

    Args:
        path: Path to the input file.
        fmt: File format - "jsonl", "csv", or "parquet".

    Returns:
        List of record dicts.
    """
    readers = {
        "jsonl": read_jsonl,
        "csv": read_csv,
        "parquet": read_parquet,
    }
    reader = readers.get(fmt)
    if reader is None:
        raise ValueError(f"Unsupported input format {fmt!r}. Supported: {sorted(readers)}")
    logger.info("Reading %s file: %s", fmt, path)
    return reader(path)


def write_jsonl_records(records: list[dict[str, Any]], path: str) -> None:
    """Write records to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv_records(records: list[dict[str, Any]], path: str) -> None:
    """Write records to a CSV file."""
    if not records:
        Path(path).touch()
        return
    # Collect all keys across all records for the header
    fieldnames: list[str] = []
    seen: set[str] = set()
    for rec in records:
        for key in rec:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            # Serialize nested objects to JSON strings
            flat: dict[str, Any] = {}
            for k, v in record.items():
                flat[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
            writer.writerow(flat)


def write_parquet_records(records: list[dict[str, Any]], path: str) -> None:
    """Write records to a Parquet file. Requires pyarrow."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "pyarrow is required for Parquet support. "
            "Install it with: pip install pyarrow"
        ) from None

    if not records:
        # Write an empty parquet file
        schema = pa.schema([("empty", pa.string())])
        table = pa.table({"empty": []}, schema=schema)
        pq.write_table(table, path)
        return

    # Serialize nested values to JSON strings for Parquet compatibility
    flat_records: list[dict[str, Any]] = []
    for rec in records:
        flat: dict[str, Any] = {}
        for k, v in rec.items():
            flat[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
        flat_records.append(flat)

    table = pa.Table.from_pylist(flat_records)
    pq.write_table(table, path)


def write_records(records: list[dict[str, Any]], path: str, fmt: str) -> None:
    """Write records to a file in the specified format.

    Args:
        records: List of record dicts.
        path: Output file path.
        fmt: File format - "jsonl", "csv", or "parquet".
    """
    writers = {
        "jsonl": write_jsonl_records,
        "csv": write_csv_records,
        "parquet": write_parquet_records,
    }
    writer = writers.get(fmt)
    if writer is None:
        raise ValueError(f"Unsupported output format {fmt!r}. Supported: {sorted(writers)}")
    logger.info("Writing %s file: %s", fmt, path)
    writer(records, path)

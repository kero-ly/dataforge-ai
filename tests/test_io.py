# tests/test_io.py
"""Tests for multi-format I/O (CSV, Parquet, JSONL)."""
import csv
import json
import tempfile
from pathlib import Path

import pytest

from dataforge.io import (
    read_csv,
    read_jsonl,
    read_records,
    write_csv_records,
    write_jsonl_records,
    write_records,
)

# --- JSONL ---

def test_read_jsonl():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"id": "1", "instruction": "hello"}\n')
        f.write('{"id": "2", "instruction": "world"}\n')
        f.write("\n")  # empty line
        path = f.name

    records = read_jsonl(path)
    assert len(records) == 2
    assert records[0]["id"] == "1"
    assert records[1]["instruction"] == "world"


def test_write_jsonl():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name

    records = [{"a": 1}, {"b": 2}]
    write_jsonl_records(records, path)

    lines = Path(path).read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"a": 1}


# --- CSV ---

def test_read_csv():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "instruction"])
        writer.writeheader()
        writer.writerow({"id": "1", "instruction": "hello"})
        writer.writerow({"id": "2", "instruction": "world"})
        path = f.name

    records = read_csv(path)
    assert len(records) == 2
    assert records[0]["id"] == "1"
    assert records[1]["instruction"] == "world"


def test_write_csv():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name

    records = [
        {"id": "1", "data": {"nested": True}},
        {"id": "2", "data": {"nested": False}},
    ]
    write_csv_records(records, path)

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["id"] == "1"
    # Nested dict should be JSON serialized
    assert json.loads(rows[0]["data"]) == {"nested": True}


def test_write_csv_empty():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name

    write_csv_records([], path)
    assert Path(path).exists()


# --- read_records / write_records dispatch ---

def test_read_records_jsonl():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"x": 1}\n')
        path = f.name

    records = read_records(path, "jsonl")
    assert len(records) == 1


def test_read_records_csv():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["x"])
        writer.writeheader()
        writer.writerow({"x": "1"})
        path = f.name

    records = read_records(path, "csv")
    assert len(records) == 1


def test_read_records_unsupported():
    with pytest.raises(ValueError, match="Unsupported input format"):
        read_records("test.txt", "xml")


def test_write_records_unsupported():
    with pytest.raises(ValueError, match="Unsupported output format"):
        write_records([], "test.txt", "xml")


def test_write_records_jsonl():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name

    write_records([{"a": 1}], path, "jsonl")
    lines = Path(path).read_text().strip().splitlines()
    assert len(lines) == 1


def test_write_records_csv():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name

    write_records([{"a": 1}], path, "csv")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1


# --- Parquet (optional) ---

def test_read_parquet():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        pytest.skip("pyarrow not installed")

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name

    table = pa.table({"id": ["1", "2"], "instruction": ["hello", "world"]})
    pq.write_table(table, path)

    from dataforge.io import read_parquet
    records = read_parquet(path)
    assert len(records) == 2
    assert records[0]["id"] == "1"


def test_write_parquet():
    try:
        import pyarrow.parquet as pq
    except ImportError:
        pytest.skip("pyarrow not installed")

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name

    from dataforge.io import write_parquet_records
    records = [{"id": "1", "text": "hello"}, {"id": "2", "text": "world"}]
    write_parquet_records(records, path)

    table = pq.read_table(path)
    assert table.num_rows == 2


def test_write_parquet_empty():
    try:
        import pyarrow.parquet as pq
    except ImportError:
        pytest.skip("pyarrow not installed")

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name

    from dataforge.io import write_parquet_records
    write_parquet_records([], path)

    table = pq.read_table(path)
    assert table.num_rows == 0

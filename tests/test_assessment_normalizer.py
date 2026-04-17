import json
import tempfile
from pathlib import Path

import pytest

from dataforge.assessment.normalizer import detect_source_format, normalize_row


def test_detect_source_format_variants():
    assert detect_source_format({"seed_data": {}, "synthetic_data": {}}) == "dataforge_jsonl"
    assert detect_source_format({"instruction": "q", "response": "a"}) == "instruction_response_jsonl"


def test_normalize_dataforge_row():
    record = normalize_row(
        {"id": "r1", "seed_data": {"instruction": "q"}, "synthetic_data": {"response": "a"}},
        line_number=1,
        source_path="demo.jsonl",
    )
    assert record.id == "r1"
    assert record.synthetic_data["response"] == "a"


def test_normalize_instruction_response_row():
    record = normalize_row(
        {"instruction": "q", "output": "a"},
        line_number=2,
        source_path="demo.jsonl",
    )
    assert record.id == "demo.jsonl:2"
    assert record.seed_data["instruction"] == "q"
    assert record.synthetic_data["response"] == "a"


def test_normalize_unknown_format_raises():
    with pytest.raises(ValueError):
        normalize_row({"foo": "bar"}, line_number=1, source_path="demo.jsonl")

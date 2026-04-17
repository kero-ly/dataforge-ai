# tests/test_length_filter.py
"""Tests for the LengthFilter evaluator."""
from dataforge.evaluators.length_filter import LengthFilter
from dataforge.schema import DataRecord


async def test_length_filter_accepts_above_min():
    f = LengthFilter(min_length=5)
    record = DataRecord(seed_data={}, synthetic_data={"response": "hello world"})
    assert await f.evaluate(record) is True


async def test_length_filter_rejects_below_min():
    f = LengthFilter(min_length=100)
    record = DataRecord(seed_data={}, synthetic_data={"response": "short"})
    assert await f.evaluate(record) is False


async def test_length_filter_max_length():
    f = LengthFilter(max_length=10)
    record = DataRecord(seed_data={}, synthetic_data={"response": "hello"})
    assert await f.evaluate(record) is True

    record2 = DataRecord(seed_data={}, synthetic_data={"response": "x" * 100})
    assert await f.evaluate(record2) is False


async def test_length_filter_range():
    f = LengthFilter(min_length=5, max_length=20)
    short = DataRecord(seed_data={}, synthetic_data={"response": "hi"})
    ok = DataRecord(seed_data={}, synthetic_data={"response": "hello world"})
    long = DataRecord(seed_data={}, synthetic_data={"response": "x" * 50})

    assert await f.evaluate(short) is False
    assert await f.evaluate(ok) is True
    assert await f.evaluate(long) is False


async def test_length_filter_specific_field():
    f = LengthFilter(min_length=3, field="instruction")
    record = DataRecord(
        seed_data={},
        synthetic_data={"instruction": "test question", "extra": "a"},
    )
    assert await f.evaluate(record) is True


async def test_length_filter_specific_field_missing():
    f = LengthFilter(min_length=1, field="nonexistent")
    record = DataRecord(seed_data={}, synthetic_data={"response": "hello"})
    assert await f.evaluate(record) is False


async def test_length_filter_no_synthetic_data():
    f = LengthFilter(min_length=0)
    record = DataRecord(seed_data={})
    assert await f.evaluate(record) is True


async def test_length_filter_defaults_accept_all():
    f = LengthFilter()
    record = DataRecord(seed_data={}, synthetic_data={"response": "anything"})
    assert await f.evaluate(record) is True


async def test_length_filter_registered():
    from dataforge.registry import get_evaluator

    cls = get_evaluator("length-filter")
    assert cls is LengthFilter

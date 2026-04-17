# tests/test_regex_filter.py
import inspect

from dataforge.evaluators.base import BaseEvaluator
from dataforge.evaluators.regex_filter import RegexFilter
from dataforge.schema import DataRecord


def make_record(text: str) -> DataRecord:
    r = DataRecord(seed_data={"instruction": "test"})
    r.synthetic_data = {"instruction": text}
    return r


async def test_regex_filter_passes_clean_text():
    f = RegexFilter(blacklist_patterns=[r"spam"])
    assert await f.evaluate(make_record("clean answer")) is True


async def test_regex_filter_rejects_blacklisted_text():
    f = RegexFilter(blacklist_patterns=[r"spam"])
    assert await f.evaluate(make_record("this contains spam")) is False


async def test_regex_filter_no_patterns_always_passes():
    f = RegexFilter()
    assert await f.evaluate(make_record("anything")) is True


async def test_regex_filter_empty_blacklist_always_passes():
    f = RegexFilter(blacklist_patterns=[])
    assert await f.evaluate(make_record("anything")) is True


async def test_regex_filter_multiple_patterns_any_match_rejects():
    f = RegexFilter(blacklist_patterns=[r"bad", r"evil"])
    assert await f.evaluate(make_record("evil content")) is False
    assert await f.evaluate(make_record("bad content")) is False
    assert await f.evaluate(make_record("good content")) is True


async def test_regex_filter_require_json_passes_valid_json():
    f = RegexFilter(require_json=True)
    r = DataRecord(seed_data={"instruction": "test"})
    r.synthetic_data = {"output": '{"key": "value"}'}
    assert await f.evaluate(r) is True


async def test_regex_filter_require_json_rejects_no_json():
    f = RegexFilter(require_json=True)
    assert await f.evaluate(make_record("not json at all")) is False


async def test_regex_filter_require_json_finds_embedded_json():
    f = RegexFilter(require_json=True)
    r = DataRecord(seed_data={"instruction": "test"})
    r.synthetic_data = {"instruction": 'Here is the result: {"score": 5}'}
    assert await f.evaluate(r) is True


async def test_regex_filter_none_synthetic_data_does_not_raise():
    f = RegexFilter()
    r = DataRecord(seed_data={"instruction": "test"})
    r.synthetic_data = None
    assert await f.evaluate(r) is True


def test_regex_filter_inherits_base_evaluator():
    assert issubclass(RegexFilter, BaseEvaluator)


def test_regex_filter_evaluate_is_coroutine():
    assert inspect.iscoroutinefunction(RegexFilter.evaluate)

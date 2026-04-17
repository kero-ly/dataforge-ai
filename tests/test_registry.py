# tests/test_registry.py
"""Tests for the plugin registry system."""
import pytest

from dataforge.evaluators.base import BaseEvaluator
from dataforge.registry import (
    _evaluator_registry,
    _strategy_registry,
    get_evaluator,
    get_strategy,
    list_evaluators,
    list_strategies,
    register_evaluator,
    register_strategy,
)
from dataforge.schema import DataRecord
from dataforge.strategies.base import BaseStrategy


def test_builtin_strategy_registered():
    """EvolInstruct should be auto-registered when strategies are imported."""
    import dataforge.strategies  # noqa: F401

    assert "evol-instruct" in _strategy_registry


def test_builtin_evaluators_registered():
    """RegexFilter and LLMJudge should be auto-registered."""
    import dataforge.evaluators  # noqa: F401

    assert "regex-filter" in _evaluator_registry
    assert "llm-judge" in _evaluator_registry
    assert "length-filter" in _evaluator_registry


def test_get_strategy_returns_class():
    import dataforge.strategies  # noqa: F401

    cls = get_strategy("evol-instruct")
    from dataforge.strategies.evol_instruct import EvolInstruct

    assert cls is EvolInstruct


def test_get_evaluator_returns_class():
    import dataforge.evaluators  # noqa: F401

    cls = get_evaluator("regex-filter")
    from dataforge.evaluators.regex_filter import RegexFilter

    assert cls is RegexFilter


def test_get_strategy_unknown_raises():
    with pytest.raises(KeyError, match="Unknown strategy"):
        get_strategy("nonexistent-strategy")


def test_get_evaluator_unknown_raises():
    with pytest.raises(KeyError, match="Unknown evaluator"):
        get_evaluator("nonexistent-evaluator")


def test_list_strategies():
    import dataforge.strategies  # noqa: F401

    result = list_strategies()
    assert "evol-instruct" in result
    assert isinstance(result, dict)


def test_list_evaluators():
    import dataforge.evaluators  # noqa: F401

    result = list_evaluators()
    assert "regex-filter" in result
    assert "llm-judge" in result
    assert "length-filter" in result


def test_register_custom_strategy():
    @register_strategy("test-custom-strategy")
    class CustomStrategy(BaseStrategy):
        async def apply(self, record: DataRecord) -> DataRecord:
            return record

    assert get_strategy("test-custom-strategy") is CustomStrategy
    # Clean up
    _strategy_registry.pop("test-custom-strategy", None)


def test_register_custom_evaluator():
    @register_evaluator("test-custom-evaluator")
    class CustomEvaluator(BaseEvaluator):
        async def evaluate(self, record: DataRecord) -> bool:
            return True

    assert get_evaluator("test-custom-evaluator") is CustomEvaluator
    # Clean up
    _evaluator_registry.pop("test-custom-evaluator", None)


def test_overwrite_warns(caplog):
    """Registering the same name twice should log a warning."""
    import logging

    @register_strategy("test-overwrite")
    class First(BaseStrategy):
        async def apply(self, record: DataRecord) -> DataRecord:
            return record

    with caplog.at_level(logging.WARNING):
        @register_strategy("test-overwrite")
        class Second(BaseStrategy):
            async def apply(self, record: DataRecord) -> DataRecord:
                return record

    assert "Overwriting strategy" in caplog.text
    assert get_strategy("test-overwrite") is Second
    # Clean up
    _strategy_registry.pop("test-overwrite", None)

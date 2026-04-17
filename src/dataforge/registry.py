# src/dataforge/registry.py
"""Plugin registry for strategies and evaluators.

Allows dynamic registration via decorators::

    from dataforge.registry import register_strategy, register_evaluator

    @register_strategy("my-strategy")
    class MyStrategy(BaseStrategy):
        async def apply(self, record: DataRecord) -> DataRecord: ...

    @register_evaluator("my-evaluator")
    class MyEvaluator(BaseEvaluator):
        async def evaluate(self, record: DataRecord) -> bool: ...

Registered plugins are automatically available in YAML configs.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_strategy_registry: dict[str, type] = {}
_evaluator_registry: dict[str, type] = {}
_assessment_suite_registry: dict[str, object] = {}
_benchmark_registry: dict[str, object] = {}


def register_strategy(name: str):
    """Class decorator to register a strategy under *name*."""

    def wrapper(cls: type) -> type:
        if name in _strategy_registry:
            logger.warning(
                "Overwriting strategy %r: %s -> %s",
                name,
                _strategy_registry[name].__qualname__,
                cls.__qualname__,
            )
        _strategy_registry[name] = cls
        return cls

    return wrapper


def register_evaluator(name: str):
    """Class decorator to register an evaluator under *name*."""

    def wrapper(cls: type) -> type:
        if name in _evaluator_registry:
            logger.warning(
                "Overwriting evaluator %r: %s -> %s",
                name,
                _evaluator_registry[name].__qualname__,
                cls.__qualname__,
            )
        _evaluator_registry[name] = cls
        return cls

    return wrapper


def get_strategy(name: str) -> type:
    """Look up a registered strategy by name. Raises KeyError if not found."""
    try:
        return _strategy_registry[name]
    except KeyError:
        available = ", ".join(sorted(_strategy_registry)) or "(none)"
        raise KeyError(
            f"Unknown strategy {name!r}. Available: {available}"
        ) from None


def get_evaluator(name: str) -> type:
    """Look up a registered evaluator by name. Raises KeyError if not found."""
    try:
        return _evaluator_registry[name]
    except KeyError:
        available = ", ".join(sorted(_evaluator_registry)) or "(none)"
        raise KeyError(
            f"Unknown evaluator {name!r}. Available: {available}"
        ) from None


def list_strategies() -> dict[str, type]:
    """Return a copy of the strategy registry."""
    return dict(_strategy_registry)


def list_evaluators() -> dict[str, type]:
    """Return a copy of the evaluator registry."""
    return dict(_evaluator_registry)


def register_assessment_suite(name: str):
    """Register an assessment suite builder or instance under *name*."""

    def wrapper(obj: object) -> object:
        if name in _assessment_suite_registry:
            logger.warning("Overwriting assessment suite %r", name)
        _assessment_suite_registry[name] = obj
        return obj

    return wrapper


def get_assessment_suite(name: str) -> object:
    try:
        return _assessment_suite_registry[name]
    except KeyError:
        available = ", ".join(sorted(_assessment_suite_registry)) or "(none)"
        raise KeyError(
            f"Unknown assessment suite {name!r}. Available: {available}"
        ) from None


def list_assessment_suites() -> dict[str, object]:
    return dict(_assessment_suite_registry)


def register_benchmark(name: str):
    """Register a benchmark task builder or instance under *name*."""

    def wrapper(obj: object) -> object:
        if name in _benchmark_registry:
            logger.warning("Overwriting benchmark %r", name)
        _benchmark_registry[name] = obj
        return obj

    return wrapper


def get_benchmark(name: str) -> object:
    try:
        return _benchmark_registry[name]
    except KeyError:
        available = ", ".join(sorted(_benchmark_registry)) or "(none)"
        raise KeyError(
            f"Unknown benchmark {name!r}. Available: {available}"
        ) from None


def list_benchmarks() -> dict[str, object]:
    return dict(_benchmark_registry)


def _load_entry_points() -> None:
    """Discover third-party plugins via ``dataforge.strategies`` and
    ``dataforge.evaluators`` entry-point groups.

    Each entry point should point to a module that registers its classes
    by importing from ``dataforge.registry`` at module scope.
    """
    try:
        from importlib.metadata import entry_points
    except ImportError:
        return

    for group in ("dataforge.strategies", "dataforge.evaluators"):
        eps = entry_points().get(group, []) if callable(getattr(entry_points(), "get", None)) else entry_points(group=group)
        for ep in eps:
            try:
                ep.load()
                logger.debug("Loaded plugin entry point: %s (%s)", ep.name, group)
            except Exception:
                logger.warning("Failed to load plugin entry point: %s (%s)", ep.name, group, exc_info=True)

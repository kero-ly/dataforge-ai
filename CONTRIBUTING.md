# Contributing to DataForge

Thank you for your interest in contributing! This guide covers setup, workflow, and standards.

## Development Setup

```bash
# Clone and install with dev dependencies
git clone https://github.com/kero-ly/dataforge.git
cd dataforge
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests (skip slow rate-limit exhaustion test)
pytest tests/ -k "not rpm_exhaustion"

# Run a single file
pytest tests/test_pipeline.py -v

# Run with coverage (minimum 70% required)
pytest tests/ --cov=src/dataforge --cov-report=html
```

## Code Quality

```bash
# Lint
ruff check .

# Type check
mypy src/dataforge/

# Auto-fix lint issues
ruff check . --fix
```

All PRs must pass `ruff check` and `mypy` with zero errors.

## Branch Workflow

1. Fork the repo and create a branch from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```
2. Make your changes with focused commits.
3. Push and open a Pull Request against `main`.

## Adding a Custom Strategy

Subclass `BaseStrategy` and implement `async def apply(record: DataRecord) -> DataRecord`:

```python
from dataforge import DataRecord
from dataforge.strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    async def apply(self, record: DataRecord) -> DataRecord:
        # generate synthetic_data from record.seed_data
        record.synthetic_data = {"output": "..."}
        return record
```

## Adding a Custom Evaluator

Subclass `BaseEvaluator` and implement `async def evaluate(record: DataRecord) -> bool`:

```python
from dataforge import DataRecord
from dataforge.evaluators.base import BaseEvaluator

class MyEvaluator(BaseEvaluator):
    async def evaluate(self, record: DataRecord) -> bool:
        # return True to keep, False to reject
        return len(record.synthetic_data.get("output", "")) > 50
```

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `refactor:` code change without feature/fix
- `test:` adding or updating tests
- `chore:` build, CI, or tooling changes

## Pull Request Checklist

- [ ] Tests added or updated for changed behavior
- [ ] `pytest tests/ -k "not rpm_exhaustion"` passes
- [ ] `ruff check .` passes
- [ ] `mypy src/dataforge/` passes
- [ ] PR description explains *why*, not just *what*

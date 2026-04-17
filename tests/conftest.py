# tests/conftest.py
# pytest-asyncio auto mode is set in pyproject.toml
# Add shared fixtures here as the project grows
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
for candidate in (str(_SRC), str(_REPO_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

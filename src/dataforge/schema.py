# src/dataforge/schema.py
from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RecordStatus(str, Enum):
    PENDING = "PENDING"
    GENERATED = "GENERATED"
    EVALUATING = "EVALUATING"
    REJECTED = "REJECTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class DataRecord(BaseModel):
    """Data unit flowing through the pipeline, carrying seed input and synthetic output."""

    model_config = ConfigDict(validate_assignment=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    seed_data: dict[str, Any]
    synthetic_data: dict[str, Any] | None = None
    score: float | None = None
    status: RecordStatus = RecordStatus.PENDING
    metadata: dict[str, Any] = Field(default_factory=dict)

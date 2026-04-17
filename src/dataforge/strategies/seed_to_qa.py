# src/dataforge/strategies/seed_to_qa.py
from __future__ import annotations

import json
import logging
import re

from dataforge.clients.base import LLMProtocol
from dataforge.registry import register_strategy
from dataforge.schema import DataRecord
from dataforge.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

_QA_PROMPT = (
    "Read the following passage carefully, then generate {n} question-answer pairs "
    "based on its content.\n\n"
    "Passage:\n{passage}\n\n"
    "Difficulty level(s): {levels}\n\n"
    "Return a JSON array of objects, each with \"question\", \"answer\", and "
    "\"difficulty\" keys. Output ONLY the JSON array, no explanation.\n\n"
    "Example format:\n"
    '[{{"question": "...", "answer": "...", "difficulty": "easy"}}]'
)

_REPAIR_PROMPT = (
    "The following text was supposed to be a valid JSON array of question-answer "
    "pairs but failed to parse:\n---\n{text}\n---\n"
    "Parse error: {error}\n\n"
    "Return ONLY the corrected JSON array. Do not add any explanation."
)


@register_strategy("seed-to-qa")
class SeedToQA(BaseStrategy):
    """Generate question-answer pairs from text passages.

    Takes a passage from seed_data and generates structured QA pairs using an LLM.
    Suitable for building QA datasets from domain documents (papers, manuals, etc.).

    Args:
        llm: Any object with ``async def generate(prompt: str) -> str``.
        qa_per_passage: Number of QA pairs to generate per passage. Default 3.
        difficulty_levels: List of difficulty levels. Default ["easy", "medium", "hard"].
        source_field: Field name in seed_data containing the passage. Default "passage".
        max_repair_attempts: Max JSON repair attempts on parse failure. Default 2.
    """

    def __init__(
        self,
        llm: LLMProtocol,
        qa_per_passage: int = 3,
        difficulty_levels: list[str] | None = None,
        source_field: str = "passage",
        max_repair_attempts: int = 2,
    ) -> None:
        if qa_per_passage < 1:
            raise ValueError(f"qa_per_passage must be >= 1, got {qa_per_passage}")
        self.llm = llm
        self.qa_per_passage = qa_per_passage
        self.difficulty_levels = difficulty_levels or ["easy", "medium", "hard"]
        self.source_field = source_field
        self.max_repair_attempts = max_repair_attempts

    async def apply(self, record: DataRecord) -> DataRecord:
        passage = record.seed_data.get(self.source_field, "")
        if not passage:
            raise ValueError(
                f"seed_data missing required field {self.source_field!r}"
            )

        prompt = _QA_PROMPT.format(
            n=self.qa_per_passage,
            passage=passage,
            levels=", ".join(self.difficulty_levels),
        )
        raw = (await self.llm.generate(prompt)).strip()
        qa_pairs = await self._parse_qa_json(raw)

        record.synthetic_data = {
            "passage": passage,
            "qa_pairs": qa_pairs,
        }
        logger.debug(
            "Generated %d QA pairs for record %s",
            len(qa_pairs),
            record.id,
        )
        return record

    async def _parse_qa_json(self, text: str) -> list[dict]:
        """Parse QA JSON array with self-repair on failure."""
        for attempt in range(self.max_repair_attempts + 1):
            try:
                extracted = self._extract_json_array(text)
                result = json.loads(extracted)
                if not isinstance(result, list):
                    raise ValueError("Expected a JSON array, got " + type(result).__name__)
                return result
            except (ValueError, json.JSONDecodeError) as e:
                if attempt < self.max_repair_attempts:
                    logger.warning(
                        "QA JSON parse failed (attempt %d/%d): %s",
                        attempt + 1,
                        self.max_repair_attempts + 1,
                        e,
                    )
                    repair_prompt = _REPAIR_PROMPT.format(text=text, error=e)
                    text = (await self.llm.generate(repair_prompt)).strip()

        raise ValueError(
            f"QA JSON self-repair exhausted after {self.max_repair_attempts + 1} attempts"
        )

    @staticmethod
    def _extract_json_array(text: str) -> str:
        """Extract a JSON array from text, stripping markdown fences."""
        stripped = text.strip()
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped)
        if fence_match:
            return fence_match.group(1).strip()
        if stripped.startswith("["):
            return stripped
        arr_match = re.search(r"\[[\s\S]*\]", stripped)
        if arr_match:
            return arr_match.group()
        return stripped

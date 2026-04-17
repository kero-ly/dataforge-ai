# src/dataforge/strategies/paraphrase.py
from __future__ import annotations

import logging

from dataforge.clients.base import LLMProtocol
from dataforge.registry import register_strategy
from dataforge.schema import DataRecord
from dataforge.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

_PARAPHRASE_PROMPT = (
    "Rewrite the following text to be semantically equivalent but use different "
    "vocabulary, sentence structure, and phrasing. Preserve all key information "
    "and meaning.\n\n"
    "Original:\n{text}\n\n"
    "Rewritten version (output ONLY the rewritten text, no explanation):"
)

_MULTI_VARIANT_PROMPT = (
    "Generate {n} distinct paraphrased versions of the following text. Each version "
    "should be semantically equivalent but use different vocabulary, sentence structure, "
    "and phrasing. Preserve all key information and meaning.\n\n"
    "Original:\n{text}\n\n"
    "Output each version on a separate line, prefixed with its number (e.g., '1. ...').\n"
    "Do not add any other explanation."
)


@register_strategy("paraphrase")
class Paraphrase(BaseStrategy):
    """Paraphrase / rewrite strategy for data augmentation.

    Generates semantically equivalent but lexically different variants of input text.
    Useful for training data augmentation and diversity improvement.

    Args:
        llm: Any object with ``async def generate(prompt: str) -> str``.
        n_variants: Number of paraphrased variants to generate. Default 1.
        source_field: Field name in seed_data to paraphrase. Default "instruction".
    """

    def __init__(
        self,
        llm: LLMProtocol,
        n_variants: int = 1,
        source_field: str = "instruction",
    ) -> None:
        if n_variants < 1:
            raise ValueError(f"n_variants must be >= 1, got {n_variants}")
        self.llm = llm
        self.n_variants = n_variants
        self.source_field = source_field

    async def apply(self, record: DataRecord) -> DataRecord:
        text = record.seed_data.get(self.source_field, "")
        if not text:
            raise ValueError(
                f"seed_data missing required field {self.source_field!r}"
            )

        if self.n_variants == 1:
            prompt = _PARAPHRASE_PROMPT.format(text=text)
            paraphrased = (await self.llm.generate(prompt)).strip()
            record.synthetic_data = {
                self.source_field: text,
                "paraphrase": paraphrased,
            }
        else:
            prompt = _MULTI_VARIANT_PROMPT.format(n=self.n_variants, text=text)
            raw = (await self.llm.generate(prompt)).strip()
            variants = self._parse_variants(raw, self.n_variants)
            record.synthetic_data = {
                self.source_field: text,
                "variants": variants,
            }

        logger.debug(
            "Paraphrased record %s (%d variant(s))",
            record.id,
            self.n_variants,
        )
        return record

    @staticmethod
    def _parse_variants(text: str, expected: int) -> list[str]:
        """Parse numbered lines like '1. ...' from LLM output."""
        lines = text.strip().splitlines()
        variants: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Strip leading number prefix: "1. ", "2) ", etc.
            for i in range(1, expected + 2):
                for sep in (". ", ") ", ": "):
                    prefix = f"{i}{sep}"
                    if stripped.startswith(prefix):
                        stripped = stripped[len(prefix):]
                        break
            variants.append(stripped)
        return variants[:expected] if len(variants) >= expected else variants

# src/dataforge/strategies/evol_instruct.py
from __future__ import annotations

import json
import logging
import random
import re

from dataforge.clients.base import LLMProtocol
from dataforge.registry import register_strategy
from dataforge.schema import DataRecord
from dataforge.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


@register_strategy("evol-instruct")
class EvolInstruct(BaseStrategy):
    """Instruction evolution strategy (WizardLM Evol-Instruct).

    Evolves a seed instruction over ``depth`` rounds. Each round picks a random
    mutation operator and calls the LLM. Optionally generates a CoT response or
    enforces valid JSON output with self-repair.

    Args:
        llm: Any object with ``async def generate(prompt: str) -> str``.
        depth: Number of evolution rounds. Default 3.
        mutation_types: Subset of ``VALID_MUTATION_TYPES``. Defaults to all.
        require_reasoning: After evolution, generate a ``<think>`` CoT response.
        require_json: Force JSON output with regex strip + LLM self-repair fallback.
    """

    VALID_MUTATION_TYPES = ("constraints", "deepen", "concretize")

    _MUTATION_PROMPTS: dict[str, str] = {
        "constraints": (
            "Rewrite the following instruction to add 2–3 additional hard constraints "
            "that make it significantly more challenging:\n\n{instruction}\n\n"
            "Output only the rewritten instruction."
        ),
        "deepen": (
            "Rewrite the following instruction so that answering it requires deeper "
            "domain knowledge and multi-step reasoning:\n\n{instruction}\n\n"
            "Output only the rewritten instruction."
        ),
        "concretize": (
            "Rewrite the following abstract instruction with concrete real-world scenarios, "
            "specific numbers, and tangible examples:\n\n{instruction}\n\n"
            "Output only the rewritten instruction."
        ),
    }

    _COT_SUFFIX = (
        "\n\nBefore answering, think step by step inside <think>...</think> tags, "
        "then give your final answer after the closing tag."
    )

    _JSON_SUFFIX = "\n\nRespond with valid JSON only. Do not add any explanation."

    _REPAIR_TEMPLATE = (
        "The following text was supposed to be valid JSON but failed to parse:\n"
        "---\n{text}\n---\n"
        "Parse error: {error}\n\n"
        "Return ONLY the corrected JSON. Do not add any explanation."
    )

    from typing import Any

    _SYSTEM_PROMPTS: dict[str, str] = {
        "constraints": (
            "You rewrite instructions by adding 2-3 additional hard constraints "
            "that make them significantly more challenging. "
            "Output only the rewritten instruction."
        ),
        "deepen": (
            "You rewrite instructions so that answering them requires deeper "
            "domain knowledge and multi-step reasoning. "
            "Output only the rewritten instruction."
        ),
        "concretize": (
            "You rewrite abstract instructions with concrete real-world scenarios, "
            "specific numbers, and tangible examples. "
            "Output only the rewritten instruction."
        ),
    }

    _VALID_SCHEDULES = ("random", "round_robin", "batch")

    def __init__(
        self,
        llm: LLMProtocol,
        depth: int = 3,
        mutation_types: list[str] | None = None,
        require_reasoning: bool = False,
        require_json: bool = False,
        mutation_schedule: str = "random",
        batch_size: int = 50,
        use_system_prompt: bool = False,
    ) -> None:
        if mutation_types is None:
            mutation_types = list(self.VALID_MUTATION_TYPES)
        invalid = [m for m in mutation_types if m not in self.VALID_MUTATION_TYPES]
        if invalid:
            raise ValueError(
                f"Invalid mutation_types: {invalid}. "
                f"Valid options: {list(self.VALID_MUTATION_TYPES)}"
            )
        if mutation_schedule not in self._VALID_SCHEDULES:
            raise ValueError(
                f"Invalid mutation_schedule: {mutation_schedule!r}. "
                f"Valid options: {list(self._VALID_SCHEDULES)}"
            )
        self.llm = llm
        self.depth = depth
        self.mutation_types = mutation_types
        self.require_reasoning = require_reasoning
        self.require_json = require_json
        self.mutation_schedule = mutation_schedule
        self.batch_size = batch_size
        self.use_system_prompt = use_system_prompt
        self._counter: int = 0

    def _next_mutation(self) -> str:
        """Return the next mutation type based on the configured schedule."""
        if self.mutation_schedule == "random":
            return random.choice(self.mutation_types)
        if self.mutation_schedule == "round_robin":
            mutation = self.mutation_types[self._counter % len(self.mutation_types)]
            self._counter += 1
            return mutation
        # batch
        mutation = self.mutation_types[
            (self._counter // self.batch_size) % len(self.mutation_types)
        ]
        self._counter += 1
        return mutation

    def build_prompts(
        self, seeds: list[dict[str, Any]],
    ) -> list[tuple[str, list[dict[str, str]]]]:
        """Pre-compute all prompts for depth=1 in a tight loop.

        Returns a list of ``(mutation_type, messages)`` tuples where *messages*
        is a chat-format list ready for ``generate_raw()``.

        When ``use_system_prompt`` is True, each message list contains a system
        message (shared across all seeds of the same mutation type, enabling
        vLLM prefix caching) and a user message with only the instruction.

        When False, the full prompt is in a single user message.
        """
        result: list[tuple[str, list[dict[str, str]]]] = []
        templates = self._MUTATION_PROMPTS
        sys_prompts = self._SYSTEM_PROMPTS
        use_sys = self.use_system_prompt

        for seed in seeds:
            instruction = str(seed["instruction"])
            mutation = self._next_mutation()
            if use_sys:
                messages = [
                    {"role": "system", "content": sys_prompts[mutation]},
                    {"role": "user", "content": instruction},
                ]
            else:
                prompt = templates[mutation].format(instruction=instruction)
                messages = [{"role": "user", "content": prompt}]
            result.append((mutation, messages))
        return result

    @property
    def supports_build_prompts(self) -> bool:
        """Whether build_prompts() can be used (depth=1, no special output)."""
        return self.depth == 1 and not self.require_reasoning and not self.require_json

    async def apply_seed_data(self, seed_data: dict[str, object]) -> dict[str, object]:
        instruction = str(seed_data["instruction"])
        llm_generate = self.llm.generate
        prompt_templates = self._MUTATION_PROMPTS
        sys_prompts = self._SYSTEM_PROMPTS
        use_sys = self.use_system_prompt
        next_mutation = self._next_mutation

        for i in range(self.depth):
            mutation = next_mutation()
            if use_sys:
                prompt: str | list[dict] = [
                    {"role": "system", "content": sys_prompts[mutation]},
                    {"role": "user", "content": instruction},
                ]
            else:
                prompt = prompt_templates[mutation].format(instruction=instruction)
            instruction = (await llm_generate(prompt)).strip()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Evolution round %d/%d: mutation=%s", i + 1, self.depth, mutation)

        if self.require_reasoning:
            cot_prompt = instruction + self._COT_SUFFIX
            raw = await llm_generate(cot_prompt)
            think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
            reasoning = think_match.group(1).strip() if think_match else ""
            answer = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            return {
                "instruction": instruction,
                "response": answer,
                "reasoning": reasoning,
            }
        if self.require_json:
            return await self._generate_json(instruction)
        return {"instruction": instruction}

    async def apply(self, record: DataRecord) -> DataRecord:
        object.__setattr__(record, "synthetic_data", await self.apply_seed_data(record.seed_data))
        return record

    async def _generate_json(self, instruction: str, max_repair_attempts: int = 2) -> dict:
        """Generate JSON output with self-repair on parse failure."""
        prompt = instruction + self._JSON_SUFFIX
        text = await self.llm.generate(prompt)

        for attempt in range(max_repair_attempts + 1):
            try:
                return json.loads(self._extract_json(text))
            except (ValueError, json.JSONDecodeError) as e:
                if attempt < max_repair_attempts:
                    logger.warning("JSON parse failed (attempt %d/%d): %s", attempt + 1, max_repair_attempts + 1, e)
                    repair_prompt = self._REPAIR_TEMPLATE.format(text=text, error=e)
                    text = await self.llm.generate(repair_prompt)

        raise ValueError(
            f"JSON self-repair exhausted after {max_repair_attempts + 1} attempts"
        )

    @staticmethod
    def _extract_json(text: str) -> str:
        """Strip markdown fences and extract a JSON block from text."""
        stripped = text.strip()
        # Step 1: strip markdown code fences (free, no LLM call)
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped)
        if fence_match:
            return fence_match.group(1).strip()
        # Step 2: try to find a bare JSON object or array
        if stripped.startswith(("{", "[")):
            return stripped
        obj_match = re.search(r"\{[\s\S]*\}", stripped)
        if obj_match:
            return obj_match.group()
        return stripped

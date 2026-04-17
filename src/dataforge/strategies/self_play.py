# src/dataforge/strategies/self_play.py
"""Self-Play strategy: two LLM agents generate multi-turn dialogue data."""
from __future__ import annotations

import logging

from dataforge.clients.base import LLMProtocol
from dataforge.registry import register_strategy
from dataforge.schema import DataRecord
from dataforge.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

_SYSTEM_TEMPLATE = """\
You are playing the role of "{role}" in a conversation about the following topic:

{topic}

Respond naturally in character. Keep your response focused and concise.
"""

_OPENER_TEMPLATE = """\
You are "{role}". Start a conversation about the following topic. \
Ask a question or make an opening statement.

Topic: {topic}
"""


@register_strategy("self-play")
class SelfPlay(BaseStrategy):
    """Multi-agent adversarial/cooperative dialogue generation.

    Two LLM agents alternate turns to generate multi-turn conversation data.
    Useful for producing debate, interview, or Q&A training datasets.

    Args:
        llm: Primary LLM client (used for role_a, and role_b if llm_b is None).
        llm_b: Optional second LLM for role_b. If None, ``llm`` is used for both.
        turns: Number of *exchange rounds* (each round = one message from each role).
        role_a: Display name for agent A.
        role_b: Display name for agent B.
        system_prompt_a: Custom system prompt for agent A (overrides default).
        system_prompt_b: Custom system prompt for agent B (overrides default).
        source_field: Seed data field containing the conversation topic.
    """

    def __init__(
        self,
        llm: LLMProtocol,
        llm_b: LLMProtocol | None = None,
        turns: int = 3,
        role_a: str = "Interviewer",
        role_b: str = "Interviewee",
        system_prompt_a: str | None = None,
        system_prompt_b: str | None = None,
        source_field: str = "topic",
    ) -> None:
        if turns < 1:
            raise ValueError("turns must be >= 1")
        self.llm_a = llm
        self.llm_b = llm_b or llm
        self.turns = turns
        self.role_a = role_a
        self.role_b = role_b
        self.system_prompt_a = system_prompt_a
        self.system_prompt_b = system_prompt_b
        self.source_field = source_field

    async def apply(self, record: DataRecord) -> DataRecord:
        topic = record.seed_data.get(self.source_field)
        if topic is None:
            raise ValueError(
                f"Seed data missing required field '{self.source_field}'"
            )

        conversation: list[dict[str, str]] = []

        # Build system prompts
        sys_a = self.system_prompt_a or _SYSTEM_TEMPLATE.format(
            role=self.role_a, topic=topic
        )
        sys_b = self.system_prompt_b or _SYSTEM_TEMPLATE.format(
            role=self.role_b, topic=topic
        )

        # Role A opens the conversation
        opener_prompt = _OPENER_TEMPLATE.format(role=self.role_a, topic=topic)
        response_a = await self.llm_a.generate(opener_prompt)
        conversation.append({"role": self.role_a, "content": response_a.strip()})

        for turn in range(self.turns):
            # Role B responds to what Role A said
            messages_b = self._build_messages(sys_b, conversation, self.role_b)
            response_b = await self.llm_b.generate(messages_b)
            conversation.append({"role": self.role_b, "content": response_b.strip()})

            # Role A responds (skip on last turn to avoid dangling message)
            if turn < self.turns - 1:
                messages_a = self._build_messages(sys_a, conversation, self.role_a)
                response_a = await self.llm_a.generate(messages_a)
                conversation.append({"role": self.role_a, "content": response_a.strip()})

        record.synthetic_data = {
            "topic": topic,
            "role_a": self.role_a,
            "role_b": self.role_b,
            "conversation": conversation,
            "num_turns": len(conversation),
        }
        return record

    @staticmethod
    def _build_messages(
        system_prompt: str,
        conversation: list[dict[str, str]],
        current_role: str,
    ) -> list[dict[str, str]]:
        """Build a chat message list from the conversation history."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        for msg in conversation:
            # Map conversation roles to chat API roles
            chat_role = "assistant" if msg["role"] == current_role else "user"
            messages.append({"role": chat_role, "content": msg["content"]})
        return messages

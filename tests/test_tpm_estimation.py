# tests/test_tpm_estimation.py
"""Tests for TPM token estimation in OpenAIClient."""

from dataforge.clients.openai_client import OpenAIClient


def test_estimate_tokens_string():
    # 100 chars → ~20 tokens (5 chars per token)
    prompt = "a" * 100
    assert OpenAIClient._estimate_tokens(prompt) == 20


def test_estimate_tokens_short_string():
    # 3 chars → max(1, 0) = 1
    assert OpenAIClient._estimate_tokens("hi!") == 1


def test_estimate_tokens_empty_string():
    # empty → max(1, 0) = 1
    assert OpenAIClient._estimate_tokens("") == 1


def test_estimate_tokens_message_list():
    messages = [
        {"role": "system", "content": "You are helpful."},  # 16 chars
        {"role": "user", "content": "Hello world!"},         # 12 chars
    ]
    # total = 28 chars → 5 tokens (28 // 5)
    result = OpenAIClient._estimate_tokens(messages)
    assert result == 5


def test_estimate_tokens_empty_messages():
    messages = [{"role": "user", "content": ""}]
    assert OpenAIClient._estimate_tokens(messages) == 1


def test_estimate_tokens_missing_content():
    messages = [{"role": "user"}]
    assert OpenAIClient._estimate_tokens(messages) == 1

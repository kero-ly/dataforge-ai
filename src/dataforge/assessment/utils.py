from __future__ import annotations

import hashlib
import math
import re
from collections import Counter


def estimate_tokens(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    return max(1, len(text) // 4)


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def normalize_text(text: str) -> str:
    tokens = tokenize_text(text)
    return " ".join(tokens)


def distinct_n(texts: list[str], n: int) -> float:
    total = 0
    unique: set[tuple[str, ...]] = set()
    for text in texts:
        tokens = tokenize_text(text)
        if len(tokens) < n:
            continue
        grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        total += len(grams)
        unique.update(grams)
    if total == 0:
        return 0.0
    return len(unique) / total


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = max(0, min(len(ordered) - 1, int(math.ceil(q * len(ordered)) - 1)))
    return ordered[idx]


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def five_gram_jaccard(a: str, b: str) -> float:
    a_tokens = tokenize_text(a)
    b_tokens = tokenize_text(b)
    if len(a_tokens) < 5 or len(b_tokens) < 5:
        return 1.0 if normalize_text(a) == normalize_text(b) else 0.0
    a_grams = {tuple(a_tokens[i : i + 5]) for i in range(len(a_tokens) - 4)}
    b_grams = {tuple(b_tokens[i : i + 5]) for i in range(len(b_tokens) - 4)}
    union = a_grams | b_grams
    if not union:
        return 0.0
    return len(a_grams & b_grams) / len(union)


def count_duplicates(texts: list[str]) -> tuple[int, dict[str, int]]:
    normalized = [normalize_text(text) for text in texts if normalize_text(text)]
    counts = Counter(normalized)
    duplicate_count = sum(count - 1 for count in counts.values() if count > 1)
    return duplicate_count, dict(counts)

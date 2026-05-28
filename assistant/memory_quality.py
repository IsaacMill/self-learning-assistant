"""Memory quality scoring, duplicate detection, and contradiction helpers."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from assistant.schemas import MemoryItem, TrustLevel

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def normalize_memory_text(text: str) -> str:
    """Normalize memory text for duplicate checks."""

    return " ".join(TOKEN_RE.findall(text.lower()))


def token_set(text: str) -> set[str]:
    """Return normalized tokens."""

    return set(TOKEN_RE.findall(text.lower()))


def similarity(left: str, right: str) -> float:
    """Return Jaccard token similarity."""

    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def recency_score(timestamp: datetime, half_life_days: float = 30.0) -> float:
    """Score memory recency with exponential decay."""

    now = datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (now - timestamp).total_seconds() / 86400)
    return max(0.0, min(1.0, math.exp(-age_days / half_life_days)))


def usefulness_score(item: MemoryItem) -> float:
    """Estimate whether a memory is likely to be useful later."""

    content_length = len(item.content.strip())
    tag_bonus = min(len(item.tags) * 0.05, 0.2)
    kind_bonus = 0.15 if item.kind.value in {"semantic", "failure"} else 0.0
    length_score = min(content_length / 120, 1.0)
    score = 0.2 + (0.45 * length_score) + tag_bonus + kind_bonus
    return max(0.0, min(1.0, score))


def confidence_score(item: MemoryItem) -> float:
    """Estimate memory confidence from explicit confidence and trust."""

    trust_bonus = {
        TrustLevel.trusted: 0.2,
        TrustLevel.proposed: 0.05,
        TrustLevel.raw: -0.1,
        TrustLevel.rejected: -0.4,
    }[item.trust]
    return max(0.0, min(1.0, item.confidence + trust_bonus))


def quality_score(item: MemoryItem) -> float:
    """Combined score used for cleanup and ranking."""

    return (
        0.4 * item.confidence_score
        + 0.35 * item.usefulness_score
        + 0.2 * item.recency_score
        + 0.05 * min(len(item.tags) / 5, 1.0)
    )


def apply_quality_scores(item: MemoryItem) -> MemoryItem:
    """Populate quality scores on a memory item."""

    item.usefulness_score = usefulness_score(item)
    item.confidence_score = confidence_score(item)
    item.recency_score = recency_score(item.timestamp)
    return item


def is_duplicate(left: MemoryItem, right: MemoryItem, threshold: float = 0.86) -> bool:
    """Return True when two memories are duplicates."""

    return normalize_memory_text(left.content) == normalize_memory_text(right.content) or (
        similarity(left.content, right.content) >= threshold
    )


def looks_contradictory(left: MemoryItem, right: MemoryItem) -> bool:
    """Flag simple natural-language contradictions."""

    left_text = normalize_memory_text(left.content)
    right_text = normalize_memory_text(right.content)
    if not left_text or not right_text:
        return False
    pairs = [
        (" prefers ", " does not prefer "),
        (" likes ", " does not like "),
        (" is ", " is not "),
        (" should ", " should not "),
    ]
    for positive, negative in pairs:
        if positive.strip() in left_text and negative.strip() in right_text:
            return similarity(left_text.replace(positive.strip(), ""), right_text.replace(negative.strip(), "")) > 0.45
        if negative.strip() in left_text and positive.strip() in right_text:
            return similarity(left_text.replace(negative.strip(), ""), right_text.replace(positive.strip(), "")) > 0.45
    return False

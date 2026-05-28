"""Post-interaction reflection and memory proposal logic."""

from __future__ import annotations

import re

from assistant.judge import memory_write_judge
from assistant.safety import redact_secrets
from assistant.schemas import MemoryItem, MemoryKind, ReflectionResult, TrustLevel

PREFERENCE_DECLARATION_RE = re.compile(r"(?i)\b(i prefer|my preference is|my planning style is)\b")


def reflect(
    user_message: str,
    assistant_answer: str,
    allow_sensitive_memory: bool = False,
) -> ReflectionResult:
    """Summarize an interaction and propose safe memory updates."""

    safe_user_message = redact_secrets(user_message)
    safe_assistant_answer = redact_secrets(assistant_answer)
    summary = f"User asked: {safe_user_message[:160]}. Assistant answered: {safe_assistant_answer[:160]}."
    learned: list[str] = []
    possible_mistakes: list[str] = []

    is_question = "?" in safe_user_message
    if "remember that" in safe_user_message.lower():
        lesson = safe_user_message.lower().split("remember that", 1)[1].strip(" .")
        learned.append(lesson)
    elif not is_question and PREFERENCE_DECLARATION_RE.search(safe_user_message):
        learned.append(safe_user_message.strip())

    if "I am running in local fallback mode" in safe_assistant_answer:
        possible_mistakes.append("Answer quality is limited because no LLM API key was configured.")

    proposed: list[MemoryItem] = []
    for lesson in learned:
        item = MemoryItem(
            kind=MemoryKind.semantic,
            content=lesson,
            source="reflection",
            confidence=0.7,
            tags=["lesson", "reflection"],
            trust=TrustLevel.proposed,
        )
        if memory_write_judge(item, allow_sensitive=allow_sensitive_memory).passed:
            proposed.append(item)

    return ReflectionResult(
        summary=summary,
        learned=learned,
        possible_mistakes=possible_mistakes,
        proposed_memories=proposed,
    )

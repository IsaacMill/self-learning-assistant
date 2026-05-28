"""Heuristic judges for MVP answer quality and memory writes."""

from __future__ import annotations

from assistant import safety
from assistant.schemas import JudgeResult, MemoryItem, RetrievedMemory


def factuality_judge(answer: str, used_memories: list[RetrievedMemory]) -> JudgeResult:
    """Check whether confident claims are grounded when memory context exists."""

    risky_phrases = ["always", "never", "guaranteed", "certainly"]
    score = 0.8
    explanation = "No obvious unsupported factuality issue detected."
    if any(phrase in answer.lower() for phrase in risky_phrases):
        score -= 0.2
        explanation = "Answer uses absolute language that may need qualification."
    if used_memories and not any(mem.memory.content[:20].lower() in answer.lower() for mem in used_memories):
        score -= 0.1
        explanation = "Answer used memory context but did not clearly cite it."
    return JudgeResult(
        passed=score >= 0.6,
        score=max(score, 0.0),
        explanation=explanation,
        suggested_correction=None if score >= 0.6 else "Qualify claims and ground them in retrieved memories.",
    )


def safety_judge(text: str) -> JudgeResult:
    """Check that the response stays inside the MVP safety boundary."""

    if safety.is_blocked_action_request(text):
        if any(marker in text.lower() for marker in ["cannot", "can't", "do not", "not execute", "refuse"]):
            return JudgeResult(
                passed=True,
                score=0.9,
                explanation="Text mentions a blocked action only to refuse it.",
            )
        return JudgeResult(
            passed=False,
            score=0.1,
            explanation="Text appears to request or endorse blocked autonomous actions.",
            suggested_correction="Refuse execution and explain the current safety boundary.",
        )
    if safety.contains_secret(text):
        return JudgeResult(
            passed=False,
            score=0.0,
            explanation="Text contains a likely secret.",
            suggested_correction="Remove credentials from the response.",
        )
    return JudgeResult(passed=True, score=1.0, explanation="No blocked action or secret detected.")


def usefulness_judge(user_message: str, answer: str) -> JudgeResult:
    """Check whether the answer appears substantive and responsive."""

    if len(answer.strip()) < 20:
        return JudgeResult(
            passed=False,
            score=0.2,
            explanation="Answer is too short to be useful.",
            suggested_correction="Provide a more complete response.",
        )
    overlap = set(user_message.lower().split()) & set(answer.lower().split())
    score = 0.7 + min(len(overlap) / 20, 0.3)
    return JudgeResult(passed=True, score=min(score, 1.0), explanation="Answer appears responsive.")


def consistency_judge(answer: str, used_memories: list[RetrievedMemory]) -> JudgeResult:
    """Check for direct contradiction markers against memory context."""

    if not used_memories:
        return JudgeResult(passed=True, score=0.8, explanation="No memories were used, so consistency risk is low.")
    lowered = answer.lower()
    for retrieved in used_memories:
        memory_text = retrieved.memory.content.lower()
        if "not " in lowered and memory_text.replace("not ", "") in lowered:
            return JudgeResult(
                passed=False,
                score=0.3,
                explanation="Answer may contradict retrieved memory.",
                suggested_correction="Reconcile the answer with the retrieved memory.",
            )
    return JudgeResult(passed=True, score=0.9, explanation="No obvious contradiction with retrieved memories.")


def memory_write_judge(item: MemoryItem, allow_sensitive: bool = False) -> JudgeResult:
    """Approve only durable, safe, non-trivial memory candidates."""

    safety_result = safety.validate_memory_for_storage(item, allow_sensitive=allow_sensitive)
    if not safety_result.passed:
        return safety_result
    if len(item.content.strip()) < 15:
        return JudgeResult(
            passed=False,
            score=0.3,
            explanation="Candidate memory is too short to be durable or useful.",
            suggested_correction="Store a specific lesson with context.",
        )
    if "?" in item.content:
        return JudgeResult(
            passed=False,
            score=0.3,
            explanation="Candidate memory appears to be a question rather than a durable lesson.",
            suggested_correction="Do not store questions as semantic memory.",
        )
    if item.confidence < 0.5:
        return JudgeResult(
            passed=False,
            score=0.4,
            explanation="Candidate memory confidence is too low.",
            suggested_correction="Keep it as an interaction log rather than semantic memory.",
        )
    return JudgeResult(passed=True, score=0.85, explanation="Candidate memory is safe and useful enough to propose.")

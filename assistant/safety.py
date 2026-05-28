"""Safety checks for assistant behavior and memory handling."""

from __future__ import annotations

import re
from pathlib import Path

from assistant.schemas import JudgeResult, MemoryItem

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?i)(api[_ -]?key|secret|password|token)\s*[:=]\s*\S+"),
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{12,}"),
]

SENSITIVE_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b\d{12,19}\b"),
    re.compile(r"(?i)\b(diagnosed with|medical record|ssn|social security)\b"),
]

BLOCKED_ACTION_PATTERNS = [
    re.compile(r"(?i)\b(run|execute)\s+(shell|command|bash|zsh|python)\b"),
    re.compile(r"(?i)\b(run|execute)\s+.*\b(unit tests?|tests?|code)\b"),
    re.compile(r"(?i)\b(delete|remove|unlink)\s+.*\b(file|folder|directory)\b"),
    re.compile(r"(?i)\b(edit|rewrite|modify)\s+.*\b(assistant/|source files?|own code)\b"),
    re.compile(r"(?i)\b(patch|modify|rewrite)\s+.*\b(own runtime|bypass safety|future prompts)\b"),
]


def contains_secret(text: str) -> bool:
    """Return True when text appears to contain credentials."""

    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def redact_secrets(text: str) -> str:
    """Replace likely credentials with a fixed redaction marker."""

    redacted = text
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED_SECRET]", redacted)
    return redacted


def contains_sensitive_personal_info(text: str) -> bool:
    """Return True when text appears to contain sensitive personal data."""

    return any(pattern.search(text) for pattern in SENSITIVE_PATTERNS)


def is_blocked_action_request(text: str) -> bool:
    """Return True for requests outside the MVP safety boundary."""

    return any(pattern.search(text) for pattern in BLOCKED_ACTION_PATTERNS)


def validate_memory_for_storage(item: MemoryItem, allow_sensitive: bool = False) -> JudgeResult:
    """Check whether a memory item may be stored."""

    if item.content != redact_secrets(item.content):
        return JudgeResult(
            passed=False,
            score=0.0,
            explanation="Memory contains a likely secret or credential.",
            suggested_correction="Remove secrets before storing the memory.",
        )
    if contains_secret(item.content):
        return JudgeResult(
            passed=False,
            score=0.0,
            explanation="Memory contains a likely secret or credential.",
            suggested_correction="Remove secrets before storing the memory.",
        )
    if contains_sensitive_personal_info(item.content) and not allow_sensitive:
        return JudgeResult(
            passed=False,
            score=0.2,
            explanation="Memory contains sensitive personal information without explicit approval.",
            suggested_correction="Ask for explicit permission or store a non-sensitive abstraction.",
        )
    return JudgeResult(passed=True, score=1.0, explanation="Memory passed safety checks.")


def assert_not_self_modifying(path: Path) -> None:
    """Raise if a requested edit targets the assistant source package."""

    normalized = path.resolve()
    assistant_root = Path(__file__).resolve().parent
    if assistant_root in normalized.parents or normalized == assistant_root:
        raise PermissionError("Editing assistant source files is blocked at runtime.")

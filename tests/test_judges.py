from assistant.judge import memory_write_judge, safety_judge, usefulness_judge
from assistant.schemas import MemoryItem, MemoryKind, TrustLevel


def test_safety_judge_blocks_secret() -> None:
    result = safety_judge("Here is an api_key=super-secret-value")

    assert not result.passed
    assert result.score == 0.0


def test_usefulness_judge_rejects_tiny_answer() -> None:
    result = usefulness_judge("Explain memory", "ok")

    assert not result.passed


def test_memory_write_judge_requires_nontrivial_safe_memory() -> None:
    item = MemoryItem(
        kind=MemoryKind.semantic,
        content="User prefers brief answers with direct next steps.",
        source="test",
        confidence=0.9,
        tags=["preference"],
        trust=TrustLevel.proposed,
    )

    result = memory_write_judge(item)

    assert result.passed
    assert result.score > 0.8

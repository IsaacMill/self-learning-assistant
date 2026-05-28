from datetime import datetime, timedelta, timezone
from pathlib import Path

from assistant.memory import MemoryStore
from assistant.retrieval import MemoryRetriever
from assistant.schemas import MemoryItem, MemoryKind, TrustLevel


def test_memory_gets_quality_fields(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "test.sqlite3")
    item = MemoryItem(
        kind=MemoryKind.semantic,
        content="User prefers concise answers with clear next steps.",
        source="test",
        source_type="user",
        confidence=0.8,
        tags=["preference"],
        trust=TrustLevel.proposed,
    )

    stored = store.add_memory(item)

    assert stored.usefulness_score > 0
    assert stored.confidence_score > 0
    assert stored.recency_score > 0
    assert stored.source_type == "user"


def test_duplicate_memories_merge(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "test.sqlite3")
    first = MemoryItem(
        kind=MemoryKind.semantic,
        content="User prefers concise answers.",
        source="test",
        confidence=0.7,
        tags=["preference"],
        trust=TrustLevel.proposed,
    )
    duplicate = MemoryItem(
        kind=MemoryKind.semantic,
        content="User prefers concise answers.",
        source="test-2",
        confidence=0.9,
        tags=["format"],
        trust=TrustLevel.proposed,
    )

    stored_first = store.add_memory(first)
    stored_duplicate = store.add_memory(duplicate)

    assert stored_duplicate.id == stored_first.id
    assert duplicate.id in stored_duplicate.merged_from
    assert set(stored_duplicate.tags) == {"preference", "format"}


def test_contradictory_memories_are_flagged(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "test.sqlite3")
    first = store.add_memory(
        MemoryItem(
            kind=MemoryKind.semantic,
            content="User likes concise reports.",
            source="test",
            confidence=0.8,
            tags=["preference"],
            trust=TrustLevel.proposed,
        )
    )
    second = store.add_memory(
        MemoryItem(
            kind=MemoryKind.semantic,
            content="User does not like concise reports.",
            source="test",
            confidence=0.8,
            tags=["preference"],
            trust=TrustLevel.proposed,
        )
    )

    assert first.id in second.contradicts
    assert second.id in store.get_memory(first.id).contradicts


def test_cleanup_rejects_old_low_quality_memory(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "test.sqlite3")
    old = MemoryItem(
        kind=MemoryKind.episodic,
        content="Tiny note",
        source="test",
        confidence=0.1,
        timestamp=datetime.now(timezone.utc) - timedelta(days=365),
        tags=[],
    )
    store.add_memory(old)

    result = store.cleanup_memories(min_quality=0.5)
    loaded = store.get_memory(old.id)

    assert result["rejected"] >= 1
    assert loaded.trust == TrustLevel.rejected


def test_search_prefers_high_confidence_memory(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "test.sqlite3")
    low = MemoryItem(
        kind=MemoryKind.semantic,
        content="User prefers concise answers.",
        source="test",
        confidence=0.2,
        tags=["preference"],
        trust=TrustLevel.proposed,
    )
    high = MemoryItem(
        kind=MemoryKind.semantic,
        content="User prefers concise answers with clear next steps.",
        source="test",
        confidence=0.95,
        tags=["preference"],
        trust=TrustLevel.proposed,
    )
    store.add_memory(low)
    promoted = store.add_memory(high)

    results = MemoryRetriever(store).search("concise answers", limit=2)

    assert results[0].memory.id == promoted.id

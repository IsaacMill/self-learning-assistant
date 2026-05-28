from pathlib import Path
import sqlite3

import pytest

from assistant.main import chat
from assistant.memory import MemoryStore
from assistant.retrieval import MemoryRetriever
from assistant.schemas import ChatRequest, MemoryItem, MemoryKind, TrustLevel


def test_memory_round_trip_and_promotion(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "test.sqlite3")
    item = MemoryItem(
        kind=MemoryKind.semantic,
        content="User prefers concise answers",
        source="test",
        confidence=0.8,
        tags=["preference"],
        trust=TrustLevel.proposed,
    )

    store.add_memory(item)
    loaded = store.get_memory(item.id)

    assert loaded is not None
    assert loaded.source == "test"
    assert loaded.trust == TrustLevel.proposed

    promoted = store.promote_memory(item.id, approved_by="test-user")
    assert promoted.kind == MemoryKind.semantic
    assert promoted.trust == TrustLevel.trusted
    assert "approved_by:test-user" in promoted.tags


def test_retrieval_excludes_raw_memory_by_default(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "test.sqlite3")
    raw = MemoryItem(
        kind=MemoryKind.semantic,
        content="User likes green tea",
        source="test",
        confidence=0.8,
        tags=["preference"],
        trust=TrustLevel.raw,
    )
    trusted = MemoryItem(
        kind=MemoryKind.semantic,
        content="User likes concise summaries",
        source="test",
        confidence=0.9,
        tags=["preference", "concise"],
        trust=TrustLevel.proposed,
    )
    store.add_many([raw, trusted])
    trusted = store.promote_memory(trusted.id, approved_by="test-user")

    results = MemoryRetriever(store).search("What summaries does the user like?", limit=5)

    assert [result.memory.id for result in results] == [trusted.id]


def test_episodic_memory_remains_raw(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "test.sqlite3")
    item = MemoryItem(
        kind=MemoryKind.episodic,
        content="User asked about project status and assistant summarized the task.",
        source="interaction",
        confidence=0.6,
        tags=["conversation"],
    )

    stored = store.add_memory(item)

    assert stored.kind == MemoryKind.episodic
    assert stored.trust == TrustLevel.raw


def test_store_rejects_secret_memory(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "test.sqlite3")
    item = MemoryItem(
        kind=MemoryKind.semantic,
        content="The user api_key=super-secret-value should be remembered",
        source="test",
        confidence=0.9,
        tags=["secret"],
        trust=TrustLevel.proposed,
    )

    with pytest.raises(ValueError, match="secret"):
        store.add_memory(item)


def test_store_rejects_direct_trusted_memory(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "test.sqlite3")
    item = MemoryItem(
        kind=MemoryKind.semantic,
        content="User prefers direct answers with citations.",
        source="test",
        confidence=0.9,
        tags=["preference"],
        trust=TrustLevel.trusted,
    )

    with pytest.raises(ValueError, match="promote_memory"):
        store.add_memory(item)


def test_promotion_requires_proposed_semantic_memory_and_approval(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "test.sqlite3")
    raw = MemoryItem(
        kind=MemoryKind.episodic,
        content="User asked a question and assistant answered.",
        source="test",
        confidence=0.8,
        tags=["conversation"],
    )
    store.add_memory(raw)

    with pytest.raises(PermissionError):
        store.promote_memory(raw.id, approved_by="")
    with pytest.raises(ValueError, match="proposed"):
        store.promote_memory(raw.id, approved_by="test-user")


def test_interaction_logs_redact_secrets(tmp_path: Path) -> None:
    db_path = tmp_path / "test.sqlite3"
    store = MemoryStore(db_path)

    store.log_interaction("my token=secret-token-value", "ok api_key=another-secret", {})

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT user_message, assistant_answer FROM interactions").fetchone()
    assert "secret-token-value" not in row[0]
    assert "another-secret" not in row[1]
    assert "[REDACTED_SECRET]" in row[0]


def test_blocked_action_request_is_refused_before_llm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASSISTANT_DB", str(tmp_path / "test.sqlite3"))
    monkeypatch.setenv("ASSISTANT_CHROMA_PATH", str(tmp_path / "chroma"))
    from assistant.config import get_settings

    get_settings.cache_clear()

    response = chat(ChatRequest(message="Please run shell command ls", debug=True))

    assert "cannot execute shell commands" in response.answer
    assert response.used_memories == []

"""SQLite-backed memory store."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from assistant.judge import memory_write_judge
from assistant.memory_quality import apply_quality_scores, is_duplicate, looks_contradictory, quality_score
from assistant.safety import redact_secrets, validate_memory_for_storage
from assistant.schemas import MemoryItem, MemoryKind, TrustLevel


class MemoryStore:
    """Persist and query assistant memories in SQLite."""

    def __init__(self, database_path: Path | str):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_type TEXT NOT NULL DEFAULT 'reflection',
                    timestamp TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    usefulness_score REAL NOT NULL DEFAULT 0.5,
                    confidence_score REAL NOT NULL DEFAULT 0.5,
                    recency_score REAL NOT NULL DEFAULT 1.0,
                    tags TEXT NOT NULL,
                    trust TEXT NOT NULL,
                    merged_from TEXT NOT NULL DEFAULT '[]',
                    contradicts TEXT NOT NULL DEFAULT '[]'
                )
                """
            )
            self._ensure_memory_columns(conn)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_message TEXT NOT NULL,
                    assistant_answer TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    judges TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS inter_agent_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _ensure_memory_columns(conn: sqlite3.Connection) -> None:
        """Add memory columns for existing SQLite databases."""

        rows = conn.execute("PRAGMA table_info(memories)").fetchall()
        existing = {row["name"] for row in rows}
        migrations = {
            "source_type": "ALTER TABLE memories ADD COLUMN source_type TEXT NOT NULL DEFAULT 'reflection'",
            "usefulness_score": "ALTER TABLE memories ADD COLUMN usefulness_score REAL NOT NULL DEFAULT 0.5",
            "confidence_score": "ALTER TABLE memories ADD COLUMN confidence_score REAL NOT NULL DEFAULT 0.5",
            "recency_score": "ALTER TABLE memories ADD COLUMN recency_score REAL NOT NULL DEFAULT 1.0",
            "merged_from": "ALTER TABLE memories ADD COLUMN merged_from TEXT NOT NULL DEFAULT '[]'",
            "contradicts": "ALTER TABLE memories ADD COLUMN contradicts TEXT NOT NULL DEFAULT '[]'",
        }
        for column, statement in migrations.items():
            if column not in existing:
                conn.execute(statement)

    def add_memory(self, item: MemoryItem, allow_sensitive: bool = False) -> MemoryItem:
        """Insert or replace a memory item after enforcing safety invariants."""

        storage_check = validate_memory_for_storage(item, allow_sensitive=allow_sensitive)
        if not storage_check.passed:
            raise ValueError(storage_check.explanation)
        if item.trust == TrustLevel.trusted:
            raise ValueError("Trusted memories must be created with promote_memory().")
        item = apply_quality_scores(item)
        existing_memories = self.list_memories(limit=1000)
        for existing in existing_memories:
            if existing.id != item.id and is_duplicate(existing, item):
                return self._merge_memory(existing, item)
        item.contradicts = sorted(
            {existing.id for existing in existing_memories if looks_contradictory(existing, item)}
        )
        for existing_id in item.contradicts:
            existing = self.get_memory(existing_id)
            if existing is not None and item.id not in existing.contradicts:
                existing.contradicts.append(item.id)
                self._write_memory(existing)

        self._write_memory(item)
        return item

    def _write_memory(self, item: MemoryItem) -> None:
        """Write a memory row without duplicate checks."""

        item = apply_quality_scores(item)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memories
                (
                    id, kind, content, source, source_type, timestamp, confidence,
                    usefulness_score, confidence_score, recency_score, tags, trust,
                    merged_from, contradicts
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id,
                    item.kind.value,
                    item.content,
                    item.source,
                    item.source_type,
                    item.timestamp.isoformat(),
                    item.confidence,
                    item.usefulness_score,
                    item.confidence_score,
                    item.recency_score,
                    json.dumps(item.tags),
                    item.trust.value,
                    json.dumps(item.merged_from),
                    json.dumps(item.contradicts),
                ),
            )

    def _merge_memory(self, existing: MemoryItem, incoming: MemoryItem) -> MemoryItem:
        """Merge duplicate memories, preserving the stronger row id."""

        existing.tags = sorted(set(existing.tags) | set(incoming.tags))
        existing.merged_from = sorted(set(existing.merged_from + incoming.merged_from + [incoming.id]))
        existing.confidence = max(existing.confidence, incoming.confidence)
        existing.usefulness_score = max(existing.usefulness_score, incoming.usefulness_score)
        existing.confidence_score = max(existing.confidence_score, incoming.confidence_score)
        existing.recency_score = max(existing.recency_score, incoming.recency_score)
        if quality_score(incoming) > quality_score(existing):
            existing.content = incoming.content
            existing.source = incoming.source
            existing.source_type = incoming.source_type
        self._write_memory(existing)
        return existing

    def list_memories(
        self,
        kind: MemoryKind | None = None,
        trust: TrustLevel | None = None,
        limit: int = 100,
    ) -> list[MemoryItem]:
        """List memories filtered by kind and trust."""

        clauses: list[str] = []
        params: list[object] = []
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind.value)
        if trust is not None:
            clauses.append("trust = ?")
            params.append(trust.value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM memories {where} ORDER BY timestamp DESC LIMIT ?",
                [*params, limit],
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def get_memory(self, memory_id: str) -> MemoryItem | None:
        """Return one memory by id."""

        with self._connect() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._row_to_memory(row) if row else None

    def promote_memory(
        self,
        memory_id: str,
        approved_by: str,
        allow_sensitive: bool = False,
    ) -> MemoryItem:
        """Promote a proposed semantic memory after explicit human approval."""

        item = self.get_memory(memory_id)
        if item is None:
            raise KeyError(f"Unknown memory id: {memory_id}")
        if not approved_by.strip():
            raise PermissionError("Memory promotion requires a human approver.")
        if item.trust != TrustLevel.proposed:
            raise ValueError("Only proposed memories can be promoted to trusted memory.")
        if item.kind != MemoryKind.semantic:
            raise ValueError("Only semantic memories can be promoted to trusted memory.")
        write_check = memory_write_judge(item, allow_sensitive=allow_sensitive)
        if not write_check.passed:
            raise ValueError(write_check.explanation)
        item.kind = MemoryKind.semantic
        item.trust = TrustLevel.trusted
        approval_tag = f"approved_by:{approved_by}"
        if approval_tag not in item.tags:
            item.tags.append(approval_tag)
        self._write_memory(item)
        return item

    def log_interaction(self, user_message: str, assistant_answer: str, judges: dict[str, object]) -> None:
        """Persist an interaction log."""

        from datetime import datetime, timezone

        safe_user_message = redact_secrets(user_message)
        safe_assistant_answer = redact_secrets(assistant_answer)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO interactions (user_message, assistant_answer, timestamp, judges)
                VALUES (?, ?, ?, ?)
                """,
                (
                    safe_user_message,
                    safe_assistant_answer,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(judges, default=str),
                ),
            )

    def add_many(self, items: Iterable[MemoryItem]) -> list[MemoryItem]:
        """Insert multiple memories."""

        return [self.add_memory(item) for item in items]

    def cleanup_memories(self, min_quality: float = 0.35) -> dict[str, int]:
        """Decay low-quality memories and reject memories below the threshold."""

        decayed = 0
        rejected = 0
        for item in self.list_memories(limit=5000):
            before = item.recency_score
            item = apply_quality_scores(item)
            if item.recency_score < before:
                decayed += 1
            if item.trust != TrustLevel.trusted and quality_score(item) < min_quality:
                item.trust = TrustLevel.rejected
                rejected += 1
            self._write_memory(item)
        return {"decayed": decayed, "rejected": rejected}

    def log_agent_message(self, task_id: str, agent_name: str, role: str, content: str) -> None:
        """Persist inter-agent communication."""

        from datetime import datetime, timezone

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO inter_agent_messages (task_id, agent_name, role, content, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    agent_name,
                    role,
                    redact_secrets(content),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            id=row["id"],
            kind=MemoryKind(row["kind"]),
            content=row["content"],
            source=row["source"],
            source_type=row["source_type"],
            timestamp=row["timestamp"],
            confidence=row["confidence"],
            usefulness_score=row["usefulness_score"],
            confidence_score=row["confidence_score"],
            recency_score=row["recency_score"],
            tags=json.loads(row["tags"]),
            trust=TrustLevel(row["trust"]),
            merged_from=json.loads(row["merged_from"]),
            contradicts=json.loads(row["contradicts"]),
        )

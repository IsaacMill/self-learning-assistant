"""Memory retrieval with optional ChromaDB indexing and lexical fallback."""

from __future__ import annotations

import math
import re
from pathlib import Path

from assistant.memory_quality import quality_score
from assistant.memory import MemoryStore
from assistant.schemas import MemoryItem, RetrievedMemory, TrustLevel

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> set[str]:
    """Tokenize text for deterministic fallback retrieval."""

    return {token.lower() for token in TOKEN_RE.findall(text)}


class MemoryRetriever:
    """Retrieve relevant trusted and proposed memories before answering."""

    def __init__(self, store: MemoryStore, chroma_path: Path | str | None = None):
        self.store = store
        self.chroma_path = Path(chroma_path) if chroma_path else None
        self._collection = None
        if chroma_path is not None:
            self._init_chroma()

    def _init_chroma(self) -> None:
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(self.chroma_path))
            self._collection = client.get_or_create_collection("assistant_memory")
        except Exception:
            self._collection = None

    def index_memory(self, item: MemoryItem) -> None:
        """Add a memory to the optional vector index."""

        if self._collection is None:
            return
        self._collection.upsert(
            ids=[item.id],
            documents=[item.content],
            metadatas=[{"kind": item.kind.value, "trust": item.trust.value, "source": item.source}],
        )

    def search(self, query: str, limit: int = 5, include_raw: bool = False) -> list[RetrievedMemory]:
        """Search memory and return the most relevant items."""

        if self._collection is not None:
            results = self._search_chroma(query, limit, include_raw)
            if results:
                return results
        return self._search_lexical(query, limit, include_raw)

    def _search_chroma(self, query: str, limit: int, include_raw: bool) -> list[RetrievedMemory]:
        result = self._collection.query(query_texts=[query], n_results=limit * 3)
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        retrieved: list[RetrievedMemory] = []
        for memory_id, distance in zip(ids, distances):
            item = self.store.get_memory(memory_id)
            if item is None:
                continue
            if not include_raw and item.trust == TrustLevel.raw:
                continue
            base_score = max(0.0, 1.0 - float(distance))
            retrieved.append(RetrievedMemory(memory=item, score=self._rank_score(base_score, item)))
            if len(retrieved) >= limit:
                break
        return retrieved

    def _search_lexical(self, query: str, limit: int, include_raw: bool) -> list[RetrievedMemory]:
        query_tokens = tokenize(query)
        candidates = self.store.list_memories(limit=500)
        scored: list[RetrievedMemory] = []
        for item in candidates:
            if not include_raw and item.trust == TrustLevel.raw:
                continue
            item_tokens = tokenize(" ".join([item.content, *item.tags]))
            if not item_tokens:
                continue
            overlap = len(query_tokens & item_tokens)
            if overlap == 0:
                continue
            base_score = overlap / math.sqrt(len(query_tokens or {""}) * len(item_tokens))
            score = self._rank_score(base_score, item)
            scored.append(RetrievedMemory(memory=item, score=score))
        scored.sort(key=lambda found: found.score, reverse=True)
        return scored[:limit]

    @staticmethod
    def _rank_score(relevance_score: float, item: MemoryItem) -> float:
        """Blend relevance with memory quality, preferring high-confidence memories."""

        trust_boost = {
            TrustLevel.trusted: 0.15,
            TrustLevel.proposed: 0.02,
            TrustLevel.raw: -0.15,
            TrustLevel.rejected: -1.0,
        }[item.trust]
        contradiction_penalty = 0.15 if item.contradicts else 0.0
        score = (0.65 * relevance_score) + (0.35 * quality_score(item)) + trust_boost - contradiction_penalty
        return max(0.0, min(1.0, score))

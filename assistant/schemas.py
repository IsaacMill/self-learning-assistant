"""Pydantic schemas shared across the assistant."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class MemoryKind(str, Enum):
    """Supported memory categories."""

    episodic = "episodic"
    semantic = "semantic"
    failure = "failure"


class TrustLevel(str, Enum):
    """Memory trust states."""

    raw = "raw"
    proposed = "proposed"
    trusted = "trusted"
    rejected = "rejected"


class MemoryItem(BaseModel):
    """A stored memory with provenance and trust metadata."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    kind: MemoryKind
    content: str
    source: str
    source_type: str = "reflection"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = Field(ge=0.0, le=1.0)
    usefulness_score: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    recency_score: float = Field(default=1.0, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    trust: TrustLevel = TrustLevel.raw
    merged_from: list[str] = Field(default_factory=list)
    contradicts: list[str] = Field(default_factory=list)


class RetrievedMemory(BaseModel):
    """A memory returned by retrieval."""

    memory: MemoryItem
    score: float = Field(ge=0.0)


class JudgeResult(BaseModel):
    """Standard result returned by every judge."""

    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    explanation: str
    suggested_correction: Optional[str] = None


class ChatRequest(BaseModel):
    """Request body for the API chat endpoint."""

    message: str
    debug: bool = False
    allow_sensitive_memory: bool = False


class ChatResponse(BaseModel):
    """Response body for the API chat endpoint."""

    answer: str
    judges: dict[str, JudgeResult]
    used_memories: list[RetrievedMemory] = Field(default_factory=list)
    proposed_memories: list[MemoryItem] = Field(default_factory=list)


class ReflectionResult(BaseModel):
    """Reflection output after an interaction."""

    summary: str
    learned: list[str] = Field(default_factory=list)
    possible_mistakes: list[str] = Field(default_factory=list)
    proposed_memories: list[MemoryItem] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

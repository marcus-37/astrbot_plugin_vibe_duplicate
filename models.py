from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class StoredMessage:
    id: int
    user_id: str
    message: str
    normalized_message: str
    timestamp: int
    semantic_tag: str
    message_embedding: list[float]
    embedding_model: str
    style_vector: list[float]
    quality_score: float


@dataclass(slots=True)
class PendingMessage:
    user_id: str
    message: str
    normalized_message: str
    timestamp: int
    semantic_tag: str
    message_embedding: list[float]
    embedding_model: str
    style_vector: list[float]
    quality_score: float


@dataclass(slots=True)
class GeneratedProfile:
    user_id: str
    persona_summary: str
    updated_at: int
    message_count: int
    persona_version: int


@dataclass(slots=True)
class RetrievedExample:
    message: str
    semantic_tag: str
    score: float
    timestamp: int
    semantic_score: float = 0.0
    recency_score: float = 0.0
    style_match_score: float = 0.0
    semantic_tag_score: float = 0.0
    quality_score: float = 0.0
    style_brief: str = ""


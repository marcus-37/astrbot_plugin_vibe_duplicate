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


@dataclass(slots=True)
class PendingMessage:
    user_id: str
    message: str
    normalized_message: str
    timestamp: int
    semantic_tag: str
    message_embedding: list[float]
    embedding_model: str


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


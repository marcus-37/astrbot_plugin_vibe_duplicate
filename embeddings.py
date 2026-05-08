from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Protocol


TOKEN_RE = re.compile(r"[\w]+|[\u4e00-\u9fff]")


@dataclass(slots=True)
class EmbeddingResult:
    vector: list[float]
    model: str


class EmbeddingProvider(Protocol):
    model_name: str

    async def embed(self, text: str) -> EmbeddingResult:
        ...


class PlaceholderEmbeddingProvider:
    """Deterministic local embedding provider used until a real provider is configured."""

    model_name = "placeholder-hash-v1"

    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions

    async def embed(self, text: str) -> EmbeddingResult:
        vector = [0.0] * self.dimensions
        tokens = TOKEN_RE.findall(text.lower()) or list(text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return EmbeddingResult([v / norm for v in vector], self.model_name)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot = sum(left[i] * right[i] for i in range(size))
    l_norm = math.sqrt(sum(v * v for v in left[:size])) or 1.0
    r_norm = math.sqrt(sum(v * v for v in right[:size])) or 1.0
    return dot / (l_norm * r_norm)


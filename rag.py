from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from .embeddings import EmbeddingProvider, cosine_similarity
from .models import RetrievedExample
from .storage import AvatarStore


class RagRetriever:
    def __init__(
        self,
        store: AvatarStore,
        embedding_provider: EmbeddingProvider,
        *,
        cache_ttl_seconds: int = 60,
        scan_limit: int = 1000,
    ) -> None:
        self.store = store
        self.embedding_provider = embedding_provider
        self.cache_ttl_seconds = cache_ttl_seconds
        self.scan_limit = scan_limit

    async def retrieve(self, user_id: str, query: str, top_k: int = 8) -> list[RetrievedExample]:
        query = (query or "").strip()
        if not query:
            return []
        cache_key = self._cache_key(query, top_k)
        cached = await self.store.get_cache(user_id, cache_key, self.cache_ttl_seconds)
        if cached:
            try:
                return [RetrievedExample(**item) for item in json.loads(cached)]
            except (TypeError, json.JSONDecodeError):
                pass

        query_embedding = (await self.embedding_provider.embed(query)).vector
        candidates = await self.store.messages_for_retrieval(user_id, self.scan_limit)
        scored = []
        for item in candidates:
            score = cosine_similarity(query_embedding, item.message_embedding)
            if score <= 0:
                continue
            scored.append(
                RetrievedExample(
                    message=item.normalized_message,
                    semantic_tag=item.semantic_tag,
                    score=score,
                    timestamp=item.timestamp,
                ),
            )
        scored.sort(key=lambda item: (item.score, item.timestamp), reverse=True)
        results = scored[:top_k]
        await self.store.set_cache(
            user_id,
            cache_key,
            json.dumps([asdict(item) for item in results], ensure_ascii=False),
        )
        return results

    def _cache_key(self, query: str, top_k: int) -> str:
        digest = hashlib.blake2b(query.encode("utf-8"), digest_size=12).hexdigest()
        return f"rag:{self.embedding_provider.model_name}:{top_k}:{digest}"

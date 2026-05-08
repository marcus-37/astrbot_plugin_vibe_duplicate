from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict

from .embeddings import EmbeddingProvider, cosine_similarity
from .models import GeneratedProfile, RetrievedExample, StoredMessage
from .storage import AvatarStore
from .style import StyleAnalyzer


class RagRetriever:
    def __init__(
        self,
        store: AvatarStore,
        embedding_provider: EmbeddingProvider,
        *,
        cache_ttl_seconds: int = 60,
        scan_limit: int = 1000,
        recall_k: int = 50,
    ) -> None:
        self.store = store
        self.embedding_provider = embedding_provider
        self.cache_ttl_seconds = cache_ttl_seconds
        self.scan_limit = scan_limit
        self.recall_k = recall_k
        self.style_analyzer = StyleAnalyzer()

    async def retrieve(
        self,
        user_id: str,
        query: str,
        top_k: int = 8,
        profile: GeneratedProfile | None = None,
        query_tag: str = "neutral",
    ) -> list[RetrievedExample]:
        query = (query or "").strip()
        if not query:
            return []
        cache_key = self._cache_key(query, top_k, profile)
        cached = await self.store.get_cache(user_id, cache_key, self.cache_ttl_seconds)
        if cached:
            try:
                return [RetrievedExample(**item) for item in json.loads(cached)]
            except (TypeError, json.JSONDecodeError):
                pass

        query_embedding = (await self.embedding_provider.embed(query)).vector
        query_style = self.style_analyzer.analyze(query, query_tag)
        candidates = await self.store.messages_for_retrieval(user_id, self.scan_limit)
        semantic_recall = self._semantic_recall(candidates, query_embedding)
        reranked = self._rerank(
            semantic_recall,
            query_style.vector,
            query_tag,
            profile.persona_summary if profile else "",
        )
        results = reranked[:top_k]
        await self.store.set_cache(
            user_id,
            cache_key,
            json.dumps([asdict(item) for item in results], ensure_ascii=False),
        )
        return results

    def _semantic_recall(
        self,
        candidates: list[StoredMessage],
        query_embedding: list[float],
    ) -> list[tuple[StoredMessage, float]]:
        scored = []
        for item in candidates:
            if item.quality_score < 0.15:
                continue
            semantic_score = max(0.0, cosine_similarity(query_embedding, item.message_embedding))
            if semantic_score <= 0:
                continue
            scored.append((item, semantic_score))
        scored.sort(key=lambda pair: (pair[1], pair[0].quality_score, pair[0].timestamp), reverse=True)
        return scored[: self.recall_k]

    def _rerank(
        self,
        recalled: list[tuple[StoredMessage, float]],
        query_style_vector: list[float],
        query_tag: str,
        persona_summary: str,
    ) -> list[RetrievedExample]:
        candidates: list[RetrievedExample] = []
        selected_text_keys: set[str] = set()
        selected_day_buckets: dict[int, int] = {}

        for item, semantic_score in recalled:
            recency_score = self.style_analyzer.recency_score(item.timestamp)
            style_score = self.style_analyzer.style_similarity(query_style_vector, item.style_vector)
            tag_score = self._semantic_tag_score(query_tag, item.semantic_tag)
            persona_score = self._persona_score(item, persona_summary)
            spam_penalty = 0.5 if item.quality_score < 0.25 else 1.0
            final_score = (
                semantic_score * 0.50
                + recency_score * 0.20
                + style_score * 0.20
                + tag_score * 0.10
            )
            final_score *= 0.75 + item.quality_score * 0.25
            final_score *= 0.9 + persona_score * 0.1
            final_score *= spam_penalty
            style_brief = self.style_analyzer.style_brief(
                item.normalized_message,
                item.semantic_tag,
                item.style_vector,
            )
            candidates.append(
                RetrievedExample(
                    message=item.normalized_message,
                    semantic_tag=item.semantic_tag,
                    score=final_score,
                    timestamp=item.timestamp,
                    semantic_score=semantic_score,
                    recency_score=recency_score,
                    style_match_score=style_score,
                    semantic_tag_score=tag_score,
                    quality_score=item.quality_score,
                    style_brief=style_brief,
                ),
            )

        selected: list[RetrievedExample] = []
        remaining = candidates[:]
        while remaining:
            best_index = 0
            best_score = -1.0
            for index, candidate in enumerate(remaining):
                adjusted = candidate.score
                adjusted *= self._repetition_penalty(candidate.message, selected_text_keys)
                adjusted *= self._temporal_penalty(candidate.timestamp, selected_day_buckets)
                if adjusted > best_score:
                    best_score = adjusted
                    best_index = index
            chosen = remaining.pop(best_index)
            chosen.score = best_score
            selected.append(chosen)
            selected_text_keys.add(self._text_key(chosen.message))
            day = chosen.timestamp // 86400
            selected_day_buckets[day] = selected_day_buckets.get(day, 0) + 1

        return selected

    def _semantic_tag_score(self, query_tag: str, item_tag: str) -> float:
        if not query_tag or query_tag == "neutral":
            return 0.5 if item_tag == "neutral" else 0.65
        if query_tag == item_tag:
            return 1.0
        compatible = {
            "complaint": {"sarcasm", "silly", "meme"},
            "sarcasm": {"complaint", "meme"},
            "silly": {"meme", "complaint"},
            "meme": {"silly", "sarcasm"},
            "serious_explain": {"banter", "neutral"},
        }
        return 0.7 if item_tag in compatible.get(query_tag, set()) else 0.25

    def _persona_score(self, item: StoredMessage, persona_summary: str) -> float:
        if not persona_summary:
            return item.quality_score
        summary = persona_summary.lower()
        text = item.normalized_message.lower()
        tokens = set(re.findall(r"[\w]+|[\u4e00-\u9fff]{2,}", text))
        matched = sum(1 for token in tokens if token and token in summary)
        return min(1.0, item.quality_score * 0.7 + matched * 0.1)

    def _repetition_penalty(self, text: str, selected_keys: set[str]) -> float:
        key = self._text_key(text)
        if key in selected_keys:
            return 0.3
        return 1.0

    def _temporal_penalty(self, timestamp: int, selected_day_buckets: dict[int, int]) -> float:
        count = selected_day_buckets.get(timestamp // 86400, 0)
        if count <= 1:
            return 1.0
        if count == 2:
            return 0.85
        return 0.7

    def _text_key(self, text: str) -> str:
        return re.sub(r"\s+", "", text.lower())[:48]

    def _cache_key(self, query: str, top_k: int, profile: GeneratedProfile | None) -> str:
        digest = hashlib.blake2b(query.encode("utf-8"), digest_size=12).hexdigest()
        version = profile.persona_version if profile else 0
        return f"rag:v3:{self.embedding_provider.model_name}:{top_k}:{version}:{digest}"

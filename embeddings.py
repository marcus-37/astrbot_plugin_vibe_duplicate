from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


TOKEN_RE = re.compile(r"[\w]+|[\u4e00-\u9fff]")


@dataclass(slots=True)
class EmbeddingResult:
    vector: list[float]
    model: str


class EmbeddingProvider(Protocol):
    model_name: str

    async def embed(self, text: str) -> EmbeddingResult:
        ...

    async def embed_many(self, texts: list[str]) -> list[EmbeddingResult]:
        ...


class BaseEmbeddingProvider:
    model_name = "base"

    async def embed(self, text: str) -> EmbeddingResult:
        results = await self.embed_many([text])
        return results[0]

    async def embed_many(self, texts: list[str]) -> list[EmbeddingResult]:
        return [await self.embed(text) for text in texts]


class PlaceholderEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic local fallback. Good for bootstrapping, not semantic search."""

    model_name = "placeholder-hash-v2"

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
        return EmbeddingResult(normalize_vector(vector), self.model_name)

    async def embed_many(self, texts: list[str]) -> list[EmbeddingResult]:
        return [await self.embed(text) for text in texts]


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        dimensions: int | None = None,
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key
        self.model_name = model
        self.base_url = base_url.rstrip("/").removesuffix("/embeddings")
        self.dimensions = dimensions
        self.timeout = timeout

    async def embed_many(self, texts: list[str]) -> list[EmbeddingResult]:
        payload: dict[str, Any] = {"model": self.model_name, "input": texts}
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        data = await asyncio.to_thread(
            post_json,
            f"{self.base_url}/embeddings",
            payload,
            {"Authorization": f"Bearer {self.api_key}"},
            self.timeout,
        )
        vectors = [item["embedding"] for item in data["data"]]
        return [EmbeddingResult(normalize_vector(vector), self.model_name) for vector in vectors]


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-embedding-exp-03-07",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key
        self.model_name = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def embed_many(self, texts: list[str]) -> list[EmbeddingResult]:
        if not texts:
            return []
        url = f"{self.base_url}/models/{self.model_name}:batchEmbedContents?key={self.api_key}"
        model_ref = f"models/{self.model_name}"
        payload = {
            "requests": [
                {
                    "model": model_ref,
                    "content": {"parts": [{"text": text}]},
                }
                for text in texts
            ],
        }
        try:
            data = await asyncio.to_thread(post_json, url, payload, {}, self.timeout)
            vectors = [item["values"] for item in data["embeddings"]]
        except Exception:
            vectors = await self._embed_one_by_one(texts)
        return [EmbeddingResult(normalize_vector(vector), self.model_name) for vector in vectors]

    async def _embed_one_by_one(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            url = f"{self.base_url}/models/{self.model_name}:embedContent?key={self.api_key}"
            payload = {"content": {"parts": [{"text": text}]}}
            data = await asyncio.to_thread(post_json, url, payload, {}, self.timeout)
            vectors.append(data["embedding"]["values"])
        return vectors


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(
        self,
        *,
        model: str = "bge-m3",
        base_url: str = "http://127.0.0.1:11434",
        timeout: int = 60,
    ) -> None:
        self.model_name = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def embed_many(self, texts: list[str]) -> list[EmbeddingResult]:
        data = await asyncio.to_thread(
            post_json,
            f"{self.base_url}/api/embed",
            {"model": self.model_name, "input": texts},
            {},
            self.timeout,
        )
        vectors = data.get("embeddings") or [data["embedding"]]
        return [EmbeddingResult(normalize_vector(vector), self.model_name) for vector in vectors]


class SentenceTransformersEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model: str = "BAAI/bge-m3") -> None:
        self.model_name = model
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    async def embed_many(self, texts: list[str]) -> list[EmbeddingResult]:
        def run() -> list[list[float]]:
            model = self._load_model()
            vectors = model.encode(texts, normalize_embeddings=True)
            return [vector.tolist() for vector in vectors]

        vectors = await asyncio.to_thread(run)
        return [EmbeddingResult(vector, self.model_name) for vector in vectors]


class AstrBotEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, context: Any, provider_id: str = "") -> None:
        self.context = context
        self.provider_id = provider_id
        self.model_name = f"astrbot:{provider_id or 'default'}"

    def _resolve_provider(self):
        if self.provider_id:
            provider = self.context.get_provider_by_id(self.provider_id)
            if provider:
                return provider
        providers = self.context.get_all_embedding_providers()
        return providers[0] if providers else None

    def _resolve_model_name(self, provider: Any) -> str:
        raw_model = ""
        getter = getattr(provider, "get_model", None)
        if callable(getter):
            raw_model = str(getter() or "").strip()
        if not raw_model:
            raw_model = str(getattr(provider, "model_name", "") or "").strip()
        if not raw_model:
            raw_model = self.provider_id or "default"
        model_name = raw_model if raw_model.startswith("astrbot:") else f"astrbot:{raw_model}"
        self.model_name = model_name
        return model_name

    async def embed_many(self, texts: list[str]) -> list[EmbeddingResult]:
        provider = self._resolve_provider()
        if provider is None:
            raise RuntimeError("No AstrBot embedding provider is configured.")
        model_name = self._resolve_model_name(provider)
        if hasattr(provider, "get_embeddings"):
            vectors = await provider.get_embeddings(texts)
        else:
            vectors = [await provider.get_embedding(text) for text in texts]
        return [EmbeddingResult(normalize_vector(vector), model_name) for vector in vectors]


class CachedEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, provider: EmbeddingProvider, max_items: int = 4096) -> None:
        self.provider = provider
        self.max_items = max_items
        self._cache: dict[str, EmbeddingResult] = {}

    @property
    def model_name(self) -> str:
        return self.provider.model_name

    async def embed_many(self, texts: list[str]) -> list[EmbeddingResult]:
        results: list[EmbeddingResult | None] = []
        misses: list[str] = []
        miss_indexes: list[int] = []
        for text in texts:
            key = self._key(text)
            cached = self._cache.get(key)
            results.append(cached)
            if cached is None:
                miss_indexes.append(len(results) - 1)
                misses.append(text)
        if misses:
            embedded = await self.provider.embed_many(misses)
            for idx, result in zip(miss_indexes, embedded, strict=False):
                results[idx] = result
                self._cache[self._key(misses[miss_indexes.index(idx)])] = result
            self._trim()
        return [result for result in results if result is not None]

    def _key(self, text: str) -> str:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
        return f"{self.provider.model_name}:{digest}"

    def _trim(self) -> None:
        while len(self._cache) > self.max_items:
            self._cache.pop(next(iter(self._cache)))


class EmbeddingQueue:
    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        batch_size: int = 16,
        flush_interval: float = 0.05,
    ) -> None:
        self.provider = provider
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._queue: asyncio.Queue[tuple[str, asyncio.Future[EmbeddingResult]] | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        if self._task:
            await self._queue.put(None)
            await self._task

    async def embed(self, text: str) -> EmbeddingResult:
        self.start()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[EmbeddingResult] = loop.create_future()
        await self._queue.put((text, future))
        return await future

    async def _worker(self) -> None:
        while True:
            first = await self._queue.get()
            if first is None:
                self._queue.task_done()
                return
            batch = [first]
            deadline = asyncio.get_running_loop().time() + self.flush_interval
            while len(batch) < self.batch_size:
                timeout = max(0.0, deadline - asyncio.get_running_loop().time())
                if timeout == 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    break
                if item is None:
                    self._queue.task_done()
                    await self._flush(batch)
                    return
                batch.append(item)
            await self._flush(batch)

    async def _flush(self, batch: list[tuple[str, asyncio.Future[EmbeddingResult]]]) -> None:
        texts = [item[0] for item in batch]
        futures = [item[1] for item in batch]
        try:
            results = await self.provider.embed_many(texts)
            for future, result in zip(futures, results, strict=False):
                if not future.done():
                    future.set_result(result)
        except Exception as exc:
            for future in futures:
                if not future.done():
                    future.set_exception(exc)
        finally:
            for _ in batch:
                self._queue.task_done()


def build_embedding_provider(config: Any, context: Any | None = None) -> EmbeddingProvider:
    provider_type = str(config_get(config, "embedding_provider", "placeholder")).lower()
    dimensions = int(config_get(config, "embedding_dimensions", 128))
    if provider_type == "openai":
        provider: EmbeddingProvider = OpenAIEmbeddingProvider(
            api_key=str(config_get(config, "embedding_api_key", "")),
            model=str(config_get(config, "embedding_model", "text-embedding-3-small")),
            base_url=str(config_get(config, "embedding_api_base", "") or "https://api.openai.com/v1"),
            dimensions=dimensions or None,
        )
    elif provider_type == "gemini":
        provider = GeminiEmbeddingProvider(
            api_key=str(config_get(config, "embedding_api_key", "")),
            model=str(config_get(config, "embedding_model", "gemini-embedding-exp-03-07")),
            base_url=str(config_get(config, "embedding_api_base", "") or "https://generativelanguage.googleapis.com/v1beta"),
        )
    elif provider_type == "ollama":
        provider = OllamaEmbeddingProvider(
            model=str(config_get(config, "embedding_model", "bge-m3")),
            base_url=str(config_get(config, "embedding_api_base", "") or "http://127.0.0.1:11434"),
        )
    elif provider_type == "sentence_transformers":
        provider = SentenceTransformersEmbeddingProvider(
            model=str(config_get(config, "embedding_model", "BAAI/bge-m3")),
        )
    elif provider_type == "astrbot" and context is not None:
        provider = AstrBotEmbeddingProvider(
            context,
            provider_id=str(config_get(config, "astrbot_embedding_provider_id", "")),
        )
    else:
        provider = PlaceholderEmbeddingProvider(dimensions=dimensions)
    if bool(config_get(config, "embedding_cache_enabled", True)):
        return CachedEmbeddingProvider(provider, max_items=int(config_get(config, "embedding_cache_size", 4096)))
    return provider


def post_json(url: str, payload: dict, headers: dict[str, str], timeout: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json", **headers}
    request = urllib.request.Request(url, data=body, headers=req_headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Embedding request failed: {exc.code} {detail}") from exc


def normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in vector)) or 1.0
    return [float(v) / norm for v in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot = sum(left[i] * right[i] for i in range(size))
    l_norm = math.sqrt(sum(v * v for v in left[:size])) or 1.0
    r_norm = math.sqrt(sum(v * v for v in right[:size])) or 1.0
    return dot / (l_norm * r_norm)


def config_get(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    try:
        return config.get(key, default)
    except AttributeError:
        return getattr(config, key, default)

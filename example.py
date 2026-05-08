"""Minimal local example for the Vibe Duplicate core modules.

Run from the AstrBot project root:

    python data/plugins/astrbot_plugin_vibe_duplicate/example.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time
from pathlib import Path

ASTRBOT_ROOT = Path(__file__).resolve().parents[3]
if str(ASTRBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(ASTRBOT_ROOT))

from data.plugins.astrbot_plugin_vibe_duplicate.embeddings import (
    PlaceholderEmbeddingProvider,
)
from data.plugins.astrbot_plugin_vibe_duplicate.models import PendingMessage
from data.plugins.astrbot_plugin_vibe_duplicate.rag import RagRetriever
from data.plugins.astrbot_plugin_vibe_duplicate.storage import AvatarStore


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = AvatarStore(Path(tmp))
        embeddings = PlaceholderEmbeddingProvider()
        retriever = RagRetriever(store, embeddings)

        samples = [
            "this issue is weird but still explainable",
            "lol wait, let me think how to say it",
            "the reason is simple: the context did not line up",
        ]
        for text in samples:
            embedded = await embeddings.embed(text)
            await store.add_message(
                PendingMessage(
                    user_id="demo_user",
                    message=text,
                    normalized_message=text,
                    timestamp=int(time.time()),
                    semantic_tag="demo",
                    message_embedding=embedded.vector,
                    embedding_model=embedded.model,
                ),
            )

        results = await retriever.retrieve("demo_user", "why did the context not line up", top_k=2)
        for item in results:
            print(f"{item.score:.3f} [{item.semantic_tag}] {item.message}")


if __name__ == "__main__":
    asyncio.run(main())

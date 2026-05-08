from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any

from astrbot.api import logger

from .prompting import build_persona_update_prompt
from .storage import AvatarStore


class PersonaUpdater:
    def __init__(
        self,
        store: AvatarStore,
        context: Any,
        *,
        threshold: int = 50,
        min_summary_length: int = 40,
    ) -> None:
        self.store = store
        self.context = context
        self.threshold = threshold
        self.min_summary_length = min_summary_length
        self._locks: dict[str, asyncio.Lock] = {}

    async def update_persona_if_needed(
        self,
        user_id: str,
        *,
        force: bool = False,
        umo: str | None = None,
    ) -> bool:
        lock = self._locks.setdefault(user_id, asyncio.Lock())
        async with lock:
            profile = await self.store.get_generated_profile(user_id)
            total = await self.store.message_count(user_id)
            learned = profile.message_count if profile else 0
            if not force and total - learned < self.threshold:
                return False

            new_limit = max(1, min(total - learned if profile else self.threshold, 200))
            new_messages = await self.store.recent_messages(user_id, new_limit)
            annotations = await self.store.notes(user_id, "admin_annotations", 20)
            memories = await self.store.notes(user_id, "third_party_memories", 20)
            old_summary = profile.persona_summary if profile else ""

            summary = await self._generate_summary(
                user_id=user_id,
                old_summary=old_summary,
                new_messages=new_messages,
                admin_annotations=annotations,
                third_party_memories=memories,
                umo=umo,
            )
            if not self._passes_consistency_guard(summary, old_summary):
                logger.warning("[VibeDuplicate] persona update rejected by guard for %s", user_id)
                return False

            await self.store.upsert_generated_profile(user_id, summary, total)
            return True

    async def _generate_summary(
        self,
        *,
        user_id: str,
        old_summary: str,
        new_messages: list,
        admin_annotations: list[str],
        third_party_memories: list[str],
        umo: str | None,
    ) -> str:
        prompt = build_persona_update_prompt(
            user_id=user_id,
            old_summary=old_summary,
            new_messages=new_messages,
            admin_annotations=admin_annotations,
            third_party_memories=third_party_memories,
        )
        provider = None
        try:
            provider = self.context.get_using_provider(umo)
        except Exception as exc:
            logger.warning("[VibeDuplicate] cannot resolve chat provider: %s", exc)

        if provider:
            try:
                response = await provider.text_chat(prompt=prompt)
                text = (getattr(response, "completion_text", "") or "").strip()
                if text:
                    return text
            except Exception as exc:
                logger.warning("[VibeDuplicate] LLM persona update failed: %s", exc)

        return self._fallback_summary(old_summary, new_messages)

    def _fallback_summary(self, old_summary: str, new_messages: list) -> str:
        tags = Counter(item.semantic_tag for item in new_messages)
        samples = [item.normalized_message for item in new_messages[-8:]]
        tag_text = ", ".join(f"{tag}:{count}" for tag, count in tags.most_common())
        sample_text = " / ".join(samples[:5])
        pieces = []
        if old_summary:
            pieces.append(old_summary.strip())
        pieces.append(
            "Incremental observations: "
            f"recent style tags are {tag_text or 'neutral'}; "
            f"representative utterances include: {sample_text or 'none'}."
        )
        pieces.append(
            "Keep persona stable; imitate rhythm and wording only when supported by repeated evidence."
        )
        return "\n".join(pieces)

    def _passes_consistency_guard(self, summary: str, old_summary: str) -> bool:
        if len(summary.strip()) < self.min_summary_length:
            return False
        banned = ("我是本人", "我就是真人", "real human", "I am the real")
        if any(item.lower() in summary.lower() for item in banned):
            return False
        if old_summary and len(summary) > max(6000, len(old_summary) * 5):
            return False
        return True


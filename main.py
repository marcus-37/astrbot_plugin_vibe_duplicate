# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import time
from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register

from .cleaning import MessageCleaner, StyleClassifier
from .embeddings import PlaceholderEmbeddingProvider
from .models import PendingMessage
from .persona import PersonaUpdater
from .prompting import build_avatar_prompt
from .rag import RagRetriever
from .storage import AvatarStore


DISPLAY_NAME = "Vibe Duplicate"
VERSION = "2.0.0"
DATA_ROOT = Path("data/astrtbot_plugin_echo_avatar")


def cfg(config: AstrBotConfig, key: str, default):
    try:
        return config.get(key, default)
    except AttributeError:
        return default


@register(
    DISPLAY_NAME,
    "LumineStory",
    "Long-term persona learning with dynamic RAG style injection.",
    VERSION,
    "https://github.com/oyxning/astrtbot_plugin_echo_avatar",
)
class VibeDuplicatePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.context = context
        self.config = config
        self.store = AvatarStore(DATA_ROOT)
        self.embedding_provider = PlaceholderEmbeddingProvider(
            dimensions=int(cfg(config, "embedding_dimensions", 128)),
        )
        self.cleaner = MessageCleaner(
            max_length=int(cfg(config, "max_message_length", 1200)),
            min_length=int(cfg(config, "min_message_length", 2)),
            blacklist_words=list(cfg(config, "blacklist_words", [])),
            filter_commands=bool(cfg(config, "filter_commands", True)),
        )
        self.classifier = StyleClassifier()
        self.retriever = RagRetriever(
            self.store,
            self.embedding_provider,
            cache_ttl_seconds=int(cfg(config, "retrieval_cache_ttl_seconds", 60)),
            scan_limit=int(cfg(config, "retrieval_scan_limit", 1000)),
        )
        self.persona_updater = PersonaUpdater(
            self.store,
            context,
            threshold=int(cfg(config, "persona_update_threshold", 50)),
        )
        self.write_queue: asyncio.Queue[PendingMessage | None] = asyncio.Queue(
            maxsize=int(cfg(config, "write_queue_size", 1000)),
        )
        self.writer_task: asyncio.Task | None = None
        self.update_tasks: set[asyncio.Task] = set()
        logger.info("[VibeDuplicate] loaded; target users: %s", self.target_users)

    @property
    def target_users(self) -> list[str]:
        return [str(item) for item in cfg(self.config, "target_users", [])]

    async def initialize(self):
        self.writer_task = asyncio.create_task(self._writer_loop())

    @filter.event_message_type(filter.EventMessageType.ALL, priority=100)
    async def message_recorder(self, event: AstrMessageEvent):
        sender_id = event.get_sender_id()
        if sender_id not in self.target_users:
            return

        clean = self.cleaner.clean(event.message_str)
        if not clean.accepted:
            logger.debug("[VibeDuplicate] skipped message from %s: %s", sender_id, clean.reason)
            return

        await self.store.init_user(sender_id)
        if await self.store.is_duplicate_recent(sender_id, clean.normalized):
            logger.debug("[VibeDuplicate] skipped duplicate message from %s", sender_id)
            return

        embedding = await self.embedding_provider.embed(clean.normalized)
        timestamp = int(getattr(event.message_obj, "timestamp", 0) or time.time())
        pending = PendingMessage(
            user_id=sender_id,
            message=event.message_str.strip(),
            normalized_message=clean.normalized,
            timestamp=timestamp,
            semantic_tag=self.classifier.classify(clean.normalized),
            message_embedding=embedding.vector,
            embedding_model=embedding.model,
        )
        await self.write_queue.put(pending)

    async def _writer_loop(self) -> None:
        while True:
            item = await self.write_queue.get()
            try:
                if item is None:
                    return
                await self.store.add_message(item)
                task = asyncio.create_task(
                    self.persona_updater.update_persona_if_needed(item.user_id),
                )
                self.update_tasks.add(task)
                task.add_done_callback(self.update_tasks.discard)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("[VibeDuplicate] writer failed: %s", exc)
            finally:
                self.write_queue.task_done()

    @filter.on_llm_request()
    async def inject_avatar_prompt(
        self,
        event: AstrMessageEvent,
        request: ProviderRequest,
    ) -> None:
        if not bool(cfg(self.config, "enable_prompt_injection", True)):
            return
        user_id = self._select_avatar_user(event)
        if not user_id:
            return

        current_context = self._current_context(event, request)
        profile = await self.store.get_generated_profile(user_id)
        examples = await self.retriever.retrieve(
            user_id,
            current_context or request.prompt or "",
            top_k=int(cfg(self.config, "rag_top_k", 8)),
        )
        annotations = await self.store.notes(user_id, "admin_annotations", 20)
        memories = await self.store.notes(user_id, "third_party_memories", 20)
        recent = await self.store.recent_messages(
            user_id,
            int(cfg(self.config, "recent_user_context", 12)),
        )
        avatar_prompt = build_avatar_prompt(
            user_id=user_id,
            profile=profile,
            similar_examples=examples,
            current_context=current_context,
            admin_annotations=annotations,
            third_party_memories=memories,
            recent_messages=recent,
        )
        request.system_prompt = "\n\n".join(
            part for part in (request.system_prompt.strip(), avatar_prompt) if part
        )

    def _select_avatar_user(self, event: AstrMessageEvent) -> str | None:
        configured = str(cfg(self.config, "avatar_user_id", "") or "").strip()
        if configured:
            return configured
        targets = self.target_users
        if len(targets) == 1:
            return targets[0]
        mentioned = event.get_extra("duplicate_user_id")
        if mentioned:
            return str(mentioned)
        return targets[0] if targets else None

    def _current_context(self, event: AstrMessageEvent, request: ProviderRequest) -> str:
        parts: list[str] = []
        for ctx in (request.contexts or [])[-6:]:
            role = ctx.get("role", "unknown")
            content = ctx.get("content", "")
            if isinstance(content, str) and content.strip():
                parts.append(f"{role}: {content.strip()}")
        if request.prompt:
            parts.append(f"user: {request.prompt.strip()}")
        elif event.message_str:
            parts.append(f"user: {event.message_str.strip()}")
        return "\n".join(parts)

    @filter.command_group("duplicate", alias={"echo_avatar"})
    def duplicate_group(self):
        """Manage Vibe Duplicate."""
        pass

    @filter.permission_type(filter.PermissionType.ADMIN)
    @duplicate_group.command("status")
    async def status(self, event: AstrMessageEvent):
        users = self.target_users
        known = await self.store.user_ids()
        queue_size = self.write_queue.qsize()
        yield event.plain_result(
            "[VibeDuplicate]\n"
            f"version: {VERSION}\n"
            f"target_users: {', '.join(users) or 'none'}\n"
            f"known_user_dbs: {len(known)}\n"
            f"write_queue: {queue_size}\n"
            f"prompt_injection: {bool(cfg(self.config, 'enable_prompt_injection', True))}"
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @duplicate_group.command("profile")
    async def set_profile(self, event: AstrMessageEvent, user_id: str, key: str, *, value: str):
        await self.store.set_profile_item(user_id, key, value)
        yield event.plain_result(f"Updated profile `{key}` for {user_id}.")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @duplicate_group.command("annotate")
    async def add_annotation(self, event: AstrMessageEvent, user_id: str, *, text: str):
        await self.store.add_annotation(user_id, text, event.get_sender_id())
        yield event.plain_result(f"Added admin annotation for {user_id}.")

    @duplicate_group.command("memory")
    async def add_memory(self, event: AstrMessageEvent, user_id: str, *, text: str):
        await self.store.add_memory(user_id, text, event.get_sender_id())
        yield event.plain_result(f"Added third-party memory for {user_id}.")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @duplicate_group.command("update")
    async def update_persona(self, event: AstrMessageEvent, user_id: str):
        ok = await self.persona_updater.update_persona_if_needed(
            user_id,
            force=True,
            umo=event.unified_msg_origin,
        )
        yield event.plain_result(
            f"Persona update {'completed' if ok else 'skipped'} for {user_id}."
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @duplicate_group.command("rollback")
    async def rollback_persona(self, event: AstrMessageEvent, user_id: str):
        profile = await self.store.rollback_generated_profile(user_id)
        if not profile:
            yield event.plain_result(f"No previous persona version for {user_id}.")
            return
        yield event.plain_result(f"Rolled back {user_id} to persona version {profile.persona_version}.")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @duplicate_group.command("prompt")
    async def show_prompt(self, event: AstrMessageEvent, user_id: str, *, query: str = ""):
        profile = await self.store.get_generated_profile(user_id)
        examples = await self.retriever.retrieve(user_id, query or event.message_str, top_k=8)
        annotations = await self.store.notes(user_id, "admin_annotations", 20)
        memories = await self.store.notes(user_id, "third_party_memories", 20)
        recent = await self.store.recent_messages(user_id, 12)
        prompt = build_avatar_prompt(
            user_id=user_id,
            profile=profile,
            similar_examples=examples,
            current_context=query,
            admin_annotations=annotations,
            third_party_memories=memories,
            recent_messages=recent,
        )
        yield event.plain_result(prompt[:3500])

    @filter.permission_type(filter.PermissionType.ADMIN)
    @duplicate_group.command("preview")
    async def preview(self, event: AstrMessageEvent, user_id: str):
        profile = await self.store.get_generated_profile(user_id)
        total = await self.store.message_count(user_id)
        recent = await self.store.recent_messages(user_id, 8)
        tags = ", ".join(f"{item.semantic_tag}" for item in recent) or "none"
        summary = profile.persona_summary[:800] if profile else "No persona generated yet."
        yield event.plain_result(
            f"[VibeDuplicate preview: {user_id}]\n"
            f"messages: {total}\n"
            f"persona_version: {profile.persona_version if profile else 0}\n"
            f"recent_tags: {tags}\n\n"
            f"{summary}"
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @duplicate_group.command("clear")
    async def clear_user_data(self, event: AstrMessageEvent, user_id: str):
        removed = await self.store.clear_user(user_id)
        yield event.plain_result(
            f"Cleared data for {user_id}." if removed else f"No data found for {user_id}."
        )

    async def terminate(self):
        if self.writer_task:
            await self.write_queue.put(None)
            await self.write_queue.join()
            self.writer_task.cancel()
        for task in list(self.update_tasks):
            task.cancel()
        logger.info("[VibeDuplicate] terminated")

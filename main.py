# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register

from .cleaning import MessageCleaner, StyleClassifier
from .embeddings import EmbeddingQueue, build_embedding_provider
from .importer import ImportedMessage, extract_history_text, load_chat_records
from .models import PendingMessage
from .persona import PersonaUpdater
from .prompting import build_avatar_prompt
from .rag import RagRetriever
from .storage import AvatarStore
from .style import StyleAnalyzer


DISPLAY_NAME = "Vibe Duplicate"
VERSION = "2.0.0"
DATA_ROOT = Path("data/astrtbot_plugin_echo_avatar")


@dataclass(slots=True)
class ImportStats:
    total: int = 0
    imported: int = 0
    skipped: int = 0
    duplicate: int = 0
    failed: int = 0


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
    "https://github.com/marcus-37/astrbot_plugin_vibe_duplicate",
)
class VibeDuplicatePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.context = context
        self.config = config
        self.store = AvatarStore(DATA_ROOT)
        self.embedding_provider = build_embedding_provider(config, context)
        self.embedding_queue = EmbeddingQueue(
            self.embedding_provider,
            batch_size=int(cfg(config, "embedding_batch_size", 16)),
            flush_interval=float(cfg(config, "embedding_batch_flush_seconds", 0.05)),
        )
        self.cleaner = MessageCleaner(
            max_length=int(cfg(config, "max_message_length", 1200)),
            min_length=int(cfg(config, "min_message_length", 2)),
            blacklist_words=list(cfg(config, "blacklist_words", [])),
            filter_commands=bool(cfg(config, "filter_commands", True)),
        )
        self.classifier = StyleClassifier()
        self.style_analyzer = StyleAnalyzer()
        self.retriever = RagRetriever(
            self.store,
            self.embedding_provider,
            cache_ttl_seconds=int(cfg(config, "retrieval_cache_ttl_seconds", 60)),
            scan_limit=int(cfg(config, "retrieval_scan_limit", 1000)),
            recall_k=int(cfg(config, "retrieval_recall_k", 50)),
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
        self.embedding_queue.start()
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

        semantic_tag = self.classifier.classify(clean.normalized)
        style_profile = self.style_analyzer.analyze(clean.normalized, semantic_tag)
        embedding = await self.embedding_queue.embed(clean.normalized)
        timestamp = int(getattr(event.message_obj, "timestamp", 0) or time.time())
        pending = PendingMessage(
            user_id=sender_id,
            message=event.message_str.strip(),
            normalized_message=clean.normalized,
            timestamp=timestamp,
            semantic_tag=semantic_tag,
            message_embedding=embedding.vector,
            embedding_model=embedding.model,
            style_vector=style_profile.vector,
            quality_score=style_profile.quality_score,
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
        query_tag = self.classifier.classify(current_context or request.prompt or "")
        examples = await self.retriever.retrieve(
            user_id,
            current_context or request.prompt or "",
            top_k=int(cfg(self.config, "rag_top_k", 8)),
            profile=profile,
            query_tag=query_tag,
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

    @duplicate_group.command("import")
    async def import_chat_file(self, event: AstrMessageEvent, user_id: str, *, file_path: str):
        if not event.is_admin():
            yield event.plain_result(
                "权限不足：/duplicate import 只能由管理员执行。\n"
                f"sender_id: {event.get_sender_id()}\n"
                f"role: {event.role}"
            )
            return

        path = self._resolve_import_path(file_path)
        yield event.plain_result(
            "[VibeDuplicate] 已收到导入命令，开始读取聊天记录。\n"
            f"user_id: {user_id}\n"
            f"file: {path}"
        )
        if not path.exists() or not path.is_file():
            yield event.plain_result(f"找不到聊天记录文件：{path}")
            return

        try:
            records = await asyncio.to_thread(load_chat_records, path)
        except Exception as exc:
            logger.error("[VibeDuplicate] import file failed: %s", exc)
            yield event.plain_result(f"导入失败：无法解析 {path.name}。")
            return

        yield event.plain_result(
            "[VibeDuplicate] 聊天记录解析完成，开始清洗、去重和写入。\n"
            f"读取记录: {len(records)}"
        )
        stats = ImportStats(total=len(records))
        async for update in self._learn_imported_records_with_progress(
            user_id,
            records,
            umo=event.unified_msg_origin,
        ):
            if isinstance(update, ImportStats):
                stats = update
            else:
                yield event.plain_result(update)
        yield event.plain_result(
            "[VibeDuplicate 导入完成]\n"
            f"user_id: {user_id}\n"
            f"file: {path}\n"
            f"读取: {stats.total}\n"
            f"写入: {stats.imported}\n"
            f"跳过: {stats.skipped}\n"
            f"重复: {stats.duplicate}\n"
            f"失败: {stats.failed}"
        )

    @duplicate_group.command("backfill")
    async def backfill_history(
        self,
        event: AstrMessageEvent,
        user_id: str,
        limit: int = 500,
        platform_id: str = "",
        session_id: str = "",
    ):
        if not event.is_admin():
            yield event.plain_result(
                "权限不足：/duplicate backfill 只能由管理员执行。\n"
                f"sender_id: {event.get_sender_id()}\n"
                f"role: {event.role}"
            )
            return

        manager = getattr(self.context, "message_history_manager", None)
        if manager is None or not hasattr(manager, "get"):
            yield event.plain_result("当前 AstrBot 没有可用的 message_history_manager，无法自动回填。")
            return

        limit = max(1, min(int(limit), 5000))
        platform = platform_id.strip() or event.get_platform_id()
        sessions = self._history_session_candidates(event, session_id)
        yield event.plain_result(
            "[VibeDuplicate] 已收到历史回填命令，开始查询 AstrBot 消息历史。\n"
            f"user_id: {user_id}\n"
            f"platform_id: {platform}\n"
            f"candidate_sessions: {', '.join(sessions) or 'none'}\n"
            f"limit: {limit}"
        )
        records = []
        used_session = ""
        for candidate in sessions:
            records = await self._fetch_history_records(manager, platform, candidate, user_id, limit)
            if records:
                used_session = candidate
                break

        if not records:
            yield event.plain_result(
                "没有找到可回填的历史消息。请确认是在目标群/会话里执行，"
                "或手动指定 platform_id 与 session_id。"
            )
            return

        yield event.plain_result(
            "[VibeDuplicate] 历史记录读取完成，开始清洗、去重和写入。\n"
            f"session_id: {used_session}\n"
            f"读取记录: {len(records)}"
        )
        stats = ImportStats(total=len(records))
        async for update in self._learn_imported_records_with_progress(
            user_id,
            records,
            umo=event.unified_msg_origin,
        ):
            if isinstance(update, ImportStats):
                stats = update
            else:
                yield event.plain_result(update)
        yield event.plain_result(
            "[VibeDuplicate 历史回填完成]\n"
            f"user_id: {user_id}\n"
            f"platform_id: {platform}\n"
            f"session_id: {used_session}\n"
            f"读取: {stats.total}\n"
            f"写入: {stats.imported}\n"
            f"跳过: {stats.skipped}\n"
            f"重复: {stats.duplicate}\n"
            f"失败: {stats.failed}"
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
        query_text = query or event.message_str
        examples = await self.retriever.retrieve(
            user_id,
            query_text,
            top_k=8,
            profile=profile,
            query_tag=self.classifier.classify(query_text),
        )
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

    def _resolve_import_path(self, file_path: str) -> Path:
        text = file_path.strip().strip('"').strip("'")
        path = Path(text).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (Path(__file__).resolve().parent / path).resolve()

    def _history_session_candidates(self, event: AstrMessageEvent, override: str = "") -> list[str]:
        candidates = [
            override.strip(),
            event.get_session_id(),
            event.get_group_id(),
        ]
        seen: set[str] = set()
        result: list[str] = []
        for item in candidates:
            item = str(item or "").strip()
            if item and item not in seen:
                seen.add(item)
                result.append(item)
        return result

    async def _fetch_history_records(
        self,
        manager,
        platform_id: str,
        session_id: str,
        target_user_id: str,
        limit: int,
    ) -> list[ImportedMessage]:
        records: list[ImportedMessage] = []
        page = 1
        page_size = min(200, limit)
        while len(records) < limit:
            history_items = await manager.get(platform_id, session_id, page=page, page_size=page_size)
            if not history_items:
                break
            for item in history_items:
                content = getattr(item, "content", {}) or {}
                if isinstance(content, dict) and content.get("type") == "bot":
                    continue
                sender_id = str(getattr(item, "sender_id", "") or "")
                if sender_id and sender_id != target_user_id:
                    continue
                text = extract_history_text(content)
                if not text:
                    continue
                created_at = getattr(item, "created_at", None)
                timestamp = int(created_at.timestamp()) if hasattr(created_at, "timestamp") else int(time.time())
                records.append(
                    ImportedMessage(
                        text=text,
                        timestamp=timestamp,
                        sender_id=sender_id,
                        sender_name=str(getattr(item, "sender_name", "") or ""),
                        sender_keys=tuple(
                            value
                            for value in (
                                sender_id,
                                str(getattr(item, "sender_name", "") or ""),
                            )
                            if value
                        ),
                    )
                )
                if len(records) >= limit:
                    break
            if len(history_items) < page_size:
                break
            page += 1
        records.sort(key=lambda record: record.timestamp)
        return records

    async def _learn_imported_records(
        self,
        user_id: str,
        records: list[ImportedMessage],
        *,
        umo: str = "",
    ) -> ImportStats:
        stats = ImportStats(total=len(records))
        async for update in self._learn_imported_records_with_progress(
            user_id,
            records,
            umo=umo,
        ):
            if isinstance(update, ImportStats):
                stats = update
        return stats

    async def _learn_imported_records_with_progress(
        self,
        user_id: str,
        records: list[ImportedMessage],
        *,
        umo: str = "",
    ):
        stats = ImportStats(total=len(records))
        await self.store.init_user(user_id)
        prepared = []
        seen_normalized: set[str] = set()
        duplicate_window = int(cfg(self.config, "import_duplicate_window", 200))
        should_filter_sender = any(self._imported_sender_keys(record) for record in records)

        for record in records:
            if should_filter_sender and user_id not in self._imported_sender_keys(record):
                stats.skipped += 1
                continue
            clean = self.cleaner.clean(record.text)
            if not clean.accepted:
                stats.skipped += 1
                continue
            if clean.normalized in seen_normalized:
                stats.duplicate += 1
                continue
            if await self.store.is_duplicate_recent(user_id, clean.normalized, duplicate_window):
                stats.duplicate += 1
                continue
            seen_normalized.add(clean.normalized)
            semantic_tag = self.classifier.classify(clean.normalized)
            style_profile = self.style_analyzer.analyze(clean.normalized, semantic_tag)
            prepared.append((record, clean.normalized, semantic_tag, style_profile))

        yield (
            "[VibeDuplicate] 清洗完成，开始生成 embedding 并写入数据库。\n"
            f"读取: {stats.total}\n"
            f"待写入: {len(prepared)}\n"
            f"已跳过: {stats.skipped}\n"
            f"重复: {stats.duplicate}"
        )

        batch_size = max(1, int(cfg(self.config, "import_batch_size", 64)))
        progress_step = max(batch_size, 1000)
        next_progress = progress_step
        for start in range(0, len(prepared), batch_size):
            chunk = prepared[start : start + batch_size]
            try:
                embeddings = await self.embedding_provider.embed_many([item[1] for item in chunk])
            except Exception as exc:
                stats.failed += len(chunk)
                logger.error("[VibeDuplicate] import embedding failed: %s", exc)
                continue

            for (record, normalized, semantic_tag, style_profile), embedding in zip(chunk, embeddings):
                try:
                    await self.store.add_message(
                        PendingMessage(
                            user_id=user_id,
                            message=record.text.strip(),
                            normalized_message=normalized,
                            timestamp=record.timestamp,
                            semantic_tag=semantic_tag,
                            message_embedding=embedding.vector,
                            embedding_model=embedding.model,
                            style_vector=style_profile.vector,
                            quality_score=style_profile.quality_score,
                        )
                    )
                    stats.imported += 1
                except Exception as exc:
                    stats.failed += 1
                    logger.error("[VibeDuplicate] import write failed: %s", exc)

            processed = start + len(chunk)
            if processed >= next_progress and processed < len(prepared):
                yield (
                    "[VibeDuplicate] 导入中...\n"
                    f"进度: {processed}/{len(prepared)}\n"
                    f"已写入: {stats.imported}\n"
                    f"失败: {stats.failed}"
                )
                next_progress += progress_step

        if stats.imported:
            yield (
                "[VibeDuplicate] 消息写入完成，正在更新 persona。\n"
                f"已写入: {stats.imported}"
            )
            await self.persona_updater.update_persona_if_needed(user_id, force=True, umo=umo)
        yield stats

    def _imported_sender_keys(self, record: ImportedMessage) -> set[str]:
        return {
            str(value).strip()
            for value in (*record.sender_keys, record.sender_id, record.sender_name)
            if str(value).strip()
        }

    async def terminate(self):
        if self.writer_task:
            await self.write_queue.put(None)
            await self.write_queue.join()
            self.writer_task.cancel()
        await self.embedding_queue.stop()
        for task in list(self.update_tasks):
            task.cancel()
        logger.info("[VibeDuplicate] terminated")

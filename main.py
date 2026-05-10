# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
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
from .planner import ResponsePlanner
from .prompting import build_two_stage_avatar_prompt
from .rag import RagRetriever
from .runtime_guard import internal_llm_call_active
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
        self.planner = ResponsePlanner(self.classifier)
        self.style_analyzer = StyleAnalyzer()
        self.retriever = RagRetriever(
            self.store,
            self.embedding_provider,
            cache_ttl_seconds=int(cfg(config, "retrieval_cache_ttl_seconds", 60)),
            scan_limit=int(cfg(config, "retrieval_scan_limit", 1000)),
            recall_k=int(cfg(config, "retrieval_recall_k", 50)),
            allow_cross_model_retrieval_fallback=bool(
                cfg(config, "allow_cross_model_retrieval_fallback", False),
            ),
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
        if internal_llm_call_active():
            return
        if not bool(cfg(self.config, "enable_prompt_injection", True)):
            return
        user_id = self._select_avatar_user(event)
        if not user_id:
            return

        current_context = self._current_context(event, request)
        profile = await self.store.get_generated_profile(user_id)
        annotations = await self.store.notes(user_id, "admin_annotations", 20)
        memories = await self.store.notes(user_id, "third_party_memories", 20)
        planner_provider = self._planner_provider(event)
        reply_plan = await self.planner.build_plan(
            current_context=current_context,
            prompt=request.prompt,
            contexts=request.contexts or [],
            persona_summary=profile.persona_summary if profile else "",
            admin_annotations=annotations,
            provider=planner_provider,
        )
        if not reply_plan.should_reply:
            mode = self._planner_no_reply_mode()
            logger.debug("[VibeDuplicate] planner suggested no reply for %s; mode=%s", user_id, mode)
            if mode == "ignore":
                return

        examples = await self.retriever.retrieve_for_style(
            user_id,
            reply_plan.style_query,
            top_k=int(cfg(self.config, "rag_top_k", 8)),
            profile=profile,
            target_style_tag=reply_plan.target_style_tag,
        )
        recent = await self.store.recent_messages(
            user_id,
            int(cfg(self.config, "recent_user_context", 12)),
        )
        avatar_prompt = build_two_stage_avatar_prompt(
            user_id=user_id,
            reply_plan=reply_plan,
            profile=profile,
            style_examples=examples,
            current_context=current_context,
            admin_annotations=annotations,
            third_party_memories=memories,
            recent_messages=recent,
            no_reply_mode=self._planner_no_reply_mode(),
        )
        old_system_prompt = (getattr(request, "system_prompt", "") or "").strip()
        request.system_prompt = "\n\n".join(
            part for part in (old_system_prompt, avatar_prompt) if part
        )

    def _planner_provider(self, event: AstrMessageEvent):
        if not bool(cfg(self.config, "enable_response_planner_llm", True)):
            return None
        try:
            return self.context.get_using_provider(event.unified_msg_origin)
        except Exception as exc:
            logger.warning("[VibeDuplicate] planner provider unavailable: %s", exc)
            return None

    def _planner_no_reply_mode(self) -> str:
        mode = str(cfg(self.config, "planner_no_reply_mode", "brief_ack") or "brief_ack").strip().lower()
        return mode if mode in {"brief_ack", "empty", "ignore"} else "brief_ack"

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
            if isinstance(ctx, dict):
                role = ctx.get("role", "unknown")
                content = ctx.get("content", "")
            else:
                role = getattr(ctx, "role", "unknown")
                content = getattr(ctx, "content", "")
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
    @duplicate_group.command("embedding_status")
    async def embedding_status(self, event: AstrMessageEvent, user_id: str):
        status = await self.store.embedding_status(user_id, self.embedding_provider.model_name)
        yield event.plain_result(
            "[VibeDuplicate embedding status]\n"
            f"user_id: {user_id}\n"
            f"total messages: {status['total']}\n"
            f"ready embeddings: {status['ready_embeddings']}\n"
            f"current_model_count: {status['current_model_count']}\n"
            f"other_model_count: {status['other_model_count']}\n"
            f"missing_embedding_count: {status['missing_embedding_count']}\n"
            f"current provider model_name: {status['current_provider_model']}"
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @duplicate_group.command("reembed")
    async def reembed(self, event: AstrMessageEvent, user_id: str, limit: int = 500):
        limit = max(1, int(limit))
        messages = await self.store.messages_needing_reembed(
            user_id,
            self.embedding_provider.model_name,
            limit=limit,
        )
        yield event.plain_result(
            "[VibeDuplicate] 开始重建 embedding。\n"
            f"user_id: {user_id}\n"
            f"current_model: {self.embedding_provider.model_name}\n"
            f"limit: {limit}\n"
            f"待处理: {len(messages)}"
        )
        async for update in self._reembed_messages_with_progress(user_id, messages):
            yield event.plain_result(update)

    @filter.permission_type(filter.PermissionType.ADMIN)
    @duplicate_group.command("reembed_all")
    async def reembed_all(self, event: AstrMessageEvent, user_id: str):
        messages = await self.store.messages_needing_reembed(
            user_id,
            self.embedding_provider.model_name,
            limit=None,
        )
        yield event.plain_result(
            "[VibeDuplicate] 开始重建全部旧/缺失 embedding。\n"
            f"user_id: {user_id}\n"
            f"current_model: {self.embedding_provider.model_name}\n"
            f"待处理: {len(messages)}"
        )
        async for update in self._reembed_messages_with_progress(user_id, messages):
            yield event.plain_result(update)

    @filter.permission_type(filter.PermissionType.ADMIN)
    @duplicate_group.command("debug_retrieval")
    async def debug_retrieval(self, event: AstrMessageEvent, user_id: str, *, query: str):
        profile = await self.store.get_generated_profile(user_id)
        annotations = await self.store.notes(user_id, "admin_annotations", 20)
        memories = await self.store.notes(user_id, "third_party_memories", 20)
        provider = self._planner_provider(event)
        plan = await self.planner.build_plan(
            current_context=query,
            prompt=query,
            contexts=[],
            persona_summary=profile.persona_summary if profile else "",
            admin_annotations=annotations,
            provider=provider,
        )
        status = await self.store.embedding_status(user_id, self.embedding_provider.model_name)
        examples = await self.retriever.retrieve_for_style(
            user_id,
            plan.style_query,
            target_style_tag=plan.target_style_tag,
            profile=profile,
            top_k=int(cfg(self.config, "rag_top_k", 8)),
        )
        recent = await self.store.recent_messages(user_id, int(cfg(self.config, "recent_user_context", 12)))
        prompt = build_two_stage_avatar_prompt(
            user_id=user_id,
            reply_plan=plan,
            profile=profile,
            style_examples=examples,
            current_context=query,
            admin_annotations=annotations,
            third_party_memories=memories,
            recent_messages=recent,
            no_reply_mode=self._planner_no_reply_mode(),
        )
        fallback_happened = any(item.retrieval_fallback for item in examples)
        suggestions = self._debug_suggestions(status, examples, plan)
        lines = [
            "[VibeDuplicate debug retrieval]",
            f"provider_model: {self.embedding_provider.model_name}",
            f"current_model_count: {status['current_model_count']}",
            f"other_model_count: {status['other_model_count']}",
            f"missing_embedding_count: {status['missing_embedding_count']}",
            f"allow_cross_model_retrieval_fallback: {bool(cfg(self.config, 'allow_cross_model_retrieval_fallback', False))}",
            f"retrieval_fallback_happened: {fallback_happened}",
            f"rough_prompt_tokens: {self._rough_tokens(prompt)}",
            f"ReplyPlan: {json.dumps(asdict(plan), ensure_ascii=False)}",
            f"style_query: {plan.style_query}",
            f"target_style_tag: {plan.target_style_tag}",
            "examples:",
        ]
        if fallback_happened:
            lines.append("WARNING: 已使用跨模型检索 fallback，建议切换正式模型后执行 /duplicate reembed_all。")
        for index, item in enumerate(examples, 1):
            lines.append(
                f"{index}. final={item.score:.3f} semantic={item.semantic_score:.3f} "
                f"style={item.style_match_score:.3f} quality={item.quality_score:.3f} "
                f"model={item.embedding_model or 'unknown'} fallback={item.retrieval_fallback} "
                f"tag={item.semantic_tag} :: {item.message[:160]}"
            )
        lines.append("suggestions:")
        lines.extend(f"- {item}" for item in suggestions)
        yield event.plain_result("\n".join(lines)[:3500])

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
        annotations = await self.store.notes(user_id, "admin_annotations", 20)
        memories = await self.store.notes(user_id, "third_party_memories", 20)
        plan = await self.planner.build_plan(
            current_context=query_text,
            prompt=query_text,
            contexts=[],
            persona_summary=profile.persona_summary if profile else "",
            admin_annotations=annotations,
            provider=self._planner_provider(event),
        )
        examples = await self.retriever.retrieve_for_style(
            user_id,
            plan.style_query,
            top_k=8,
            profile=profile,
            target_style_tag=plan.target_style_tag,
        )
        recent = await self.store.recent_messages(user_id, 12)
        prompt = build_two_stage_avatar_prompt(
            user_id=user_id,
            reply_plan=plan,
            profile=profile,
            style_examples=examples,
            current_context=query_text,
            admin_annotations=annotations,
            third_party_memories=memories,
            recent_messages=recent,
            no_reply_mode=self._planner_no_reply_mode(),
        )
        header = (
            "[VibeDuplicate prompt preview]\n"
            f"planner_source: {plan.planner_source}\n"
            f"planner_no_reply_mode: {self._planner_no_reply_mode()}\n"
            f"rough_prompt_tokens: {self._rough_tokens(prompt)}\n\n"
        )
        yield event.plain_result((header + prompt)[:3500])

    @filter.permission_type(filter.PermissionType.ADMIN)
    @duplicate_group.command("eval")
    async def eval_harness(self, event: AstrMessageEvent, user_id: str, *, file_path: str):
        path = self._resolve_import_path(file_path)
        if not path.exists() or not path.is_file():
            yield event.plain_result(f"找不到 eval JSONL 文件：{path}")
            return

        try:
            cases = await asyncio.to_thread(self._load_eval_cases, path)
        except Exception as exc:
            yield event.plain_result(f"eval 文件解析失败：{exc}")
            return

        yield event.plain_result(
            "[VibeDuplicate eval] 开始离线评估。\n"
            f"user_id: {user_id}\n"
            f"file: {path}\n"
            f"samples: {len(cases)}"
        )
        profile = await self.store.get_generated_profile(user_id)
        annotations = await self.store.notes(user_id, "admin_annotations", 20)
        memories = await self.store.notes(user_id, "third_party_memories", 20)
        recent = await self.store.recent_messages(user_id, int(cfg(self.config, "recent_user_context", 12)))
        provider = self._planner_provider(event)

        debug_rows = []
        semantic_scores: list[float] = []
        style_scores: list[float] = []
        intent_hits = 0
        style_tag_hits = 0
        fallback_count = 0
        empty_examples = 0

        for index, case in enumerate(cases, 1):
            context = str(case.get("context") or "")
            expected_intent = str(case.get("expected_intent") or "")
            expected_style_tag = str(case.get("expected_style_tag") or "")
            plan = await self.planner.build_plan(
                current_context=context,
                prompt=context,
                contexts=[],
                persona_summary=profile.persona_summary if profile else "",
                admin_annotations=annotations,
                provider=provider,
            )
            examples = await self.retriever.retrieve_for_style(
                user_id,
                plan.style_query,
                top_k=int(cfg(self.config, "rag_top_k", 8)),
                profile=profile,
                target_style_tag=plan.target_style_tag,
            )
            prompt = build_two_stage_avatar_prompt(
                user_id=user_id,
                reply_plan=plan,
                profile=profile,
                style_examples=examples,
                current_context=context,
                admin_annotations=annotations,
                third_party_memories=memories,
                recent_messages=recent,
                no_reply_mode=self._planner_no_reply_mode(),
            )
            if expected_intent and plan.reply_intent == expected_intent:
                intent_hits += 1
            if expected_style_tag and plan.target_style_tag == expected_style_tag:
                style_tag_hits += 1
            if not examples:
                empty_examples += 1
            if any(item.retrieval_fallback for item in examples):
                fallback_count += 1
            if examples:
                semantic_scores.append(sum(item.semantic_score for item in examples) / len(examples))
                style_scores.append(sum(item.style_match_score for item in examples) / len(examples))
            debug_rows.append(
                {
                    "index": index,
                    "expected_intent": expected_intent,
                    "actual_intent": plan.reply_intent,
                    "expected_style_tag": expected_style_tag,
                    "actual_style_tag": plan.target_style_tag,
                    "planner_source": plan.planner_source,
                    "context_emotion": plan.context_emotion,
                    "would_target_reply": plan.would_target_reply,
                    "examples": len(examples),
                    "fallback": any(item.retrieval_fallback for item in examples),
                    "rough_prompt_tokens": self._rough_tokens(prompt),
                }
            )

        total = len(cases)
        report = {
            "user_id": user_id,
            "file": str(path),
            "total": total,
            "intent_hit_rate": intent_hits / total if total else 0,
            "style_tag_hit_rate": style_tag_hits / total if total else 0,
            "retrieval_fallback_count": fallback_count,
            "empty_examples_count": empty_examples,
            "avg_semantic_score": sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0,
            "avg_style_match_score": sum(style_scores) / len(style_scores) if style_scores else 0,
            "debug": debug_rows,
        }
        report_path = await asyncio.to_thread(self._save_eval_report, user_id, report)
        yield event.plain_result(
            "[VibeDuplicate eval 完成]\n"
            f"total: {total}\n"
            f"intent_hit_rate: {report['intent_hit_rate']:.2%}\n"
            f"style_tag_hit_rate: {report['style_tag_hit_rate']:.2%}\n"
            f"retrieval_fallback_count: {fallback_count}\n"
            f"empty_examples_count: {empty_examples}\n"
            f"avg_semantic_score: {report['avg_semantic_score']:.3f}\n"
            f"avg_style_match_score: {report['avg_style_match_score']:.3f}\n"
            f"report: {report_path}"
        )

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

    def _load_eval_cases(self, path: Path) -> list[dict]:
        cases = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_no} is not a JSON object")
            cases.append(value)
        return cases

    def _save_eval_report(self, user_id: str, report: dict) -> Path:
        safe_user_id = "".join(ch for ch in user_id if ch.isalnum() or ch in "-_@.") or "unknown"
        report_dir = DATA_ROOT / "eval_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / f"eval_{safe_user_id}_{int(time.time())}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _rough_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def _debug_suggestions(self, status: dict, examples: list, plan) -> list[str]:
        suggestions = []
        current_count = int(status.get("current_model_count", 0) or 0)
        if current_count < max(int(cfg(self.config, "rag_top_k", 8)), 20):
            suggestions.append("current_model_count 太少：运行 /duplicate reembed_all <user_id>。")
        if not examples:
            suggestions.append("examples 为空：检查 embedding provider、降低质量过滤预期，或先导入更多目标用户文本。")
        if getattr(plan, "planner_source", "") == "fallback":
            suggestions.append("planner_source=fallback：检查当前聊天 provider 是否可用，或关闭/修复 planner LLM。")
        if any(getattr(item, "retrieval_fallback", False) for item in examples):
            suggestions.append("已触发跨模型 fallback：正式使用建议关闭 fallback 并 reembed_all。")
        return suggestions or ["当前检索链路没有明显异常。"]

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

    async def _reembed_messages_with_progress(self, user_id: str, messages):
        if not messages:
            await self.store.clear_retrieval_cache(user_id)
            yield "[VibeDuplicate] 没有需要重建的 embedding，已清空检索缓存。"
            return

        batch_size = max(1, int(cfg(self.config, "reembed_batch_size", cfg(self.config, "import_batch_size", 64))))
        processed = 0
        updated = 0
        failed = 0
        for start in range(0, len(messages), batch_size):
            chunk = messages[start : start + batch_size]
            texts = [item.normalized_message or item.message for item in chunk]
            try:
                embeddings = await self.embedding_provider.embed_many(texts)
            except Exception as exc:
                failed += len(chunk)
                processed += len(chunk)
                logger.error("[VibeDuplicate] reembed batch failed: %s", exc)
                yield (
                    "[VibeDuplicate] reembed 批次失败。\n"
                    f"进度: {processed}/{len(messages)}\n"
                    f"updated: {updated}\n"
                    f"failed: {failed}\n"
                    f"error: {exc}"
                )
                continue

            for item, embedding in zip(chunk, embeddings):
                try:
                    text = item.normalized_message or item.message
                    semantic_tag = self.classifier.classify(text)
                    style_profile = self.style_analyzer.analyze(text, semantic_tag)
                    await self.store.update_message_embedding(
                        user_id=user_id,
                        message_id=item.id,
                        embedding=embedding.vector,
                        embedding_model=embedding.model,
                        style_vector=style_profile.vector,
                        quality_score=style_profile.quality_score,
                        semantic_tag=semantic_tag,
                    )
                    updated += 1
                except Exception as exc:
                    failed += 1
                    logger.error("[VibeDuplicate] reembed row failed: %s", exc)
            processed += len(chunk)
            yield (
                "[VibeDuplicate] reembed 进行中...\n"
                f"进度: {processed}/{len(messages)}\n"
                f"updated: {updated}\n"
                f"failed: {failed}"
            )

        await self.store.clear_retrieval_cache(user_id)
        yield (
            "[VibeDuplicate] reembed 完成，已清空 retrieval_cache。\n"
            f"user_id: {user_id}\n"
            f"updated: {updated}\n"
            f"failed: {failed}\n"
            f"current_model: {self.embedding_provider.model_name}"
        )

    async def terminate(self):
        if self.writer_task:
            await self.write_queue.put(None)
            await self.write_queue.join()
            self.writer_task.cancel()
        await self.embedding_queue.stop()
        for task in list(self.update_tasks):
            task.cancel()
        logger.info("[VibeDuplicate] terminated")

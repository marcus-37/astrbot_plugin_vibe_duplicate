from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from .models import GeneratedProfile, PendingMessage, StoredMessage


SCHEMA_VERSION = 2


class AvatarStore:
    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self.user_data_dir = data_root / "user_data"
        self.user_data_dir.mkdir(parents=True, exist_ok=True)

    def db_path(self, user_id: str) -> Path:
        safe_user_id = "".join(ch for ch in user_id if ch.isalnum() or ch in "-_@.")
        return self.user_data_dir / f"{safe_user_id or 'unknown'}.db"

    async def init_user(self, user_id: str) -> None:
        await asyncio.to_thread(self._init_db, self.db_path(user_id))

    def _connect(self, db_path: Path) -> sqlite3.Connection:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_db(self, db_path: Path) -> None:
        with closing(self._connect(db_path)) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS profile (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );

                CREATE TABLE IF NOT EXISTS admin_annotations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    added_by TEXT NOT NULL,
                    timestamp INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS third_party_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    added_by TEXT NOT NULL,
                    timestamp INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS generated_profile (
                    user_id TEXT PRIMARY KEY,
                    persona_summary TEXT NOT NULL,
                    updated_at INTEGER NOT NULL,
                    message_count INTEGER NOT NULL,
                    persona_version INTEGER NOT NULL DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS persona_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    persona_summary TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    message_count INTEGER NOT NULL,
                    persona_version INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS retrieval_cache (
                    cache_key TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            self._ensure_column(conn, "chat_history", "normalized_message", "TEXT")
            self._ensure_column(conn, "chat_history", "message_embedding", "TEXT")
            self._ensure_column(conn, "chat_history", "embedding_model", "TEXT")
            self._ensure_column(conn, "chat_history", "semantic_tag", "TEXT")
            self._ensure_column(conn, "chat_history", "quality_score", "REAL DEFAULT 1.0")
            self._ensure_column(conn, "generated_profile", "persona_version", "INTEGER NOT NULL DEFAULT 1")
            conn.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_timestamp ON chat_history(timestamp);
                CREATE INDEX IF NOT EXISTS idx_chat_semantic_tag ON chat_history(semantic_tag);
                CREATE INDEX IF NOT EXISTS idx_chat_embedding_model ON chat_history(embedding_model);
                INSERT OR REPLACE INTO schema_meta(key, value)
                VALUES('schema_version', '2');
                """
            )

    def _ensure_column(
        self,
        conn: sqlite3.Connection,
        table: str,
        column: str,
        column_type: str,
    ) -> None:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    async def add_message(self, pending: PendingMessage) -> int:
        return await asyncio.to_thread(self._add_message, pending)

    def _add_message(self, pending: PendingMessage) -> int:
        db_path = self.db_path(pending.user_id)
        self._init_db(db_path)
        with closing(self._connect(db_path)) as conn:
            cursor = conn.execute(
                """
                INSERT INTO chat_history (
                    user_id, message, normalized_message, timestamp,
                    message_embedding, embedding_model, semantic_tag, quality_score
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pending.user_id,
                    pending.message,
                    pending.normalized_message,
                    pending.timestamp,
                    json.dumps(pending.message_embedding, ensure_ascii=False),
                    pending.embedding_model,
                    pending.semantic_tag,
                    1.0,
                ),
            )
            return int(cursor.lastrowid)

    async def is_duplicate_recent(self, user_id: str, normalized: str, window: int = 8) -> bool:
        return await asyncio.to_thread(self._is_duplicate_recent, user_id, normalized, window)

    def _is_duplicate_recent(self, user_id: str, normalized: str, window: int) -> bool:
        db_path = self.db_path(user_id)
        self._init_db(db_path)
        with closing(self._connect(db_path)) as conn:
            rows = conn.execute(
                """
                SELECT normalized_message, message
                FROM chat_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (window,),
            ).fetchall()
        values = [(row["normalized_message"] or row["message"] or "") for row in rows]
        return normalized in values

    async def message_count(self, user_id: str) -> int:
        return await asyncio.to_thread(self._message_count, user_id)

    def _message_count(self, user_id: str) -> int:
        db_path = self.db_path(user_id)
        self._init_db(db_path)
        with closing(self._connect(db_path)) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM chat_history").fetchone()[0])

    async def get_generated_profile(self, user_id: str) -> GeneratedProfile | None:
        return await asyncio.to_thread(self._get_generated_profile, user_id)

    def _get_generated_profile(self, user_id: str) -> GeneratedProfile | None:
        db_path = self.db_path(user_id)
        self._init_db(db_path)
        with closing(self._connect(db_path)) as conn:
            row = conn.execute(
                """
                SELECT user_id, persona_summary, updated_at, message_count, persona_version
                FROM generated_profile
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
        if not row:
            return None
        return GeneratedProfile(
            user_id=row["user_id"],
            persona_summary=row["persona_summary"],
            updated_at=int(row["updated_at"]),
            message_count=int(row["message_count"]),
            persona_version=int(row["persona_version"] or 1),
        )

    async def upsert_generated_profile(
        self,
        user_id: str,
        persona_summary: str,
        message_count: int,
    ) -> GeneratedProfile:
        return await asyncio.to_thread(
            self._upsert_generated_profile,
            user_id,
            persona_summary,
            message_count,
        )

    def _upsert_generated_profile(
        self,
        user_id: str,
        persona_summary: str,
        message_count: int,
    ) -> GeneratedProfile:
        db_path = self.db_path(user_id)
        self._init_db(db_path)
        now = int(time.time())
        with closing(self._connect(db_path)) as conn:
            old = conn.execute(
                "SELECT * FROM generated_profile WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            next_version = 1
            if old:
                next_version = int(old["persona_version"] or 1) + 1
                conn.execute(
                    """
                    INSERT INTO persona_versions(
                        user_id, persona_summary, created_at,
                        message_count, persona_version
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        old["persona_summary"],
                        int(old["updated_at"]),
                        int(old["message_count"]),
                        int(old["persona_version"] or 1),
                    ),
                )
            conn.execute(
                """
                INSERT INTO generated_profile(
                    user_id, persona_summary, updated_at, message_count, persona_version
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    persona_summary = excluded.persona_summary,
                    updated_at = excluded.updated_at,
                    message_count = excluded.message_count,
                    persona_version = excluded.persona_version
                """,
                (user_id, persona_summary, now, message_count, next_version),
            )
        return GeneratedProfile(user_id, persona_summary, now, message_count, next_version)

    async def rollback_generated_profile(self, user_id: str) -> GeneratedProfile | None:
        return await asyncio.to_thread(self._rollback_generated_profile, user_id)

    def _rollback_generated_profile(self, user_id: str) -> GeneratedProfile | None:
        db_path = self.db_path(user_id)
        self._init_db(db_path)
        with closing(self._connect(db_path)) as conn:
            row = conn.execute(
                """
                SELECT * FROM persona_versions
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            if not row:
                return None
            conn.execute(
                """
                INSERT INTO generated_profile(
                    user_id, persona_summary, updated_at, message_count, persona_version
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    persona_summary = excluded.persona_summary,
                    updated_at = excluded.updated_at,
                    message_count = excluded.message_count,
                    persona_version = excluded.persona_version
                """,
                (
                    user_id,
                    row["persona_summary"],
                    int(time.time()),
                    int(row["message_count"]),
                    int(row["persona_version"]),
                ),
            )
            conn.execute("DELETE FROM persona_versions WHERE id = ?", (row["id"],))
        return GeneratedProfile(
            user_id=user_id,
            persona_summary=row["persona_summary"],
            updated_at=int(time.time()),
            message_count=int(row["message_count"]),
            persona_version=int(row["persona_version"]),
        )

    async def recent_messages(self, user_id: str, limit: int = 50) -> list[StoredMessage]:
        return await asyncio.to_thread(self._recent_messages, user_id, limit)

    def _recent_messages(self, user_id: str, limit: int) -> list[StoredMessage]:
        db_path = self.db_path(user_id)
        self._init_db(db_path)
        with closing(self._connect(db_path)) as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM chat_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_message(row) for row in reversed(rows)]

    async def messages_for_retrieval(self, user_id: str, limit: int = 1000) -> list[StoredMessage]:
        return await asyncio.to_thread(self._messages_for_retrieval, user_id, limit)

    def _messages_for_retrieval(self, user_id: str, limit: int) -> list[StoredMessage]:
        db_path = self.db_path(user_id)
        self._init_db(db_path)
        with closing(self._connect(db_path)) as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM chat_history
                WHERE message_embedding IS NOT NULL AND message_embedding != ''
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_message(row) for row in rows]

    def _row_to_message(self, row: sqlite3.Row) -> StoredMessage:
        embedding: list[float] = []
        if row["message_embedding"]:
            try:
                embedding = json.loads(row["message_embedding"])
            except json.JSONDecodeError:
                embedding = []
        return StoredMessage(
            id=int(row["id"]),
            user_id=row["user_id"],
            message=row["message"],
            normalized_message=row["normalized_message"] or row["message"],
            timestamp=int(row["timestamp"]),
            semantic_tag=row["semantic_tag"] or "neutral",
            message_embedding=embedding,
            embedding_model=row["embedding_model"] or "",
        )

    async def profile_items(self, user_id: str) -> dict[str, str]:
        return await asyncio.to_thread(self._profile_items, user_id)

    def _profile_items(self, user_id: str) -> dict[str, str]:
        db_path = self.db_path(user_id)
        self._init_db(db_path)
        with closing(self._connect(db_path)) as conn:
            rows = conn.execute("SELECT key, value FROM profile ORDER BY key").fetchall()
        return {row["key"]: row["value"] for row in rows}

    async def set_profile_item(self, user_id: str, key: str, value: str) -> None:
        await asyncio.to_thread(self._set_profile_item, user_id, key, value)

    def _set_profile_item(self, user_id: str, key: str, value: str) -> None:
        db_path = self.db_path(user_id)
        self._init_db(db_path)
        with closing(self._connect(db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO profile(key, value) VALUES(?, ?)",
                (key, value),
            )

    async def add_annotation(self, user_id: str, text: str, added_by: str) -> None:
        await asyncio.to_thread(
            self._insert_note,
            user_id,
            "admin_annotations",
            text,
            added_by,
        )

    async def add_memory(self, user_id: str, text: str, added_by: str) -> None:
        await asyncio.to_thread(
            self._insert_note,
            user_id,
            "third_party_memories",
            text,
            added_by,
        )

    def _insert_note(self, user_id: str, table: str, text: str, added_by: str) -> None:
        db_path = self.db_path(user_id)
        self._init_db(db_path)
        with closing(self._connect(db_path)) as conn:
            conn.execute(
                f"INSERT INTO {table}(text, added_by, timestamp) VALUES(?, ?, ?)",
                (text, added_by, int(time.time())),
            )

    async def notes(self, user_id: str, table: str, limit: int = 20) -> list[str]:
        return await asyncio.to_thread(self._notes, user_id, table, limit)

    def _notes(self, user_id: str, table: str, limit: int) -> list[str]:
        db_path = self.db_path(user_id)
        self._init_db(db_path)
        with closing(self._connect(db_path)) as conn:
            rows = conn.execute(
                f"SELECT text FROM {table} ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [row["text"] for row in rows]

    async def clear_user(self, user_id: str) -> bool:
        path = self.db_path(user_id)
        if not path.exists():
            return False
        await asyncio.to_thread(path.unlink)
        for suffix in ("-wal", "-shm"):
            extra = Path(str(path) + suffix)
            if extra.exists():
                await asyncio.to_thread(extra.unlink)
        return True

    async def user_ids(self) -> list[str]:
        return await asyncio.to_thread(self._user_ids)

    def _user_ids(self) -> list[str]:
        return sorted(path.stem for path in self.user_data_dir.glob("*.db"))

    async def set_cache(self, user_id: str, key: str, payload: str) -> None:
        await asyncio.to_thread(self._set_cache, user_id, key, payload)

    def _set_cache(self, user_id: str, key: str, payload: str) -> None:
        db_path = self.db_path(user_id)
        self._init_db(db_path)
        with closing(self._connect(db_path)) as conn:
            conn.execute(
                """
                INSERT INTO retrieval_cache(cache_key, payload, created_at)
                VALUES(?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload = excluded.payload,
                    created_at = excluded.created_at
                """,
                (key, payload, int(time.time())),
            )

    async def get_cache(self, user_id: str, key: str, max_age_seconds: int) -> str | None:
        return await asyncio.to_thread(self._get_cache, user_id, key, max_age_seconds)

    def _get_cache(self, user_id: str, key: str, max_age_seconds: int) -> str | None:
        db_path = self.db_path(user_id)
        self._init_db(db_path)
        with closing(self._connect(db_path)) as conn:
            row = conn.execute(
                "SELECT payload, created_at FROM retrieval_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if not row:
            return None
        if int(time.time()) - int(row["created_at"]) > max_age_seconds:
            return None
        return row["payload"]

# Vibe Duplicate

AstrBot plugin for long-term persona learning and dynamic style imitation.

## Pipeline

Group chat message -> cleaning/filtering -> SQLite storage -> placeholder embedding -> persona update -> RAG retrieval -> dynamic prompt injection -> LLM reply.

## New Structure

- `main.py`: AstrBot hooks, commands, write queue, prompt injection.
- `storage.py`: SQLite schema, migrations, profile versions, retrieval cache.
- `cleaning.py`: command/spam/low-signal filtering and semantic style tags.
- `embeddings.py`: swappable embedding provider interface plus deterministic placeholder embedding.
- `rag.py`: top-k semantic retrieval over stored utterances.
- `persona.py`: incremental persona updater with consistency guard and rollback support.
- `prompting.py`: avatar prompt and persona-update prompt builders.

## Schema Highlights

Existing per-user databases are reused under `data/astrtbot_plugin_echo_avatar/user_data`.

`chat_history` is migrated with:

- `normalized_message`
- `message_embedding`
- `embedding_model`
- `semantic_tag`
- `quality_score`

New tables:

- `generated_profile(user_id, persona_summary, updated_at, message_count, persona_version)`
- `persona_versions(...)`
- `retrieval_cache(cache_key, payload, created_at)`
- `schema_meta(key, value)`

## Commands

- `/duplicate status`
- `/duplicate profile <user_id> <key> <value>`
- `/duplicate annotate <user_id> <text>`
- `/duplicate memory <user_id> <text>`
- `/duplicate update <user_id>`
- `/duplicate rollback <user_id>`
- `/duplicate prompt <user_id> [query]`
- `/duplicate preview <user_id>`
- `/duplicate clear <user_id>`

The legacy command group alias `/echo_avatar` is also registered.

## Useful Config Keys

- `target_users`: list of user IDs to learn from.
- `avatar_user_id`: explicit user ID to imitate when injecting prompts. If unset and there is one target user, that user is used.
- `enable_prompt_injection`: default `true`.
- `persona_update_threshold`: default `50`.
- `rag_top_k`: default `8`.
- `blacklist_words`: words that should prevent learning from a message.
- `filter_commands`: default `true`.

The embedding provider is intentionally isolated behind `EmbeddingProvider`, so a real AstrBot or remote embedding provider can replace `PlaceholderEmbeddingProvider` later without changing storage or RAG code.

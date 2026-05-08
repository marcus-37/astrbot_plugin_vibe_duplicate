# Vibe Duplicate

一个用于长期人格学习与动态风格模仿的 AstrBot 插件。

## 工作流程

群聊消息 -> 清洗 / 过滤 -> 写入 SQLite -> 生成 placeholder embedding -> 更新 persona -> RAG 检索 -> 动态注入 prompt -> LLM 回复。

## 新目录结构

- `main.py`：AstrBot 钩子、命令、写入队列、prompt 注入入口。
- `storage.py`：SQLite schema、迁移、persona 版本、检索缓存。
- `cleaning.py`：命令 / 垃圾内容 / 低信号消息过滤，以及语义风格标签。
- `embeddings.py`：可替换的 embedding provider 接口，以及确定性的 placeholder embedding。
- `rag.py`：基于已存发言的 top-k 语义检索。
- `persona.py`：增量 persona 更新器，带一致性保护和回滚支持。
- `prompting.py`：avatar prompt 与 persona 更新 prompt 构建器。

## 数据库结构重点

沿用原有的按用户分库设计，路径为 `data/astrtbot_plugin_echo_avatar/user_data`。

`chat_history` 会自动迁移并新增：

- `normalized_message`
- `message_embedding`
- `embedding_model`
- `semantic_tag`
- `quality_score`

新增表：

- `generated_profile(user_id, persona_summary, updated_at, message_count, persona_version)`
- `persona_versions(...)`
- `retrieval_cache(cache_key, payload, created_at)`
- `schema_meta(key, value)`

## 命令

- `/duplicate status`
- `/duplicate profile <user_id> <key> <value>`
- `/duplicate annotate <user_id> <text>`
- `/duplicate memory <user_id> <text>`
- `/duplicate update <user_id>`
- `/duplicate rollback <user_id>`
- `/duplicate prompt <user_id> [query]`
- `/duplicate preview <user_id>`
- `/duplicate clear <user_id>`

同时保留了旧命令组别名 `/echo_avatar`。

## 常用配置项

插件目录内已提供 `_conf_schema.json`。重载插件后，AstrBot 后台会自动生成并展示以下配置项。

- `target_users`：需要长期学习的目标用户 ID 列表。
- `avatar_user_id`：注入 prompt 时明确要模仿的用户 ID。如果不设置，且 `target_users` 里只有一个用户，就默认使用该用户。
- `enable_prompt_injection`：是否启用动态 prompt 注入，默认 `true`。
- `persona_update_threshold`：每累计多少条新消息触发一次 persona 更新，默认 `50`。
- `rag_top_k`：回复时检索多少条相似历史发言，默认 `8`。
- `blacklist_words`：出现这些词时不学习该消息。
- `filter_commands`：是否过滤命令类消息，默认 `true`。

embedding provider 被刻意隔离在 `EmbeddingProvider` 接口后面，因此后续可以把 `PlaceholderEmbeddingProvider` 替换成真实的 AstrBot embedding provider 或远程 embedding 服务，而不需要改动存储层和 RAG 代码。

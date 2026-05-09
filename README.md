# Vibe Duplicate

一个用于长期人格学习与动态风格模仿的 AstrBot 插件。

## 工作流程

群聊消息 -> 清洗 / 过滤 -> 写入 SQLite -> 生成 placeholder embedding -> 更新 persona -> RAG 检索 -> 动态注入 prompt -> LLM 回复。

## 新目录结构

- `main.py`：AstrBot 钩子、命令、写入队列、prompt 注入入口。
- `storage.py`：SQLite schema、迁移、persona 版本、检索缓存。
- `cleaning.py`：命令 / 垃圾内容 / 低信号消息过滤，以及语义风格标签。
- `embeddings.py`：可替换的 embedding provider 接口，以及确定性的 placeholder embedding。
- `style.py`：提取语气、节奏、标点、emoji、口癖、抽象/发病程度等 style_vector。
- `rag.py`：语义召回 + 多维 rerank 的人格检索。
- `persona.py`：增量 persona 更新器，带一致性保护和回滚支持。
- `prompting.py`：avatar prompt 与 persona 更新 prompt 构建器。
- `importer.py`：聊天记录导入与 AstrBot 历史消息文本提取。

## 数据库结构重点

沿用原有的按用户分库设计，路径为 `data/astrtbot_plugin_echo_avatar/user_data`。

`chat_history` 会自动迁移并新增：

- `normalized_message`
- `message_embedding`
- `embedding_model`
- `semantic_tag`
- `quality_score`
- `style_vector`
- `embedding_status`
- `embedded_at`

新增表：

- `generated_profile(user_id, persona_summary, updated_at, message_count, persona_version)`
- `persona_versions(...)`
- `retrieval_cache(cache_key, payload, created_at)`
- `schema_meta(key, value)`

## Embedding 与 RAG

`embedding_provider` 支持：

- `placeholder`：无依赖 fallback，只适合测试。
- `openai`：推荐 `text-embedding-3-small` 或更高模型。
- `gemini`：推荐 `gemini-embedding-exp-03-07`。
- `ollama`：本地部署推荐 `bge-m3`。
- `sentence_transformers`：本地 Python 模型，中文推荐 `BAAI/bge-m3`、`BAAI/bge-large-zh-v1.5`。
- `astrbot`：复用 AstrBot 已配置的 embedding provider。

检索流程：

1. 先用 semantic embedding 召回 top 50。
2. 再用多维分数 rerank。
3. 最终取 top 8 注入 prompt。

评分公式：

```text
final_score =
  semantic_score * 0.5
  + recency_score * 0.2
  + style_match_score * 0.2
  + semantic_tag_score * 0.1
```

随后再叠加 `quality_score`、persona 匹配、重复惩罚、时间多样性和反 spam 惩罚。

## 中文群聊建议

- 优先使用真实 embedding provider，不要长期使用 `placeholder`。
- 中文群聊推荐 `bge-m3`，对短句、梗、口癖的召回会明显好于 hash fallback。
- `target_users` 只放真正要学习的人，避免把无关群友混进同一个人格。
- `blacklist_words` 可以放广告词、机器人命令、无意义刷屏关键词。
- 用 `/duplicate annotate` 添加管理员批注，约束 persona 漂移。
- 用 `/duplicate rollback <user_id>` 回滚错误 persona。

## 避免 retrieval 污染人格

- 低质量消息会降低 `quality_score`，例如纯“哈哈”、“6”、“？”。
- RAG 会优先召回高辨识度、高情绪、高口癖、强风格发言。
- prompt 注入使用“风格蒸馏层”，不会只把历史原句粗暴拼进去。
- persona 更新是增量式，会保留旧 summary，并带一致性保护。

## 命令

- `/duplicate status`：查看插件状态、目标用户、已知用户库数量、写入队列和 prompt 注入开关。
- `/duplicate profile <user_id> <key> <value>`：手动写入或覆盖目标用户的人格资料字段。
- `/duplicate annotate <user_id> <text>`：添加管理员批注，用来约束人格总结和后续模仿方向。
- `/duplicate memory <user_id> <text>`：添加第三方记忆，作为补充信息参与 prompt 注入。
- `/duplicate update <user_id>`：立即强制更新目标用户的 persona 总结。
- `/duplicate import <user_id> <file_path>`：从导出的聊天记录文件导入目标用户历史发言。
- `/duplicate backfill <user_id> [limit] [platform_id] [session_id]`：从 AstrBot 已保存的平台消息历史中回填目标用户发言。
- `/duplicate rollback <user_id>`：回滚到上一版 persona 总结。
- `/duplicate prompt <user_id> [query]`：预览指定用户当前会被注入给 LLM 的模仿 prompt。
- `/duplicate preview <user_id>`：查看目标用户已学习消息数、persona 版本和最近标签概览。
- `/duplicate clear <user_id>`：清空指定用户的全部学习数据。

同时保留了旧命令组别名 `/echo_avatar`，作用与 `/duplicate` 相同。

## 快速导入聊天记录

有两种方式可以快速形成基础人格库：

1. 从 AstrBot 已保存的历史消息回填：

```text
/duplicate backfill <目标用户ID> [最多读取条数]
```

在群聊里执行时，插件会优先使用当前平台和当前群/会话，只导入 `sender_id` 等于目标用户 ID 的消息。例如：

```text
/duplicate backfill 123456789 1000
```

如果当前会话查不到历史，也可以手动指定：

```text
/duplicate backfill 123456789 1000 <platform_id> <session_id>
```

2. 从导出的聊天记录文件导入：

```text
/duplicate import <目标用户ID> <文件路径>
```

支持 `.txt`、`.json`、`.jsonl`、`.csv`。相对路径会按插件目录解析，绝对路径也可用。常见字段名会自动识别，例如 `message`、`text`、`content`、`message_str`、`sender_id`、`user_id`、`timestamp`、`created_at`。也支持 QQChatExporter V5 这类 `messages[].sender.uid / uin / name` + `content.text` 的导出结构；导入时可以用目标用户的 `uid`、QQ 号 `uin` 或名称作为 `<目标用户ID>`，插件会自动只导入匹配发送者的文本消息。

普通文本每行一条消息，也支持下面这种格式：

```text
2026-05-10 12:30:01 用户名: 这是一条历史消息
```

## 常用配置项

插件目录内已提供 `_conf_schema.json`。重载插件后，AstrBot 后台会自动生成并展示以下配置项。

- `target_users`：需要长期学习的目标用户 ID 列表。
- `avatar_user_id`：注入 prompt 时明确要模仿的用户 ID。如果不设置，且 `target_users` 里只有一个用户，就默认使用该用户。
- `enable_prompt_injection`：是否启用动态 prompt 注入，默认 `true`。
- `persona_update_threshold`：每累计多少条新消息触发一次 persona 更新，默认 `50`。
- `rag_top_k`：回复时检索多少条相似历史发言，默认 `8`。
- `retrieval_recall_k`：第一阶段语义召回数量，默认 `50`。
- `embedding_provider`：embedding provider 类型，默认 `placeholder`。
- `embedding_model`：embedding 模型名。
- `embedding_api_key`：远程 embedding 服务 API Key。
- `embedding_api_base`：远程或本地 embedding 服务地址。
- `import_batch_size`：导入聊天记录时每批生成多少条 embedding，默认 `64`。
- `import_duplicate_window`：导入时向最近多少条已学习消息检查重复，默认 `200`。
- `blacklist_words`：出现这些词时不学习该消息。
- `filter_commands`：是否过滤命令类消息，默认 `true`。

embedding provider 被刻意隔离在 `EmbeddingProvider` 接口后面，因此后续可以把 `PlaceholderEmbeddingProvider` 替换成真实的 AstrBot embedding provider 或远程 embedding 服务，而不需要改动存储层和 RAG 代码。

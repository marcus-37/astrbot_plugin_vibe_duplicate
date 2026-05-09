from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


TEXT_LINE_RE = re.compile(
    r"^\s*(?:\[?(?P<time>\d{4}[-/]\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{2}(?::\d{2})?)\]?\s+)?"
    r"(?:(?P<sender>[^:：]{1,48})[:：]\s*)?(?P<text>.+?)\s*$"
)
MEDIA_PLACEHOLDER_RE = re.compile(
    r"^(?:\[(?:图片|语音|视频|文件|转发消息|表情)[^\]]*\]|\[卡片消息:.*\])$"
)


@dataclass(slots=True)
class ImportedMessage:
    text: str
    timestamp: int
    sender_id: str = ""
    sender_name: str = ""
    sender_keys: tuple[str, ...] = ()


def load_chat_records(path: Path) -> list[ImportedMessage]:
    path = path.expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json(path)
    if suffix == ".jsonl":
        return _load_jsonl(path)
    if suffix == ".csv":
        return _load_csv(path)
    return _load_txt(path)


def extract_history_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, dict):
        return ""

    if isinstance(content.get("message_str"), str):
        return content["message_str"].strip()
    if isinstance(content.get("text"), str):
        return content["text"].strip()
    if isinstance(content.get("content"), str):
        return content["content"].strip()

    message = content.get("message")
    if isinstance(message, str):
        return message.strip()
    if isinstance(message, list):
        parts = [_part_text(part) for part in message]
        return "".join(part for part in parts if part).strip()
    return ""


def parse_timestamp(value: Any, default: int | None = None) -> int:
    fallback = default if default is not None else int(time.time())
    if value is None or value == "":
        return fallback
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp = timestamp / 1000
        return int(timestamp)
    text = str(value).strip()
    if not text:
        return fallback
    if text.isdigit():
        return parse_timestamp(int(text), fallback)
    normalized = text.replace("/", "-").replace("T", " ")
    for fmt, length in (
        ("%Y-%m-%d %H:%M:%S", 19),
        ("%Y-%m-%d %H:%M", 16),
        ("%Y-%m-%d", 10),
    ):
        try:
            return int(datetime.strptime(normalized[:length], fmt).timestamp())
        except ValueError:
            pass
    try:
        return int(datetime.fromisoformat(text).timestamp())
    except ValueError:
        return fallback


def _load_txt(path: Path) -> list[ImportedMessage]:
    now = int(time.time())
    records: list[ImportedMessage] = []
    for index, line in enumerate(_read_text(path).splitlines()):
        line = line.strip()
        if not line:
            continue
        match = TEXT_LINE_RE.match(line)
        if not match:
            records.append(ImportedMessage(line, now + index))
            continue
        text = match.group("text").strip()
        sender = (match.group("sender") or "").strip()
        timestamp = parse_timestamp(match.group("time"), now + index)
        records.append(ImportedMessage(text, timestamp, sender_id=sender, sender_name=sender))
    return records


def _load_json(path: Path) -> list[ImportedMessage]:
    payload = json.loads(_read_text(path))
    if isinstance(payload, dict):
        for key in ("messages", "data", "items", "records"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            payload = [payload]
    if not isinstance(payload, list):
        return []
    return [_record_from_mapping(item) for item in payload if isinstance(item, dict)]


def _load_jsonl(path: Path) -> list[ImportedMessage]:
    records: list[ImportedMessage] = []
    for line in _read_text(path).splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            records.append(_record_from_mapping(item))
    return records


def _load_csv(path: Path) -> list[ImportedMessage]:
    rows = csv.DictReader(_read_text(path).splitlines())
    return [_record_from_mapping(row) for row in rows]


def _record_from_mapping(item: dict[str, Any]) -> ImportedMessage:
    text = _first_text(item, ("message", "text", "content", "message_str", "raw_message"))
    sender_id, sender_name, sender_keys = _sender_identity(item)
    timestamp = parse_timestamp(_first_text(item, ("timestamp", "time", "created_at", "date")))
    return ImportedMessage(
        text=_normalize_exported_text(text),
        timestamp=timestamp,
        sender_id=sender_id,
        sender_name=sender_name,
        sender_keys=sender_keys,
    )


def _first_text(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            extracted = extract_history_text(value)
            if extracted:
                return extracted
            continue
        text = str(value)
        if text:
            return text
    return ""


def _sender_identity(item: dict[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    sender = item.get("sender")
    values: list[str] = []
    sender_name = ""
    sender_keys = item.get("sender_keys")
    if isinstance(sender_keys, list):
        values.extend(str(value).strip() for value in sender_keys if value is not None)
    if isinstance(sender, dict):
        for key in ("uid", "uin", "user_id", "sender_id", "qq", "id", "name", "nickname"):
            value = sender.get(key)
            if value is not None:
                values.append(str(value).strip())
        sender_name = str(sender.get("name") or sender.get("nickname") or "").strip()
    else:
        for key in ("sender_id", "user_id", "qq", "uid", "uin", "sender"):
            value = item.get(key)
            if value is not None:
                values.append(str(value).strip())
        sender_name = str(item.get("sender_name") or item.get("nickname") or item.get("name") or "").strip()

    if sender_name:
        values.append(sender_name)
    values = [value for value in values if value]
    sender_id = values[0] if values else ""
    return sender_id, sender_name, tuple(dict.fromkeys(values))


def _normalize_exported_text(text: str) -> str:
    text = (text or "").strip()
    if MEDIA_PLACEHOLDER_RE.fullmatch(text):
        return ""
    return text


def _part_text(part: Any) -> str:
    if isinstance(part, str):
        return part
    if not isinstance(part, dict):
        return ""
    part_type = str(part.get("type", "")).lower()
    if part_type in {"plain", "text"}:
        data = part.get("data")
        if isinstance(data, dict) and isinstance(data.get("text"), str):
            return data["text"]
        return str(part.get("text") or part.get("content") or "")
    if part_type == "reply":
        selected = str(part.get("selected_text") or "")
        return f"[引用:{selected}]" if selected else ""
    for key in ("text", "content", "message_str"):
        value = part.get(key)
        if isinstance(value, str):
            return value
    return ""


def _read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="replace")

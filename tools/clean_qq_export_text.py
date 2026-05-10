from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except UnicodeDecodeError:
            continue
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


PLACEHOLDER_RE = re.compile(r"^\[(?:图片|卡片消息|文件|视频|语音|表情|合并转发):.*\]$")


def clean_text(text: str, *, keep_placeholders: bool = False) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"^\[回复 [^\]]+\]\n?", "", text).strip()
    if not keep_placeholders and PLACEHOLDER_RE.match(text):
        return ""
    return text


def sender_matches(sender: dict[str, Any], target: str) -> bool:
    target = target.strip()
    return target in {
        str(sender.get("uin", "") or "").strip(),
        str(sender.get("uid", "") or "").strip(),
        str(sender.get("name", "") or "").strip(),
    }


def extract_texts(payload: dict[str, Any], target: str, *, keep_placeholders: bool = False) -> list[str]:
    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError("input JSON does not contain a messages list")

    texts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        sender = message.get("sender") or {}
        if not isinstance(sender, dict) or not sender_matches(sender, target):
            continue
        content = message.get("content") or {}
        if not isinstance(content, dict):
            continue
        text = clean_text(str(content.get("text", "") or ""), keep_placeholders=keep_placeholders)
        if text:
            texts.append(text)
    return texts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract plain content.text lines for one sender from QQChatExporter JSON.",
    )
    parser.add_argument("input", help="QQChatExporter chat.json path")
    parser.add_argument("qq", help="target QQ uin; uid/name also works if the export has no uin")
    parser.add_argument("-o", "--output", help="output txt path; defaults to stdout")
    parser.add_argument(
        "--keep-placeholders",
        action="store_true",
        help="keep media/card placeholders such as [图片: xxx.jpg]",
    )
    args = parser.parse_args()

    payload = read_json(Path(args.input))
    texts = extract_texts(payload, args.qq, keep_placeholders=args.keep_placeholders)
    output = "\n".join(texts)
    if output:
        output += "\n"

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

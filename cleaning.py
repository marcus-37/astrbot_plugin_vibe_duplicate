from __future__ import annotations

import re
from dataclasses import dataclass


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
SPACE_RE = re.compile(r"\s+")
REPEATED_CHAR_RE = re.compile(r"(.)\1{8,}")
COMMAND_PREFIXES = ("/", "!", "#", ".", "~", "?", "！", "。", "？")


@dataclass(slots=True)
class CleanResult:
    accepted: bool
    normalized: str = ""
    reason: str = ""


class MessageCleaner:
    def __init__(
        self,
        *,
        max_length: int = 1200,
        min_length: int = 2,
        blacklist_words: list[str] | None = None,
        filter_commands: bool = True,
    ) -> None:
        self.max_length = max_length
        self.min_length = min_length
        self.blacklist_words = [w.lower() for w in (blacklist_words or []) if w]
        self.filter_commands = filter_commands

    def clean(self, message: str) -> CleanResult:
        text = SPACE_RE.sub(" ", (message or "").strip())
        if not text:
            return CleanResult(False, reason="empty")
        if self.filter_commands and self._looks_like_command(text):
            return CleanResult(False, reason="command")
        if len(text) < self.min_length:
            return CleanResult(False, reason="too_short")
        if len(text) > self.max_length:
            return CleanResult(False, reason="too_long")
        if self._is_url_spam(text):
            return CleanResult(False, reason="url_spam")
        if REPEATED_CHAR_RE.search(text):
            return CleanResult(False, reason="repeated_chars")
        lowered = text.lower()
        if any(word in lowered for word in self.blacklist_words):
            return CleanResult(False, reason="blacklist")
        if self._mostly_punctuation(text):
            return CleanResult(False, reason="low_signal")
        return CleanResult(True, normalized=text)

    def _looks_like_command(self, text: str) -> bool:
        if text.startswith(COMMAND_PREFIXES):
            return True
        first = text.split(maxsplit=1)[0].lower()
        command_words = {
            "help",
            "start",
            "stop",
            "status",
            "config",
            "echo_avatar",
            "duplicate",
        }
        return len(text) <= 50 and first in command_words

    def _is_url_spam(self, text: str) -> bool:
        urls = URL_RE.findall(text)
        if len(urls) >= 2:
            return True
        if urls and len(URL_RE.sub("", text).strip()) < 8:
            return True
        return False

    def _mostly_punctuation(self, text: str) -> bool:
        signal = sum(1 for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
        return signal < max(2, len(text) // 5)


class StyleClassifier:
    TAGS = (
        "complaint",
        "banter",
        "serious_explain",
        "silly",
        "meme",
        "sarcasm",
        "neutral",
    )

    def classify(self, text: str) -> str:
        lowered = text.lower()
        if any(x in lowered for x in ("笑死", "草", "哈哈", "hhh", "233", "乐")):
            return "silly"
        if any(x in lowered for x in ("不是", "怎么", "离谱", "无语", "吐了", "烦")):
            return "complaint"
        if any(x in lowered for x in ("因为", "所以", "其实", "比如", "解释", "原理")):
            return "serious_explain"
        if any(x in lowered for x in ("梗", "图", "表情包", "meme", "kusa")):
            return "meme"
        if any(x in lowered for x in ("是吧", "呵呵", "你猜", "不会吧", "就这")):
            return "sarcasm"
        if any(x in lowered for x in ("?", "？", "吗", "呢", "咋")):
            return "banter"
        return "neutral"


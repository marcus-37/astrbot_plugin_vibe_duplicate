from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

from .embeddings import cosine_similarity


EMOJI_RE = re.compile(
    "[\U0001f300-\U0001faff\U00002700-\U000027bf]",
    re.UNICODE,
)
TEXT_FACE_RE = re.compile(r"(\([^\w\s]{1,4}\)|[xX]?D|orz|qwq|QAQ|ww+)")
CHINESE_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
PUNCT_RE = re.compile(r"[!?！？。.,，、~～…]+")
MEME_WORDS = (
    "笑死",
    "逆天",
    "典",
    "绷不住",
    "我超",
    "唐",
    "草",
    "乐",
    "蚌埠住",
    "急",
    "孝",
    "赢",
    "麻了",
    "抽象",
    "离谱",
    "kusa",
    "hhh",
    "233",
)
TONE_PARTICLES = (
    "啊",
    "吧",
    "呢",
    "嘛",
    "呗",
    "哈",
    "呀",
    "捏",
    "罢",
    "了",
)
SARCASM_MARKERS = ("不会吧", "就这", "你猜", "是吧", "呵呵", "典中典", "急了")
HIGH_EMOTION = ("笑死", "绷不住", "麻了", "无语", "离谱", "逆天", "我超")


@dataclass(slots=True)
class StyleProfile:
    vector: list[float]
    quality_score: float
    tag: str
    brief: str


class StyleAnalyzer:
    """Extracts non-semantic style features for Chinese group chat utterances."""

    def analyze(self, text: str, semantic_tag: str = "neutral") -> StyleProfile:
        text = (text or "").strip()
        length = len(text)
        chinese_count = len(CHINESE_CHAR_RE.findall(text))
        punct_count = len(PUNCT_RE.findall(text))
        emoji_count = len(EMOJI_RE.findall(text))
        face_count = len(TEXT_FACE_RE.findall(text))
        meme_count = sum(1 for word in MEME_WORDS if word.lower() in text.lower())
        particle_count = sum(text.count(word) for word in TONE_PARTICLES)
        sarcasm_count = sum(1 for word in SARCASM_MARKERS if word in text)
        emotion_count = sum(1 for word in HIGH_EMOTION if word in text)
        question_count = text.count("?") + text.count("？")
        exclaim_count = text.count("!") + text.count("！")
        repeat_punct = 1.0 if re.search(r"([!?！？。~～])\1+", text) else 0.0

        vector = [
            self._clip(length / 80),
            self._clip(chinese_count / max(length, 1)),
            self._clip(punct_count / 6),
            self._clip(emoji_count / 3),
            self._clip(face_count / 3),
            self._clip(meme_count / 3),
            self._clip(particle_count / 4),
            self._clip(sarcasm_count / 2),
            self._clip(emotion_count / 3),
            self._clip(question_count / 3),
            self._clip(exclaim_count / 3),
            repeat_punct,
            1.0 if length <= 8 else 0.0,
            1.0 if length >= 40 else 0.0,
        ]
        quality = self.quality_score(text, vector)
        brief = self.style_brief(text, semantic_tag, vector)
        return StyleProfile(vector=vector, quality_score=quality, tag=semantic_tag, brief=brief)

    def quality_score(self, text: str, vector: list[float]) -> float:
        normalized = re.sub(r"\s+", "", text)
        if not normalized:
            return 0.0
        low_signal = {"哈", "哈哈", "哈哈哈", "6", "？", "?", "。", "草", "哦", "嗯"}
        if normalized in low_signal:
            return 0.1
        if len(set(normalized)) <= 2 and len(normalized) <= 8:
            return 0.15
        score = 0.35
        score += min(len(normalized) / 60, 0.25)
        score += vector[5] * 0.18
        score += vector[6] * 0.08
        score += vector[7] * 0.12
        score += vector[8] * 0.12
        score += vector[11] * 0.05
        if len(normalized) > 220:
            score -= 0.2
        if re.search(r"(.{2,})\1{3,}", normalized):
            score -= 0.25
        return max(0.0, min(score, 1.0))

    def style_similarity(self, left: list[float], right: list[float]) -> float:
        return max(0.0, cosine_similarity(left, right))

    def recency_score(self, timestamp: int, now: int | None = None) -> float:
        now = now or int(time.time())
        age_days = max(0.0, (now - timestamp) / 86400)
        if age_days <= 7:
            return 1.0
        if age_days >= 90:
            return 0.15
        return max(0.15, math.exp(-(age_days - 7) / 38))

    def style_brief(self, text: str, tag: str, vector: list[float]) -> str:
        parts = [f"类型: {tag}"]
        if vector[12]:
            parts.append("短句")
        if vector[13]:
            parts.append("长解释")
        if vector[5] > 0:
            parts.append("梗/抽象表达明显")
        if vector[7] > 0:
            parts.append("阴阳怪气")
        if vector[8] > 0:
            parts.append("情绪强")
        if vector[10] > 0:
            parts.append("感叹语气")
        if vector[3] > 0 or vector[4] > 0:
            parts.append("表情/颜文字")
        return "，".join(parts)

    def _clip(self, value: float) -> float:
        return max(0.0, min(float(value), 1.0))


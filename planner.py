from __future__ import annotations

import json
import logging
import re
from typing import Any

try:
    from astrbot.api import logger
except Exception:  # pragma: no cover - allows standalone smoke tests.
    logger = logging.getLogger(__name__)

from .cleaning import StyleClassifier
from .models import ReplyPlan


VALID_INTENTS = {
    "casual_reply",
    "answer_question",
    "serious_explain",
    "joke",
    "sarcasm",
    "comfort",
    "refuse",
    "ask_clarification",
    "unknown",
}


class ResponsePlanner:
    def __init__(self, classifier: StyleClassifier | None = None) -> None:
        self.classifier = classifier or StyleClassifier()

    async def build_plan(
        self,
        *,
        current_context: str,
        prompt: str | None = None,
        contexts: list[dict] | None = None,
        persona_summary: str = "",
        admin_annotations: list[str] | None = None,
        provider: Any | None = None,
    ) -> ReplyPlan:
        if provider is not None:
            try:
                return await self._llm_plan(
                    provider=provider,
                    current_context=current_context,
                    prompt=prompt,
                    contexts=contexts or [],
                    persona_summary=persona_summary,
                    admin_annotations=admin_annotations or [],
                )
            except Exception as exc:
                logger.warning("[VibeDuplicate] planner provider failed, fallback to rules: %s", exc)
        return self._fallback_plan(current_context=current_context, prompt=prompt)

    async def _llm_plan(
        self,
        *,
        provider: Any,
        current_context: str,
        prompt: str | None,
        contexts: list[dict],
        persona_summary: str,
        admin_annotations: list[str],
    ) -> ReplyPlan:
        response = await provider.text_chat(
            system_prompt=(
                "You are a reply content planner. Decide what the bot should say next "
                "from the current chat context only. Do not imitate any target user's style. "
                "Do not use retrieved history as facts. Return strict JSON only."
            ),
            prompt=self._planner_prompt(
                current_context=current_context,
                prompt=prompt,
                contexts=contexts,
                persona_summary=persona_summary,
                admin_annotations=admin_annotations,
            ),
        )
        text = getattr(response, "completion_text", "") or ""
        data = _extract_json_object(text)
        return _coerce_plan(data, source="llm")

    def _planner_prompt(
        self,
        *,
        current_context: str,
        prompt: str | None,
        contexts: list[dict],
        persona_summary: str,
        admin_annotations: list[str],
    ) -> str:
        compact_contexts = []
        for item in contexts[-8:]:
            if isinstance(item, dict):
                role = str(item.get("role", "unknown"))
                content = item.get("content", "")
            else:
                role = str(getattr(item, "role", "unknown"))
                content = getattr(item, "content", "")
            if isinstance(content, str) and content.strip():
                compact_contexts.append({"role": role, "content": content.strip()[:800]})
        return json.dumps(
            {
                "task": "Create a content-only reply plan. The plan is not the final reply.",
                "allowed_reply_intents": sorted(VALID_INTENTS),
                "rules": [
                    "content_summary must come only from current_context, prompt, and contexts.",
                    "persona_summary may influence only preference/constraints, not facts.",
                    "admin_annotations are constraints, not factual evidence about current chat.",
                    "If context is insufficient, content_summary should express uncertainty or ask for clarification.",
                    "style_query is a short abstract query for finding style examples, not the original user question.",
                    "target_style_tag should be one of complaint, banter, serious_explain, silly, meme, sarcasm, neutral.",
                ],
                "current_context": current_context,
                "prompt": prompt or "",
                "contexts": compact_contexts,
                "persona_summary": persona_summary or "",
                "admin_annotations": admin_annotations,
                "output_schema": {
                    "should_reply": "boolean",
                    "reply_intent": "string",
                    "content_summary": "string",
                    "factual_constraints": ["string"],
                    "uncertainty": "string",
                    "style_query": "string",
                    "target_style_tag": "string",
                },
            },
            ensure_ascii=False,
        )

    def _fallback_plan(self, *, current_context: str, prompt: str | None) -> ReplyPlan:
        text = (prompt or current_context or "").strip()
        tag = self.classifier.classify(text)
        lowered = text.lower()
        if not text:
            intent = "ask_clarification"
            summary = "表达不确定，让对方补充上下文。"
            uncertainty = "没有足够上下文。"
        elif any(mark in text for mark in ("?", "？", "吗", "咋", "怎么", "为什么")):
            intent = "answer_question"
            summary = "基于当前问题给出直接、谨慎的回答；不确定的部分说明需要补充信息。"
            uncertainty = "如果当前问题缺少背景，说明不确定。"
        elif any(word in lowered for word in ("别", "不要", "不能", "不行")):
            intent = "refuse"
            summary = "简短拒绝或说明不能这样做。"
            uncertainty = ""
        else:
            intent = {
                "serious_explain": "serious_explain",
                "sarcasm": "sarcasm",
                "silly": "joke",
                "complaint": "casual_reply",
            }.get(tag, "casual_reply")
            summary = "围绕当前聊天内容自然接一句，不引入历史样本里的新事实。"
            uncertainty = ""
        return ReplyPlan(
            should_reply=True,
            reply_intent=intent,
            content_summary=summary,
            factual_constraints=["只使用当前聊天上下文中的事实。"],
            uncertainty=uncertainty,
            style_query=f"{intent} {tag} {summary}",
            target_style_tag=tag,
            planner_source="fallback",
        )


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("planner response did not contain JSON")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("planner JSON is not an object")
    return value


def _coerce_plan(data: dict[str, Any], *, source: str) -> ReplyPlan:
    intent = str(data.get("reply_intent") or "unknown").strip()
    if intent not in VALID_INTENTS:
        intent = "unknown"
    constraints = data.get("factual_constraints") or []
    if not isinstance(constraints, list):
        constraints = [str(constraints)]
    tag = str(data.get("target_style_tag") or "neutral").strip() or "neutral"
    summary = str(data.get("content_summary") or "").strip()
    if not summary:
        summary = "表达不确定/让对方补充。"
    style_query = str(data.get("style_query") or "").strip()
    if not style_query:
        style_query = f"{intent} {tag} {summary}"
    return ReplyPlan(
        should_reply=bool(data.get("should_reply", True)),
        reply_intent=intent,
        content_summary=summary,
        factual_constraints=[str(item).strip() for item in constraints if str(item).strip()],
        uncertainty=str(data.get("uncertainty") or "").strip(),
        style_query=style_query,
        target_style_tag=tag,
        planner_source=source,
    )

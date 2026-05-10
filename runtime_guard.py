from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


_INTERNAL_LLM_CALL: ContextVar[bool] = ContextVar(
    "vibe_duplicate_internal_llm_call",
    default=False,
)


def internal_llm_call_active() -> bool:
    return _INTERNAL_LLM_CALL.get()


@contextmanager
def internal_llm_call_guard() -> Iterator[None]:
    token = _INTERNAL_LLM_CALL.set(True)
    try:
        yield
    finally:
        _INTERNAL_LLM_CALL.reset(token)

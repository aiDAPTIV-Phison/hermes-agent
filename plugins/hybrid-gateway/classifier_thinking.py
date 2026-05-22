"""Thinking suppression for classifier LLM calls (Qwen / Gemma 4)."""

from __future__ import annotations

import logging
import re
from typing import Any, Literal, Optional

from .classifier_prompt import NO_THINKING_SUFFIX

logger = logging.getLogger(__name__)

ThinkingStrategy = Literal["auto", "gemma4-raw", "qwen-nothink"]
THINKING_STRATEGIES: tuple[ThinkingStrategy, ...] = (
    "auto",
    "gemma4-raw",
    "qwen-nothink",
)

_GEMMA4_RE = re.compile(r"gemma[-_]?4", re.I)


def is_gemma4_model(model: str) -> bool:
    return bool(_GEMMA4_RE.search(model or ""))


def resolve_thinking_strategy(
    model: str,
    strategy: Optional[str] = None,
) -> ThinkingStrategy:
    normalized = (strategy or "auto").strip().lower()
    if normalized in THINKING_STRATEGIES and normalized != "auto":
        return normalized  # type: ignore[return-value]
    return "gemma4-raw" if is_gemma4_model(model) else "qwen-nothink"


def build_gemma4_nothink_prompt(system_content: str, user_content: str) -> str:
    """Gemma 4 raw prompt without <|think|>; empty thought channel suppresses ghost thinking."""
    return (
        f"<|turn>system\n{system_content}<turn|>\n"
        f"<|turn>user\n{user_content}<turn|>\n"
        f"<|turn>model\n"
        f"<|channel>thought\n<channel|>"
    )


def system_prompt_with_nothink(system_prompt: str) -> str:
    return system_prompt + NO_THINKING_SUFFIX


def extract_llm_text(response: Any) -> str:
    if not response or not getattr(response, "choices", None):
        return ""
    choice = response.choices[0]
    message = getattr(choice, "message", None)
    if message is not None:
        return getattr(message, "content", None) or ""
    return getattr(choice, "text", None) or ""

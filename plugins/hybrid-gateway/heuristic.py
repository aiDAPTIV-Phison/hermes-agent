"""Heuristic classifier fallback."""

from __future__ import annotations

import re
from typing import List, Tuple

from .classifier_prompt import get_complexity_value

KEYWORD_RULES: List[Tuple[re.Pattern, str, str]] = [
    (re.compile(r"\b(code|function|class|def|async|return)\b", re.I), "coding", "moderate"),
    (re.compile(r"\b(debug|error|bug|fix|crash|exception)\b", re.I), "coding", "complex"),
    (re.compile(r"\b(translate|翻譯|翻译)\b", re.I), "translation", "simple"),
    (re.compile(r"\b(search|搜尋|搜索)\b", re.I), "search", "simple"),
]

_COMPLEXITY_ORDER = ("trivial", "simple", "moderate", "complex", "expert")


def _max_complexity(a: str, b: str) -> str:
    return a if get_complexity_value(a) >= get_complexity_value(b) else b


def heuristic_classify(user_input: str) -> dict:
    n = len(user_input)
    if n < 20:
        complexity = "trivial"
    elif n < 100:
        complexity = "simple"
    elif n < 500:
        complexity = "moderate"
    else:
        complexity = "complex"
    skills: set[str] = set()
    for pattern, skill, min_c in KEYWORD_RULES:
        if pattern.search(user_input):
            skills.add(skill)
            complexity = _max_complexity(complexity, min_c)
    if not skills:
        skills.add("conversation")
    return {"complexity": complexity, "skills": sorted(skills)}

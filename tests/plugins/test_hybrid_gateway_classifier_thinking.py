"""Tests for hybrid-gateway classifier thinking helpers."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[2]
_PLUGIN_DIR = _REPO / "plugins" / "hybrid-gateway"


def _ensure_ns() -> None:
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    pkg = "hermes_plugins.hybrid_gateway"
    if pkg not in sys.modules:
        mod = types.ModuleType(pkg)
        mod.__path__ = [str(_PLUGIN_DIR)]
        mod.__package__ = pkg
        sys.modules[pkg] = mod


def _load_module(name: str, filename: str):
    _ensure_ns()
    full_name = f"hermes_plugins.hybrid_gateway.{name}"
    path = _PLUGIN_DIR / filename
    spec = importlib.util.spec_from_file_location(
        full_name,
        path,
        submodule_search_locations=[str(_PLUGIN_DIR)],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hermes_plugins.hybrid_gateway"
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def thinking_mod():
    _load_module("classifier_prompt", "classifier_prompt.py")
    return _load_module("classifier_thinking", "classifier_thinking.py")


def test_no_thinking_suffix_matches_openclaw(thinking_mod):
    assert "/nothink" in thinking_mod.NO_THINKING_SUFFIX
    assert "chain-of-thought" in thinking_mod.NO_THINKING_SUFFIX


def test_is_gemma4_model(thinking_mod):
    assert thinking_mod.is_gemma4_model("gemma-4-26B-A4B-it")
    assert thinking_mod.is_gemma4_model("Gemma_4_local")
    assert not thinking_mod.is_gemma4_model("qwen2.5-3b-instruct")


def test_resolve_thinking_strategy_auto(thinking_mod):
    assert thinking_mod.resolve_thinking_strategy("gemma-4-26B", "auto") == "gemma4-raw"
    assert thinking_mod.resolve_thinking_strategy("qwen3-8b", "auto") == "qwen-nothink"


def test_resolve_thinking_strategy_explicit(thinking_mod):
    assert thinking_mod.resolve_thinking_strategy("qwen3-8b", "gemma4-raw") == "gemma4-raw"
    assert thinking_mod.resolve_thinking_strategy("gemma-4", "qwen-nothink") == "qwen-nothink"


def test_build_gemma4_nothink_prompt(thinking_mod):
    prompt = thinking_mod.build_gemma4_nothink_prompt("sys", "user msg")
    assert "<|turn>system" in prompt
    assert "sys" in prompt
    assert "user msg" in prompt
    assert "<|channel>thought" in prompt


def test_system_prompt_with_nothink(thinking_mod):
    combined = thinking_mod.system_prompt_with_nothink("base")
    assert combined.startswith("base")
    assert thinking_mod.NO_THINKING_SUFFIX in combined


def test_extract_llm_text_chat_and_completion(thinking_mod):
    chat = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"complexity":"simple"}'))]
    )
    assert "simple" in thinking_mod.extract_llm_text(chat)

    completion = SimpleNamespace(choices=[SimpleNamespace(text='{"complexity":"trivial"}')])
    assert "trivial" in thinking_mod.extract_llm_text(completion)

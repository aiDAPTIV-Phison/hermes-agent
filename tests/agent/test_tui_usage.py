"""Tests for agent/tui_usage.py — TUI status-bar usage snapshots."""

import types

from agent.tui_usage import build_agent_usage


def test_build_agent_usage_prefers_last_prompt_tokens():
    agent = types.SimpleNamespace(
        model="custom/gemma",
        session_total_tokens=50000,
        session_api_calls=3,
        context_compressor=types.SimpleNamespace(
            last_prompt_tokens=16213,
            context_length=128000,
            compression_count=1,
        ),
    )
    usage = build_agent_usage(agent)
    assert usage["context_used"] == 16213
    assert usage["context_max"] == 128000
    assert usage["compressions"] == 1


def test_build_agent_usage_context_tokens_override():
    agent = types.SimpleNamespace(
        model="custom/gemma",
        session_total_tokens=0,
        session_api_calls=0,
        context_compressor=types.SimpleNamespace(
            last_prompt_tokens=0,
            context_length=128000,
            compression_count=0,
        ),
    )
    usage = build_agent_usage(agent, context_tokens=16213)
    assert usage["context_used"] == 16213

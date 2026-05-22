"""Auto-compression surfaces compressing/compressed status for gateway/TUI."""

import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def agent_with_status():
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.status_callback = MagicMock()
    return agent


def _messages(n: int = 12):
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
        for i in range(n)
    ]


def test_compress_context_emits_compressing_and_compressed(agent_with_status):
    agent = agent_with_status
    before = _messages(12)
    after = before[:3] + [{"role": "user", "content": "[summary]"}] + before[-2:]

    agent.context_compressor.compress = MagicMock(return_value=after)
    agent._build_system_prompt = MagicMock(return_value="sys")
    agent._invalidate_system_prompt = MagicMock()
    agent._todo_store.format_for_injection = MagicMock(return_value="")

    with patch(
        "run_agent.estimate_request_tokens_rough",
        side_effect=[50_000, 12_000],
    ):
        compressed, _ = agent._compress_context(before, "base", approx_tokens=50_000)

    assert compressed == after
    kinds = [c.args[0] for c in agent.status_callback.call_args_list]
    assert "compressing" in kinds
    assert "compressed" in kinds
    assert "ready" in kinds
    compressing_text = next(c.args[1] for c in agent.status_callback.call_args_list if c.args[0] == "compressing")
    assert "compressing 12 messages" in compressing_text
    assert "~50,000 tok" in compressing_text


def test_compress_context_skips_ui_when_disabled(agent_with_status):
    agent = agent_with_status
    before = _messages(8)
    after = before[:2]

    agent.context_compressor.compress = MagicMock(return_value=after)
    agent._build_system_prompt = MagicMock(return_value="sys")
    agent._invalidate_system_prompt = MagicMock()
    agent._todo_store.format_for_injection = MagicMock(return_value="")

    agent._compress_context(
        before,
        "base",
        approx_tokens=10_000,
        emit_progress=False,
        emit_summary=False,
    )

    agent.status_callback.assert_not_called()

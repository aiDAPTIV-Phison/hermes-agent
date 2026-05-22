"""Hybrid-gateway route status is surfaced to gateway/TUI consumers."""

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
            model="default/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.status_callback = MagicMock()
    return agent


def test_effective_compression_model_prefers_auxiliary_config(agent_with_status):
    agent = agent_with_status
    agent.context_compressor.summary_model = ""
    with patch(
        "agent.auxiliary_client.get_text_auxiliary_client",
        return_value=(MagicMock(), "stepfun/step-3.5-flash"),
    ):
        assert agent._effective_compression_model() == "stepfun/step-3.5-flash"


def test_apply_hybrid_model_resolve_emits_hybrid_status(agent_with_status):
    agent = agent_with_status
    agent.switch_model = MagicMock()

    ok = agent.apply_hybrid_model_resolve(
        {
            "provider": "zhipu",
            "model": "glm-4-flash",
            "tier": "edge",
            "reason": "policy-array",
            "models": {"edge": {}, "cloud": {}},
        }
    )

    assert ok is True
    agent.status_callback.assert_called()
    call = agent.status_callback.call_args
    assert call.args[0] == "hybrid"
    assert "edge" in call.args[1]
    assert call.kwargs.get("hybrid_tier") == "edge"


def test_emit_compressing_updates_context_meter_with_preflight_estimate(agent_with_status):
    agent = agent_with_status
    agent.context_compressor.context_length = 18_432
    agent.context_compressor.last_prompt_tokens = 9_500

    agent._emit_compressing_progress(
        [{"role": "user", "content": "hi"}] * 9,
        approx_tokens=17_399,
    )

    usage_calls = [
        c
        for c in agent.status_callback.call_args_list
        if c.args and c.args[0] == "usage"
    ]
    assert usage_calls, "expected live usage snapshot when compression starts"
    assert usage_calls[0].kwargs["usage"]["context_used"] == 17_399


def test_emit_compressing_shows_compression_badge_then_restores_edge(agent_with_status):
    agent = agent_with_status
    agent._hybrid_tier = "edge"
    agent._effective_compression_model = MagicMock(
        return_value="stepfun/step-3.5-flash"
    )
    agent.context_compressor.compress = MagicMock(return_value=[{"role": "user", "content": "x"}])
    agent._build_system_prompt = MagicMock(return_value="sys")
    agent._invalidate_system_prompt = MagicMock()
    agent._todo_store.format_for_injection = MagicMock(return_value="")

    agent._compress_context(
        [{"role": "user", "content": "hi"}] * 6,
        "base",
        approx_tokens=50_000,
        emit_summary=False,
    )

    compressing_hybrid = [
        c
        for c in agent.status_callback.call_args_list
        if c.args
        and c.args[0] == "hybrid"
        and c.kwargs.get("hybrid_tier") == "compressing"
    ]
    assert compressing_hybrid, "expected compressing badge during compaction"
    assert compressing_hybrid[0].kwargs.get("model") == "stepfun/step-3.5-flash"
    compressing_status = [
        c
        for c in agent.status_callback.call_args_list
        if c.args and c.args[0] == "compressing"
    ]
    assert compressing_status
    assert compressing_status[0].kwargs.get("model") == "stepfun/step-3.5-flash"
    restore_calls = [
        c
        for c in agent.status_callback.call_args_list
        if c.args
        and c.args[0] == "hybrid"
        and c.kwargs.get("hybrid_tier") == "edge"
    ]
    assert restore_calls, "expected edge badge restored after compression"


def test_emit_hybrid_phase_classifying(agent_with_status):
    agent = agent_with_status
    agent._emit_hybrid_phase("classifying")
    agent.status_callback.assert_called_once()
    call = agent.status_callback.call_args
    assert call.kwargs.get("hybrid_tier") == "classifying"
    assert "classifying" in call.args[1]


def test_escalate_hybrid_to_cloud_emits_cloud_status(agent_with_status):
    agent = agent_with_status
    agent._hybrid_tier = "edge"
    agent._hybrid_models = {
        "cloud": {
            "provider": "openrouter",
            "model": "anthropic/claude-sonnet-4",
        }
    }
    agent.switch_model = MagicMock()

    assert agent._escalate_hybrid_to_cloud() is True
    assert agent._hybrid_tier == "cloud"
    call = agent.status_callback.call_args
    assert call.kwargs.get("hybrid_tier") == "cloud"
    assert call.kwargs.get("hybrid_escalated") is True


def test_escalate_hybrid_to_cloud_context_note(agent_with_status):
    agent = agent_with_status
    agent._hybrid_tier = "edge"
    agent._hybrid_models = {
        "cloud": {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"}
    }
    agent.switch_model = MagicMock()

    assert agent._escalate_hybrid_to_cloud(escalated_note="context→cloud") is True
    assert "context→cloud" in agent.status_callback.call_args.args[1]


def test_escalate_hybrid_to_cloud_noop_when_already_escalated(agent_with_status):
    agent = agent_with_status
    agent._hybrid_tier = "edge"
    agent._hybrid_escalated = True
    agent._hybrid_models = {
        "cloud": {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"}
    }
    agent.switch_model = MagicMock()

    assert agent._escalate_hybrid_to_cloud(escalated_note="context→cloud") is False
    agent.switch_model.assert_not_called()

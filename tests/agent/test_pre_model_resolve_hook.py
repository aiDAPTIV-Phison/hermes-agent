"""Tests for pre_model_resolve hook and hybrid escalation helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_get_pre_model_resolve_override_first_wins():
    from hermes_cli.plugins import get_pre_model_resolve_override

    with patch("hermes_cli.plugins.invoke_hook") as mock_invoke:
        mock_invoke.return_value = [
            None,
            {"model": "m1", "provider": "custom", "tier": "edge"},
            {"model": "m2", "provider": "openrouter"},
        ]
        out = get_pre_model_resolve_override(user_message="hi")
        assert out["model"] == "m1"
        assert out["tier"] == "edge"


def test_apply_hybrid_model_resolve_sets_state():
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent.log_prefix = ""
    agent._hybrid_tier = None
    agent._hybrid_models = {}
    agent._hybrid_reason = ""
    agent._hybrid_escalated = False

    with patch.object(AIAgent, "switch_model") as mock_switch:
        ok = AIAgent.apply_hybrid_model_resolve(
            agent,
            {
                "provider": "custom",
                "model": "edge-model",
                "tier": "edge",
                "reason": "test",
                "models": {
                    "edge": {"provider": "custom", "model": "edge-model"},
                    "cloud": {"provider": "openrouter", "model": "cloud-model"},
                },
            },
        )
        assert ok
        assert agent._hybrid_tier == "edge"
        assert agent._hybrid_models["cloud"]["model"] == "cloud-model"
        mock_switch.assert_called_once()


def test_escalate_hybrid_to_cloud():
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent.log_prefix = ""
    agent._hybrid_tier = "edge"
    agent._hybrid_escalated = False
    agent._hybrid_models = {
        "cloud": {
            "provider": "openrouter",
            "model": "cloud-model",
            "base_url": "",
            "api_key": "",
            "api_mode": "",
        },
    }

    with patch.object(AIAgent, "switch_model") as mock_switch:
        assert AIAgent._escalate_hybrid_to_cloud(agent) is True
        assert agent._hybrid_tier == "cloud"
        assert agent._hybrid_escalated is True
        mock_switch.assert_called_once()

    assert AIAgent._escalate_hybrid_to_cloud(agent) is False

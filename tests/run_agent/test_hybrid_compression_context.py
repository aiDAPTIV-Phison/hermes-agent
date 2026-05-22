"""Compression thresholds follow hybrid-gateway tier (cloud vs edge)."""

from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _agent_with_compressor(*, context_length: int = 200_000, hybrid_tier: str = ""):
    agent = AIAgent.__new__(AIAgent)
    agent.log_prefix = ""
    agent._hybrid_tier = hybrid_tier
    agent._hybrid_escalated = False
    compressor = MagicMock()
    compressor.context_length = context_length
    compressor.threshold_tokens = int(context_length * 0.5)
    agent.context_compressor = compressor
    return agent


_HG_CFG = {
    "enabled": True,
    "models": {
        "edge": {
            "provider": "zhipu",
            "model": "glm-4-flash",
            "base_url": "https://example/v1",
            "api_key": "k",
            "api_mode": "chat_completions",
        },
        "cloud": {
            "provider": "openrouter",
            "model": "anthropic/claude-sonnet-4",
            "base_url": "https://openrouter/v1",
            "api_key": "ck",
            "api_mode": "chat_completions",
        },
    },
}


@patch("hermes_plugins.hybrid_gateway.model_resolve.resolve_edge_context_length", return_value=18_432)
@patch("hermes_plugins.hybrid_gateway.model_resolve.load_full_config")
def test_sync_hybrid_compression_context_edge_tier(mock_cfg, mock_edge_ctx):
    mock_cfg.return_value = _HG_CFG
    agent = _agent_with_compressor(context_length=200_000, hybrid_tier="edge")

    with patch("hermes_cli.plugins._ensure_plugins_discovered"):
        assert agent._sync_hybrid_compression_context() is True

    agent.context_compressor.update_model.assert_called_once_with(
        model="glm-4-flash",
        context_length=18_432,
        base_url="https://example/v1",
        api_key="k",
        provider="zhipu",
        api_mode="chat_completions",
    )


@patch("hermes_plugins.hybrid_gateway.model_resolve.resolve_cloud_context_length", return_value=200_000)
@patch("hermes_plugins.hybrid_gateway.model_resolve.load_full_config")
def test_sync_hybrid_compression_context_cloud_tier(mock_cfg, mock_cloud_ctx):
    mock_cfg.return_value = _HG_CFG
    agent = _agent_with_compressor(context_length=18_432, hybrid_tier="cloud")

    with patch("hermes_cli.plugins._ensure_plugins_discovered"):
        assert agent._sync_hybrid_compression_context() is True

    agent.context_compressor.update_model.assert_called_once_with(
        model="anthropic/claude-sonnet-4",
        context_length=200_000,
        base_url="https://openrouter/v1",
        api_key="ck",
        provider="openrouter",
        api_mode="chat_completions",
    )


@patch("hermes_plugins.hybrid_gateway.model_resolve.resolve_cloud_context_length", return_value=200_000)
@patch("hermes_plugins.hybrid_gateway.model_resolve.load_full_config")
def test_sync_hybrid_compression_context_escalated_uses_cloud(mock_cfg, mock_cloud_ctx):
    mock_cfg.return_value = _HG_CFG
    agent = _agent_with_compressor(context_length=18_432, hybrid_tier="edge")
    agent._hybrid_escalated = True

    with patch("hermes_cli.plugins._ensure_plugins_discovered"):
        assert agent._hybrid_compression_budget_tier() == "cloud"
        assert agent._sync_hybrid_compression_context() is True

    agent.context_compressor.update_model.assert_called_once()
    assert agent.context_compressor.update_model.call_args.kwargs["context_length"] == 200_000


@patch("hermes_plugins.hybrid_gateway.model_resolve.load_full_config", return_value=None)
def test_sync_hybrid_compression_context_noop_when_disabled(mock_cfg):
    agent = _agent_with_compressor(hybrid_tier="edge")
    with patch("hermes_cli.plugins._ensure_plugins_discovered"):
        assert agent._sync_hybrid_compression_context() is False
    agent.context_compressor.update_model.assert_not_called()


@patch("hermes_plugins.hybrid_gateway.model_resolve.resolve_edge_context_length", return_value=18_432)
@patch("hermes_plugins.hybrid_gateway.model_resolve.load_full_config")
def test_sync_hybrid_compression_context_idempotent(mock_cfg, mock_edge_ctx):
    mock_cfg.return_value = _HG_CFG
    agent = _agent_with_compressor(context_length=18_432, hybrid_tier="edge")

    with patch("hermes_cli.plugins._ensure_plugins_discovered"):
        assert agent._sync_hybrid_compression_context() is True

    agent.context_compressor.update_model.assert_not_called()

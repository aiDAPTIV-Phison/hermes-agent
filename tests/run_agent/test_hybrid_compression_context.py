"""Compression thresholds follow hybrid-gateway edge ctx when enabled."""

from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _agent_with_compressor(*, context_length: int = 200_000):
    agent = AIAgent.__new__(AIAgent)
    agent.log_prefix = ""
    compressor = MagicMock()
    compressor.context_length = context_length
    compressor.threshold_tokens = int(context_length * 0.5)
    agent.context_compressor = compressor
    return agent


@patch("hermes_plugins.hybrid_gateway.model_resolve.resolve_edge_context_length", return_value=18_432)
@patch("hermes_plugins.hybrid_gateway.model_resolve.load_full_config")
def test_sync_hybrid_edge_compression_context_updates_compressor(mock_cfg, mock_ctx):
    mock_cfg.return_value = {
        "enabled": True,
        "models": {
            "edge": {
                "provider": "zhipu",
                "model": "glm-4-flash",
                "base_url": "https://example/v1",
                "api_key": "k",
                "api_mode": "chat_completions",
            }
        },
    }
    agent = _agent_with_compressor(context_length=200_000)

    with patch("hermes_cli.plugins._ensure_plugins_discovered"):
        assert agent._sync_hybrid_edge_compression_context() is True

    agent.context_compressor.update_model.assert_called_once_with(
        model="glm-4-flash",
        context_length=18_432,
        base_url="https://example/v1",
        api_key="k",
        provider="zhipu",
        api_mode="chat_completions",
    )


@patch("hermes_plugins.hybrid_gateway.model_resolve.load_full_config", return_value=None)
def test_sync_hybrid_edge_compression_context_noop_when_disabled(mock_cfg):
    agent = _agent_with_compressor()
    with patch("hermes_cli.plugins._ensure_plugins_discovered"):
        assert agent._sync_hybrid_edge_compression_context() is False
    agent.context_compressor.update_model.assert_not_called()


@patch("hermes_plugins.hybrid_gateway.model_resolve.resolve_edge_context_length", return_value=18_432)
@patch("hermes_plugins.hybrid_gateway.model_resolve.load_full_config")
def test_sync_hybrid_edge_compression_context_idempotent(mock_cfg, mock_ctx):
    mock_cfg.return_value = {
        "enabled": True,
        "models": {"edge": {"provider": "zhipu", "model": "glm-4-flash"}},
    }
    agent = _agent_with_compressor(context_length=18_432)

    with patch("hermes_cli.plugins._ensure_plugins_discovered"):
        assert agent._sync_hybrid_edge_compression_context() is True

    agent.context_compressor.update_model.assert_not_called()

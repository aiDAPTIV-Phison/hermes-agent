"""Tests for hybrid-gateway tier model resolution."""

from hermes_plugins.hybrid_gateway.model_resolve import resolve_tier_mapping


def test_resolve_tier_mapping_openrouter_fills_base_url(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-openrouter-key")
    spec = {
        "provider": "openrouter",
        "model": "stepfun/step-3.5-flash",
    }
    result = resolve_tier_mapping(spec, {})
    assert result["provider"] == "openrouter"
    assert result["model"] == "stepfun/step-3.5-flash"
    assert "openrouter.ai" in result["base_url"]
    assert result["api_key"] == "sk-test-openrouter-key"


def test_resolve_tier_mapping_keeps_explicit_custom_base_url():
    spec = {
        "provider": "custom",
        "model": "gemma-local",
        "base_url": "http://127.0.0.1:13141/v1",
        "api_key": "local-key",
    }
    result = resolve_tier_mapping(spec, {})
    assert result["base_url"] == "http://127.0.0.1:13141/v1"
    assert result["api_key"] == "local-key"

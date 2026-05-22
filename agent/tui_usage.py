"""Usage snapshot for TUI status bar (context meter + session totals)."""

from __future__ import annotations

from typing import Any, Dict, Optional


def build_agent_usage(agent: Any, *, context_tokens: Optional[int] = None) -> Dict[str, Any]:
    """Mirror tui_gateway ``_get_usage`` so live status updates stay consistent."""

    def _g(key: str, fallback: str | None = None) -> int:
        val = getattr(agent, key, 0) or 0
        if not val and fallback:
            val = getattr(agent, fallback, 0) or 0
        return int(val)

    usage: Dict[str, Any] = {
        "model": getattr(agent, "model", "") or "",
        "input": _g("session_input_tokens", "session_prompt_tokens"),
        "output": _g("session_output_tokens", "session_completion_tokens"),
        "cache_read": _g("session_cache_read_tokens"),
        "cache_write": _g("session_cache_write_tokens"),
        "reasoning": _g("session_reasoning_tokens"),
        "prompt": _g("session_prompt_tokens"),
        "completion": _g("session_completion_tokens"),
        "total": _g("session_total_tokens"),
        "calls": _g("session_api_calls"),
    }
    comp = getattr(agent, "context_compressor", None)
    if comp:
        if context_tokens is not None:
            ctx_used = max(0, int(context_tokens))
        else:
            ctx_used = getattr(comp, "last_prompt_tokens", 0) or usage["total"] or 0
        ctx_max = getattr(comp, "context_length", 0) or 0
        if ctx_max:
            usage["context_used"] = ctx_used
            usage["context_max"] = ctx_max
            usage["context_percent"] = max(0, min(100, round(ctx_used / ctx_max * 100)))
        usage["compressions"] = getattr(comp, "compression_count", 0) or 0
    try:
        from agent.usage_pricing import CanonicalUsage, estimate_usage_cost

        cost = estimate_usage_cost(
            usage["model"],
            CanonicalUsage(
                input_tokens=usage["input"],
                output_tokens=usage["output"],
                cache_read_tokens=usage["cache_read"],
                cache_write_tokens=usage["cache_write"],
            ),
            provider=getattr(agent, "provider", None),
            base_url=getattr(agent, "base_url", None),
        )
        usage["cost_status"] = cost.status
        if cost.amount_usd is not None:
            usage["cost_usd"] = float(cost.amount_usd)
    except Exception:
        pass
    return usage

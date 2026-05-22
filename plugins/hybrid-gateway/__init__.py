"""Hybrid Gateway — per-turn classifier/edge/cloud routing."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from .classifier import classify
from .config_schema import extract_user_text
from .model_resolve import load_full_config
from .router import apply_post_rules, route

logger = logging.getLogger(__name__)

_CONFIG: Optional[Dict[str, Any]] = None


def pre_model_resolve(
    user_message: str = "",
    approximate_context_tokens: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_full_config()
    cfg = _CONFIG
    if not cfg:
        return None

    routing = cfg.get("routing") or {}
    models = cfg.get("models") or {}
    classifier_cfg = cfg.get("classifier") or {}

    prompt = user_message if isinstance(user_message, str) else str(user_message or "")
    user_text = extract_user_text(prompt)

    edge_max = routing.get("edgeMaxContextTokens")
    if edge_max is not None and approximate_context_tokens is not None:
        try:
            threshold = int(edge_max)
            if approximate_context_tokens >= threshold:
                cloud = models.get("cloud") or {}
                if cloud.get("model") and cloud.get("provider"):
                    logger.info(
                        "hybrid-gateway: context %s >= threshold %s -> cloud",
                        approximate_context_tokens,
                        threshold,
                    )
                    return _to_override(
                        cloud,
                        "cloud",
                        f"context_tokens={approximate_context_tokens}>={threshold}",
                        models,
                    )
        except (TypeError, ValueError):
            pass

    for pat in routing.get("bypassPatterns") or []:
        if isinstance(pat, str) and pat.strip():
            try:
                if re.search(pat, user_text, re.I):
                    logger.debug("hybrid-gateway: bypass pattern matched")
                    return None
            except re.error:
                continue

    if not user_text:
        return None

    classified = apply_post_rules(classify(user_text, classifier_cfg))
    decision = route(classified, cfg)
    if not decision.get("model") or not decision.get("provider"):
        logger.warning("hybrid-gateway: empty route for tier=%s", decision.get("tier"))
        return None

    tier = decision.get("tier") or "?"
    logger.info(
        "hybrid-gateway: route tier=%s model=%s/%s reason=%s",
        tier,
        decision.get("provider"),
        decision.get("model"),
        (decision.get("reason") or "")[:120],
    )
    return _to_override(decision, tier, decision.get("reason") or "", models)


def _to_override(
    mapping: Dict[str, Any],
    tier: str,
    reason: str,
    models: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    return {
        "provider": mapping.get("provider", ""),
        "model": mapping.get("model", ""),
        "base_url": mapping.get("base_url", ""),
        "api_key": mapping.get("api_key", ""),
        "api_mode": mapping.get("api_mode", ""),
        "tier": tier,
        "reason": reason,
        "models": models,
    }


def register(ctx):
    global _CONFIG
    _CONFIG = load_full_config()
    if not _CONFIG:
        logger.info(
            "hybrid-gateway: idle (set hybrid_gateway.enabled: true in config.yaml)"
        )
        return
    ctx.register_hook("pre_model_resolve", pre_model_resolve)
    logger.info("hybrid-gateway: enabled — pre_model_resolve registered")

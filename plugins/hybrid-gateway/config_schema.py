"""Load hybrid_gateway config from config.yaml."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from .router import validate_policy_array

logger = logging.getLogger(__name__)


def load_hybrid_config() -> Optional[Dict[str, Any]]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
    except Exception:
        return None

    hg = cfg.get("hybrid_gateway") if isinstance(cfg, dict) else None
    if not isinstance(hg, dict) or not hg.get("enabled"):
        return None

    routing = dict(hg.get("routing") or {})
    raw_array = routing.get("policyArray") or routing.get("policy_array")
    validated = validate_policy_array(raw_array)
    if raw_array is not None and validated is None:
        logger.warning("hybrid_gateway.routing.policyArray invalid — using policy only")
    elif validated:
        routing["policyArray"] = validated

    classifier_cfg = hg.get("classifier") if isinstance(hg.get("classifier"), dict) else {}
    thinking_strategy = classifier_cfg.get("thinkingStrategy")
    if thinking_strategy is not None:
        thinking_strategy = str(thinking_strategy).strip().lower() or "auto"
    return {
        "enabled": True,
        "classifier": {
            "mode": classifier_cfg.get("mode") or "auxiliary",
            "auxiliary_task": classifier_cfg.get("auxiliary_task") or "classifier",
            "cacheEnabled": classifier_cfg.get("cacheEnabled", True),
            "cacheTtlSeconds": int(classifier_cfg.get("cacheTtlSeconds") or 300),
            "disableThinking": bool(classifier_cfg.get("disableThinking", False)),
            "thinkingStrategy": thinking_strategy or "auto",
            "systemPrompt": classifier_cfg.get("systemPrompt"),
        },
        "routing": routing,
        "models": hg.get("models") or {},
    }


def extract_user_text(prompt: str) -> str:
    text = prompt or ""
    text = re.sub(r"Sender \(untrusted metadata\):\s*```[\s\S]*?```\s*", "", text)
    text = re.sub(r"^\[.*?\]\s*", "", text, flags=re.MULTILINE)
    return text.strip()

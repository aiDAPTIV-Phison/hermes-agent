"""Three-tier routing engine."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .classifier_prompt import get_complexity_value

VALID_TIERS = ("classifier", "edge", "cloud")


def validate_policy_array(raw: Any) -> Optional[List[str]]:
    if not isinstance(raw, list) or len(raw) != 5:
        return None
    out: List[str] = []
    for item in raw:
        if not isinstance(item, str) or item not in VALID_TIERS:
            return None
        out.append(item)
    return out


def apply_policy_array(complexity: str, policy_array: List[str]) -> str:
    idx = get_complexity_value(complexity)
    return policy_array[idx] if idx < len(policy_array) else "edge"


def match_skill_route(skills: List[str], routes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for route in routes:
        pattern = route.get("skillPattern") or route.get("skill_pattern") or ""
        if not pattern:
            continue
        try:
            regex = re.compile(pattern, re.I)
        except re.error:
            continue
        for skill in skills:
            if regex.search(skill):
                return route
    return None


def apply_policy(complexity: str, policy: str) -> str:
    if policy == "edge-first":
        return "edge"
    if policy == "cloud-first":
        return "cloud"
    val = get_complexity_value(complexity)
    if policy == "cost-optimize-L1":
        return "edge" if val <= 1 else "cloud"
    if policy == "cost-optimize-L2":
        if val <= 1:
            return "classifier"
        if val == 2:
            return "edge"
        return "cloud"
    if policy == "cost-optimize-L3":
        if val <= 1:
            return "classifier"
        if val <= 3:
            return "edge"
        return "cloud"
    return "edge" if val <= 1 else "cloud"


def apply_post_rules(classify_result: Dict[str, Any]) -> Dict[str, Any]:
    skills = list(classify_result.get("skills") or [])
    complexity = classify_result.get("complexity") or "simple"
    if len(skills) >= 4 and get_complexity_value(complexity) < get_complexity_value("complex"):
        complexity = "complex"
    return {"complexity": complexity, "skills": skills}


def _mapping(models: Dict[str, Any], tier: str) -> Dict[str, str]:
    m = models.get(tier) or {}
    return {
        "provider": m.get("provider", ""),
        "model": m.get("model", ""),
        "base_url": m.get("base_url", ""),
        "api_key": m.get("api_key", ""),
        "api_mode": m.get("api_mode", ""),
    }


def route(classify_result: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    routing = config.get("routing") or {}
    models = config.get("models") or {}
    skills = classify_result.get("skills") or []
    complexity = classify_result.get("complexity") or "simple"

    matched = match_skill_route(skills, routing.get("skillRoutes") or routing.get("skill_routes") or [])
    if matched:
        force_tier = matched.get("forceTier") or matched.get("force_tier")
        prefer = matched.get("preferModel") or matched.get("prefer_model")
        tier = (force_tier or "cloud") if prefer else force_tier
        if tier and tier in models:
            d = _mapping(models, tier)
            d["tier"] = tier
            d["reason"] = f"skill-route: {matched.get('skillPattern', '')}"
            return d

    validated = validate_policy_array(routing.get("policyArray") or routing.get("policy_array"))
    if validated:
        tier = apply_policy_array(complexity, validated)
        d = _mapping(models, tier)
        d["tier"] = tier
        d["reason"] = f"policy-array, complexity={complexity} -> {tier}"
        return d

    policy = routing.get("policy") or "cost-optimize-L2"
    tier = apply_policy(complexity, policy)
    d = _mapping(models, tier)
    d["tier"] = tier
    d["reason"] = f"policy={policy}, complexity={complexity} -> {tier}"
    return d

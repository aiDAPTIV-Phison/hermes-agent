"""Resolve tier model mappings and config refs."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


def _resolve_custom_provider(name: str, config: Dict[str, Any]) -> Optional[Dict[str, str]]:
    entries = config.get("custom_providers")
    if not isinstance(entries, list):
        return None
    target = name.strip().lower()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if (entry.get("name") or "").strip().lower() != target:
            continue
        api_key = (entry.get("api_key") or "").strip()
        if not api_key:
            key_env = (entry.get("key_env") or entry.get("api_key_env") or "").strip()
            if key_env:
                api_key = os.getenv(key_env, "").strip()
        return {
            "provider": "custom",
            "model": (entry.get("model") or "").strip(),
            "base_url": (entry.get("base_url") or "").strip(),
            "api_key": api_key,
            "api_mode": (entry.get("api_mode") or "").strip(),
        }
    return None


def _resolve_auxiliary_ref(task: str, config: Dict[str, Any]) -> Optional[Dict[str, str]]:
    aux = config.get("auxiliary")
    if not isinstance(aux, dict):
        return None
    task_cfg = aux.get(task)
    if not isinstance(task_cfg, dict):
        return None
    provider = (task_cfg.get("provider") or "auto").strip()
    base_url = (task_cfg.get("base_url") or "").strip()
    api_key = (task_cfg.get("api_key") or "").strip()
    if base_url and not api_key:
        key_env = (task_cfg.get("key_env") or task_cfg.get("api_key_env") or "").strip()
        if key_env:
            api_key = os.getenv(key_env, "").strip()
    if base_url and provider in ("", "auto", "main"):
        provider = "custom"
    return {
        "provider": provider,
        "model": (task_cfg.get("model") or "").strip(),
        "base_url": base_url,
        "api_key": api_key,
        "api_mode": (task_cfg.get("api_mode") or "").strip(),
    }


def _enrich_provider_runtime(mapping: Dict[str, str]) -> Dict[str, str]:
    """Fill missing base_url/api_key/api_mode via resolve_runtime_provider.

    Matches model_switch / auxiliary behavior so tier specs like
    ``provider: openrouter`` without ``base_url`` still reach OpenRouter.
    """
    provider = (mapping.get("provider") or "").strip()
    model = (mapping.get("model") or "").strip()
    if not provider or not model:
        return mapping

    base_url = (mapping.get("base_url") or "").strip()
    api_key = (mapping.get("api_key") or "").strip()
    api_mode = (mapping.get("api_mode") or "").strip()
    if base_url and api_key:
        return mapping

    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(
            requested=provider,
            explicit_api_key=api_key or None,
            explicit_base_url=base_url or None,
            target_model=model,
        )
    except Exception:
        return mapping

    out = dict(mapping)
    resolved_base = str(runtime.get("base_url") or "").strip()
    resolved_key = str(runtime.get("api_key") or "").strip()
    resolved_mode = str(runtime.get("api_mode") or "").strip()
    if resolved_base and not base_url:
        out["base_url"] = resolved_base
    if resolved_key and not api_key:
        out["api_key"] = resolved_key
    if resolved_mode and not api_mode:
        out["api_mode"] = resolved_mode
    return out


def resolve_tier_mapping(spec: Any, config: Dict[str, Any]) -> Dict[str, str]:
    if not isinstance(spec, dict):
        return {"provider": "", "model": "", "base_url": "", "api_key": "", "api_mode": ""}

    ref = (spec.get("ref") or "").strip()
    if ref:
        if ref.startswith("auxiliary."):
            resolved = _resolve_auxiliary_ref(ref.split(".", 1)[1], config)
            if resolved:
                return _enrich_provider_runtime(resolved)
        name = ref.split(".", 1)[1] if ref.startswith("custom_providers.") else ref
        resolved = _resolve_custom_provider(name, config)
        if resolved:
            return resolved

    provider = (spec.get("provider") or "").strip()
    api_key = (spec.get("api_key") or "").strip()
    if not api_key:
        key_env = (spec.get("key_env") or spec.get("api_key_env") or "").strip()
        if key_env:
            api_key = os.getenv(key_env, "").strip()
    mapping = {
        "provider": provider,
        "model": (spec.get("model") or "").strip(),
        "base_url": (spec.get("base_url") or "").strip(),
        "api_key": api_key,
        "api_mode": (spec.get("api_mode") or "").strip(),
    }
    return _enrich_provider_runtime(mapping)


def resolve_all_tier_models(models_cfg: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for tier in ("classifier", "edge", "cloud"):
        if tier in models_cfg:
            out[tier] = resolve_tier_mapping(models_cfg[tier], config)
    return out


def _resolve_tier_context_length(
    tier: str,
    hg: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """Return context length for a hybrid-gateway tier (edge or cloud)."""
    cfg = hg if hg is not None else load_full_config()
    if not cfg:
        return None
    spec = (cfg.get("models") or {}).get(tier) or {}
    model = (spec.get("model") or "").strip()
    if not model:
        return None
    try:
        from hermes_cli.config import get_compatible_custom_providers, load_config

        full = load_config()
        custom = (
            get_compatible_custom_providers(full)
            if isinstance(full, dict)
            else None
        )
    except Exception:
        custom = None
    try:
        from agent.model_metadata import get_model_context_length

        return get_model_context_length(
            model,
            base_url=(spec.get("base_url") or "").strip(),
            api_key=(spec.get("api_key") or "").strip(),
            provider=(spec.get("provider") or "").strip(),
            custom_providers=custom,
        )
    except Exception:
        return None


def resolve_edge_context_length(hg: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """Return the edge tier model context length for compression budgeting."""
    return _resolve_tier_context_length("edge", hg)


def resolve_cloud_context_length(hg: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """Return the cloud tier model context length for compression budgeting."""
    return _resolve_tier_context_length("cloud", hg)


def load_full_config() -> Optional[Dict[str, Any]]:
    from .config_schema import load_hybrid_config

    hg = load_hybrid_config()
    if not hg:
        return None
    try:
        from hermes_cli.config import load_config

        full = load_config()
    except Exception:
        full = {}
    raw_models = {}
    if isinstance(full, dict):
        raw_hg = full.get("hybrid_gateway") or {}
        if isinstance(raw_hg, dict):
            raw_models = raw_hg.get("models") or {}
    hg["models"] = resolve_all_tier_models(
        raw_models if isinstance(raw_models, dict) else {},
        full if isinstance(full, dict) else {},
    )
    return hg

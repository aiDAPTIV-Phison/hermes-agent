"""Tests for hybrid-gateway router."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_PLUGIN_ROOT = Path(__file__).resolve().parents[2] / "plugins" / "hybrid-gateway"
_PKG = "hermes_hybrid_gateway_test"


def _load_module(name: str, filename: str, package: ModuleType) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"{_PKG}.{name}",
        _PLUGIN_ROOT / filename,
        submodule_search_locations=[str(_PLUGIN_ROOT)],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = _PKG
    sys.modules[f"{_PKG}.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_router_package():
    pkg = ModuleType(_PKG)
    pkg.__path__ = [str(_PLUGIN_ROOT)]
    pkg.__package__ = _PKG
    sys.modules[_PKG] = pkg
    _load_module("classifier_prompt", "classifier_prompt.py", pkg)
    return _load_module("router", "router.py", pkg)


_router = _load_router_package()
apply_policy = _router.apply_policy
apply_post_rules = _router.apply_post_rules
route = _router.route
validate_policy_array = _router.validate_policy_array


def test_validate_policy_array():
    assert validate_policy_array(["classifier", "classifier", "edge", "cloud", "cloud"])
    assert validate_policy_array(None) is None
    assert validate_policy_array(["edge", "edge"]) is None


def test_apply_policy_l2():
    assert apply_policy("trivial", "cost-optimize-L2") == "classifier"
    assert apply_policy("moderate", "cost-optimize-L2") == "edge"
    assert apply_policy("expert", "cost-optimize-L2") == "cloud"


def test_post_rules_skills_cap():
    out = apply_post_rules(
        {"complexity": "simple", "skills": ["a", "b", "c", "d"]},
    )
    assert out["complexity"] == "complex"


def test_route_policy_array():
    cfg = {
        "routing": {
            "policyArray": ["classifier", "classifier", "edge", "cloud", "cloud"],
        },
        "models": {
            "classifier": {"provider": "custom", "model": "small"},
            "edge": {"provider": "custom", "model": "edge-m"},
            "cloud": {"provider": "openrouter", "model": "big"},
        },
    }
    decision = route({"complexity": "moderate", "skills": ["coding"]}, cfg)
    assert decision["tier"] == "edge"
    assert decision["model"] == "edge-m"


def test_route_skill_force_cloud():
    cfg = {
        "routing": {
            "policy": "cost-optimize-L2",
            "skillRoutes": [
                {"skillPattern": "image-gen", "forceTier": "cloud"},
            ],
        },
        "models": {
            "classifier": {"provider": "custom", "model": "small"},
            "edge": {"provider": "custom", "model": "edge-m"},
            "cloud": {"provider": "openrouter", "model": "big"},
        },
    }
    decision = route({"complexity": "trivial", "skills": ["image-gen"]}, cfg)
    assert decision["tier"] == "cloud"
    assert decision["model"] == "big"

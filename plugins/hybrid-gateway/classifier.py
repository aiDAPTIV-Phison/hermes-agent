"""Complexity classifier via auxiliary LLM or heuristic."""



from __future__ import annotations



import json

import logging

import re

import time

from typing import Any, Dict, Optional, Tuple



from .classifier_prompt import CLASSIFIER_SYSTEM_PROMPT

from .classifier_thinking import (

    build_gemma4_nothink_prompt,

    extract_llm_text,

    resolve_thinking_strategy,

    system_prompt_with_nothink,

)

from .heuristic import heuristic_classify



logger = logging.getLogger(__name__)



_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}





def _parse_json(text: str) -> Optional[Dict[str, Any]]:

    if not text:

        return None

    text = text.strip()

    try:

        data = json.loads(text)

        if isinstance(data, dict) and data.get("complexity"):

            skills = data.get("skills") or []

            return {

                "complexity": str(data["complexity"]),

                "skills": [str(s) for s in skills] if isinstance(skills, list) else [],

            }

    except json.JSONDecodeError:

        pass

    m = re.search(r"\{[^{}]*\"complexity\"[^{}]*\}", text, re.DOTALL)

    if m:

        try:

            data = json.loads(m.group(0))

            if isinstance(data, dict) and data.get("complexity"):

                skills = data.get("skills") or []

                return {

                    "complexity": str(data["complexity"]),

                    "skills": [str(s) for s in skills] if isinstance(skills, list) else [],

                }

        except json.JSONDecodeError:

            pass

    return None





def _resolve_classifier_system_prompt(classifier_cfg: Dict[str, Any]) -> str:

    override = classifier_cfg.get("systemPrompt")

    if isinstance(override, str) and override.strip():

        return override.strip()

    return CLASSIFIER_SYSTEM_PROMPT





def _resolve_classifier_model(task: str) -> str:

    from agent.auxiliary_client import _resolve_task_provider_model



    _, model, _, _, _ = _resolve_task_provider_model(task)

    return model or ""





def _call_classifier_llm(

    user_text: str,

    classifier_cfg: Dict[str, Any],

) -> str:

    """Invoke auxiliary classifier; returns raw model text (may be empty)."""

    task = classifier_cfg.get("auxiliary_task") or "classifier"

    sys_prompt = _resolve_classifier_system_prompt(classifier_cfg)

    disable_thinking = bool(classifier_cfg.get("disableThinking", False))



    from agent.auxiliary_client import (

        _get_task_timeout,

        call_llm,

        get_text_auxiliary_client,

    )



    if not disable_thinking:

        response = call_llm(

            task=task,

            messages=[

                {"role": "system", "content": sys_prompt},

                {"role": "user", "content": user_text},

            ],

            temperature=0,

            max_tokens=256,

        )

        return extract_llm_text(response)



    model = _resolve_classifier_model(task)

    strategy = resolve_thinking_strategy(

        model,

        classifier_cfg.get("thinkingStrategy"),

    )

    logger.info("hybrid-gateway: thinking disabled, strategy=%s", strategy)

    timeout = _get_task_timeout(task)



    if strategy == "gemma4-raw":

        client, final_model = get_text_auxiliary_client(task)

        if client is None:

            raise RuntimeError(f"No auxiliary client for classifier task={task}")

        system_with_nothink = system_prompt_with_nothink(sys_prompt)

        raw_prompt = build_gemma4_nothink_prompt(system_with_nothink, user_text)

        response = client.completions.create(

            model=final_model,

            prompt=raw_prompt,

            max_tokens=200,

            temperature=0,

            stop=["<turn|>", "<|turn>"],

            timeout=timeout,

        )

        return extract_llm_text(response)



    system_with_nothink = system_prompt_with_nothink(sys_prompt)

    response = call_llm(

        task=task,

        messages=[

            {"role": "system", "content": system_with_nothink},

            {"role": "user", "content": user_text},

        ],

        temperature=0,

        max_tokens=200,

        extra_body={"chat_template_kwargs": {"enable_thinking": False}},

    )

    return extract_llm_text(response)





def classify(user_text: str, classifier_cfg: Dict[str, Any]) -> Dict[str, Any]:

    if (classifier_cfg.get("mode") or "auxiliary").strip().lower() == "heuristic":

        return heuristic_classify(user_text)



    cache_enabled = classifier_cfg.get("cacheEnabled", True)

    ttl = int(classifier_cfg.get("cacheTtlSeconds") or 300)

    if cache_enabled and user_text in _CACHE:

        ts, cached = _CACHE[user_text]

        if time.monotonic() - ts < ttl:

            logger.info("hybrid-gateway: classifier cache hit")

            return dict(cached)



    task = classifier_cfg.get("auxiliary_task") or "classifier"

    try:

        logger.info("hybrid-gateway: calling auxiliary classifier (task=%s)", task)

        content = _call_classifier_llm(user_text, classifier_cfg)

        parsed = _parse_json(content)

        if parsed:

            logger.info(

                "hybrid-gateway: classified complexity=%s skills=%s",

                parsed.get("complexity"),

                parsed.get("skills"),

            )

            if cache_enabled:

                _CACHE[user_text] = (time.monotonic(), parsed)

            return parsed

        logger.warning("hybrid-gateway: classifier returned invalid JSON, using heuristic")

    except Exception as exc:

        logger.warning("hybrid-gateway: classifier LLM failed (%s), using heuristic", exc)



    result = heuristic_classify(user_text)

    logger.info(

        "hybrid-gateway: heuristic complexity=%s skills=%s",

        result.get("complexity"),

        result.get("skills"),

    )

    if cache_enabled:

        _CACHE[user_text] = (time.monotonic(), result)

    return result



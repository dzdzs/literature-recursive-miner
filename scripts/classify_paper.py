#!/usr/bin/env python3
"""LLM-based runtime relevance/work_type classification helpers."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


class ClassificationError(RuntimeError):
    pass


def has_openai_api_key() -> bool:
    return bool(_get_api_key())


def _get_api_key() -> str:
    # Support both common names.
    return os.getenv("OPENAI_API_KEY", "").strip() or os.getenv("API_KEY", "").strip()


def _resolve_model(model: str | None) -> str:
    if model and model.strip():
        return model.strip()
    return os.getenv("OPENAI_MODEL", "").strip() or "gpt-4.1-mini"


def _resolve_chat_url(api_base_url: str | None) -> str:
    raw = (api_base_url or os.getenv("API_BASE_URL", "") or DEFAULT_OPENAI_BASE_URL).strip().rstrip("/")
    lowered = raw.lower()
    if lowered.endswith("/chat/completions"):
        return raw
    if lowered.endswith("/v1"):
        return f"{raw}/chat/completions"
    return f"{raw}/v1/chat/completions"


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int = 60) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ClassificationError(f"HTTP {exc.code}: {detail}") from exc


def _call_llm_json(*, model: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
    api_key = _get_api_key()
    if not api_key:
        raise ClassificationError("OPENAI_API_KEY (or API_KEY) is not set")

    model_name = _resolve_model(model)
    chat_url = _resolve_chat_url(None)
    payload = {
        "model": model_name,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = _post_json(chat_url, payload=payload, headers=headers)
    content = response["choices"][0]["message"]["content"]
    return json.loads(content)


def classify_with_llm(
    *,
    title: str,
    abstract: str,
    work_type_instruction: str,
    model: str = "gpt-4.1-mini",
    full_text_snippet: str = "",
) -> dict[str, Any]:
    """Return labels JSON. On failure, status is llm_failed with empty labels."""
    system_prompt = (
        "You are a strict paper classifier. "
        "Follow the provided work_type instruction exactly. "
        "Return only valid json with keys labels (string array) and rationale (string)."
    )
    user_prompt = (
        "Classify this paper using the runtime instruction. "
        "Use multi-label output when appropriate. Return json only.\n\n"
        f"Runtime instruction:\n{work_type_instruction}\n\n"
        f"Title:\n{title}\n\n"
        f"Abstract:\n{abstract or '(empty)'}\n\n"
        f"Optional full-text snippet:\n{full_text_snippet or '(not provided)'}"
    )

    try:
        parsed = _call_llm_json(model=model, system_prompt=system_prompt, user_prompt=user_prompt)
        labels = parsed.get("labels")
        rationale = parsed.get("rationale", "")
        if not isinstance(labels, list) or not all(isinstance(item, str) for item in labels):
            raise ClassificationError("Invalid labels format from LLM")
        if not isinstance(rationale, str):
            rationale = str(rationale)
        return {"labels": labels, "rationale": rationale, "status": "ok"}
    except Exception as exc:
        return {
            "labels": [],
            "rationale": f"LLM classification failed: {exc}",
            "status": "llm_failed",
        }


def classify_relevance_with_llm(
    *,
    title: str,
    abstract: str,
    theme: str,
    keywords: list[str],
    filter_conditions: list[str],
    relevance_instruction: str,
    model: str = "gpt-4.1-mini",
    full_text_snippet: str = "",
) -> dict[str, Any]:
    """
    Runtime relevance filter. Theme and extra constraints come from user input each run.
    """
    system_prompt = (
        "You are a conservative-keep paper relevance judge. "
        "Follow user runtime conditions exactly. "
        "If evidence is insufficient but potentially relevant, prefer keep. "
        "Return only valid json with keys: is_relevant (boolean), "
        "multi_turn_signal (yes|no|unclear), rationale (string)."
    )
    user_prompt = (
        "Judge whether this paper should be kept. Return json only.\n\n"
        f"Theme:\n{theme}\n\n"
        f"Keywords:\n{', '.join(keywords) if keywords else '(none)'}\n\n"
        f"Additional filter conditions:\n"
        f"{'; '.join(filter_conditions) if filter_conditions else '(none)'}\n\n"
        f"Runtime relevance instruction:\n{relevance_instruction}\n\n"
        f"Title:\n{title}\n\n"
        f"Abstract:\n{abstract or '(empty)'}\n\n"
        f"Optional full-text snippet:\n{full_text_snippet or '(not provided)'}"
    )
    try:
        parsed = _call_llm_json(model=model, system_prompt=system_prompt, user_prompt=user_prompt)
        is_relevant = parsed.get("is_relevant")
        multi_turn_signal = parsed.get("multi_turn_signal", "unclear")
        rationale = parsed.get("rationale", "")
        if not isinstance(is_relevant, bool):
            raise ClassificationError("Invalid is_relevant format from LLM")
        if multi_turn_signal not in {"yes", "no", "unclear"}:
            multi_turn_signal = "unclear"
        if not isinstance(rationale, str):
            rationale = str(rationale)
        return {
            "is_relevant": is_relevant,
            "multi_turn_signal": multi_turn_signal,
            "rationale": rationale,
            "status": "ok",
        }
    except Exception as exc:
        return {
            "is_relevant": False,
            "multi_turn_signal": "unclear",
            "rationale": f"LLM relevance classification failed: {exc}",
            "status": "llm_failed",
        }

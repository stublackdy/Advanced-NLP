"""
Gemini LLM Interface Module
===========================

Provides async chat completion access to the Gemini 2.5 Flash Lite model.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pipmaster as pm

if not pm.is_installed("google-generativeai"):
    pm.install("google-generativeai")

import google.generativeai as genai
from google.api_core.exceptions import DeadlineExceeded, ResourceExhausted, ServiceUnavailable
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from minirag.utils import locate_json_string_body_from_string, logger


def _convert_history_messages(history_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for message in history_messages or []:
        role = message.get("role", "user")
        content = message.get("content", "")
        if not content:
            continue
        if role == "assistant":
            role = "model"
        elif role not in {"user", "model"}:
            role = "user"
        converted.append({"role": role, "parts": [{"text": content}]})
    return converted


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable, DeadlineExceeded)),
)
async def gemini_25_flash_lite_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: List[Dict[str, Any]] | None = None,
    keyword_extraction: bool | None = False,
    **kwargs: Any,
) -> str:
    api_key = kwargs.pop("api_key", None) or os.environ.get("GEMINI_API_KEY")
    base_url = kwargs.pop("base_url", None)
    generation_config = kwargs.pop("generation_config", None)
    safety_settings = kwargs.pop("safety_settings", None)
    model_name = kwargs.pop("model", "gemini-2.5-flash-lite")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")

    client_options = {"api_endpoint": base_url} if base_url else None

    genai.configure(api_key=api_key, client_options=client_options)

    model_kwargs: Dict[str, Any] = {}
    if system_prompt:
        model_kwargs["system_instruction"] = system_prompt
    if generation_config:
        model_kwargs["generation_config"] = generation_config
    if safety_settings:
        model_kwargs["safety_settings"] = safety_settings

    model = genai.GenerativeModel(model_name, **model_kwargs)

    history = _convert_history_messages(history_messages or [])
    contents = history + [{"role": "user", "parts": [{"text": prompt}]}]

    response = await model.generate_content_async(contents=contents)

    text_response = getattr(response, "text", None)
    if not text_response and getattr(response, "candidates", None):
        candidate = response.candidates[0]
        parts = getattr(candidate, "content", None)
        if parts and getattr(parts, "parts", None):
            text_response = "".join(
                getattr(part, "text", "") for part in parts.parts if getattr(part, "text", None)
            )

    if not text_response:
        logger.warning("Empty response from Gemini model %s", model_name)
        text_response = ""

    if keyword_extraction:
        return locate_json_string_body_from_string(text_response)

    return text_response
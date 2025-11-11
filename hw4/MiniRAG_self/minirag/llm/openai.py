"""
OpenAI LLM Interface Module
==========================

This module provides interfaces for interacting with openai's language models,
including text generation and embedding capabilities.

Author: Lightrag team
Created: 2024-01-24
License: MIT License

Copyright (c) 2024 Lightrag

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

Version: 1.0.0

Change Log:
- 1.0.0 (2024-01-24): Initial release
    * Added async chat completion support
    * Added embedding generation
    * Added stream response capability

Dependencies:
    - openai
    - numpy
    - pipmaster
    - Python >= 3.10

Usage:
    from llm_interfaces.openai import openai_model_complete, openai_embed
"""

__version__ = "1.0.0"
__author__ = "lightrag Team"
__status__ = "Production"


import json

import sys
import os
from google import genai
from google.genai import types
import time


if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator
import pipmaster as pm  # Pipmaster for dynamic library install

# install specific modules
if not pm.is_installed("openai"):
    pm.install("openai")

from openai import (
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from minirag.utils import (
    wrap_embedding_func_with_attrs,
    locate_json_string_body_from_string,
    safe_unicode_decode,
    logger,
)
from pydantic import BaseModel
from typing import List
import numpy as np
from typing import Union


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)

class GPTKeywordExtractionFormat(BaseModel):
    high_level_keywords: List[str]
    low_level_keywords: List[str]

async def openai_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    base_url=None,
    api_key=None,
    **kwargs,
    ) -> str:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if base_url==None:
        base_url = os.environ["OPENAI_API_BASE"]
    openai_async_client = (
        AsyncOpenAI() if base_url is None else AsyncOpenAI(base_url=base_url)
    )
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # 添加日志输出
    logger.debug("===== Query Input to LLM =====")
    logger.debug(f"Query: {prompt}")
    logger.debug(f"System prompt: {system_prompt}")
    logger.debug("Full context:")

    print("===== Query Input to LLM =====")
    print(f"Query: {prompt}")
    print(f"System prompt: {system_prompt}")
    print("Full context:")

    if "response_format" in kwargs:
        response = await openai_async_client.beta.chat.completions.parse(
            model=model, messages=messages, **kwargs
        )
    else:
        response = await openai_async_client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )

    if hasattr(response, "__aiter__"):

        async def inner():
            async for chunk in response:
                content = chunk.choices[0].delta.content
                if content is None:
                    continue
                if r"\u" in content:
                    content = safe_unicode_decode(content.encode("utf-8"))
                yield content

        return inner()
    else:
        if not response or not hasattr(response, "choices") or not response.choices:
            logger.error("No valid choices returned. Full response: %s", response)
            return ""  # or raise a more specific exception
        content = response.choices[0].message.content
        if content is None:
            logger.error("The message content is None. Full response: %s", response)
            return ""
        if r"\u" in content:
            content = safe_unicode_decode(content.encode("utf-8"))
        return content

async def gemini_complete_if_cache(
    model,
    contents,
    generate_content_config,
    ) -> str:
    # if api_key:
    #     os.environ["GOOGLE_API_KEY"] = api_key

    # genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))

    # client = genai.Client(
    # api_key=os.environ["GOOGLE_API_KEY"],
    # )
    client = genai.Client(
    api_key=os.environ["GOOGLE_API_KEY"],
    )
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )

    return response.text

    # # Build dynamically formatted messages, similar to OpenAI structure
    # full_prompt = ""
    # if system_prompt:
    #     full_prompt += f"System: {system_prompt}\n\n"
    # for msg in history_messages:
    #     full_prompt += f"{msg['role'].title()}: {msg['content']}\n"
    # full_prompt += f"User: {prompt}"

    # print(f"GEMINI PROMPT:\n\n\n {full_prompt}")

    # try:
    #     response = genai.GenerativeModel(model).generate_content(full_prompt)
    #     return response.text if response.text else ""
    # except Exception as e:
    #     logger.error(f"Gemini API error: {e}")
    #     return f"Error: {str(e)}"

async def openai_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> Union[str, AsyncIterator[str]]:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = "json"
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat
    return await openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def nvidia_openai_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    result = await openai_complete_if_cache(
        "nvidia/llama-3.1-nemotron-70b-instruct",  # context length 128k
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url="https://integrate.api.nvidia.com/v1",
        **kwargs,
    )
    if keyword_extraction:  # TODO: use JSON API
        return locate_json_string_body_from_string(result)
    return result

async def openrouter_openai_complete(
    prompt, 
    system_prompt=None, 
    history_messages=[], 
    keyword_extraction=False, 
    api_key: str = None, 
    **kwargs,
) -> str:
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key

    keyword_extraction = kwargs.pop("keyword_extraction", None)
    result = await openai_complete_if_cache(
        "Qwen-7B-Chat",  # change accordingly
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        **kwargs,
    )
    if keyword_extraction:  # TODO: use JSON API
        return locate_json_string_body_from_string(result)
    return result

@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def openai_embed(
    texts: list[str],
    model: str = "text-embedding-3-small",
    base_url: str = None,
    api_key: str = None,
) -> np.ndarray:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    openai_async_client = (
        AsyncOpenAI() if base_url is None else AsyncOpenAI(base_url=base_url)
    )
    response = await openai_async_client.embeddings.create(
        model=model, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])



async def gemini_llm_complete(
         prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    keyword_extraction = kwargs.pop("keyword_extraction", None)

    contents = []

    if system_prompt:
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=system_prompt)]))

    for msg in history_messages:
        contents.append(types.Content(role="model", parts=[types.Part.from_text(text=msg["content"])]))

    # print(type(prompt))
    # print(prompt.shape)

    # with open("temp.txt", "w") as f:
    #     f.write(prompt)
    # with open

    # tools = [
    # types.Tool(googleSearch=types.GoogleSearch(
    # )),
    # ]
    
    contents.append(types.Content(role="user", parts=[
                types.Part.from_text(text=prompt),
            ],))


    generate_content_config = types.GenerateContentConfig(
    thinking_config = types.ThinkingConfig(
        thinking_budget=-1,
    ),
    )
    

    result = await gemini_complete_if_cache(
        "gemini-2.5-flash-lite",
        contents,
        generate_content_config,
    )

    if keyword_extraction:
        return locate_json_string_body_from_string(result)

    # print(result)
    print(f"GEMINI CALL: {len(result)}")
    time.sleep(6)

    # res_dict={
    #     "prompt": contents,
    #     "result": result
    # }

    # print(res_dict)

    # with open("temp.json", "r") as f:
    #     res_list=json.loads(f.read())

    # res_list.append(res_dict)

    # with open("temp.json", "w") as f:
    #     json.dump(res_list, f)
        

    return result


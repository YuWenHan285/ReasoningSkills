from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from litellm import acompletion, aembedding
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class LLMClient:
    def __init__(self, *, model: str, api_base: str | None = None, api_key: str | None = None, temperature: float = 0.0, max_tokens: int | None = None) -> None:
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

    @retry(
        reraise=True,
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(Exception),
    )
    async def complete(
        self,
        *,
        user_prompt: str,
        system_prompt: str | None = None,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        response = await acompletion(
            model=self.model,
            api_base=self.api_base,
            api_key=self.api_key,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            response_format=response_format,
            extra_body=extra_body,
            stream=False,
        )
        choice = response.choices[0]
        text = choice.message.content if hasattr(choice, "message") else choice["message"]["content"]
        usage = getattr(response, "usage", None)
        if usage is None:
            usage_dict: dict[str, Any] = {}
        else:
            usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
        return {
            "text": text,
            "finish_reason": getattr(choice, "finish_reason", None),
            "usage": usage_dict,
            "raw": response.model_dump() if hasattr(response, "model_dump") else dict(response),
        }


class EmbeddingClient:
    def __init__(self, *, model: str, api_base: str | None = None, api_key: str | None = None, dimensions: int | None = None) -> None:
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.dimensions = dimensions

    @retry(
        reraise=True,
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(Exception),
    )
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = await aembedding(
            model=self.model,
            api_base=self.api_base,
            api_key=self.api_key,
            input=texts,
            dimensions=self.dimensions,
        )
        data = response.data if hasattr(response, "data") else response["data"]
        return [item.embedding if hasattr(item, "embedding") else item["embedding"] for item in data]


async def gather_limited(coros: list, concurrency: int) -> list[Any]:
    semaphore = asyncio.Semaphore(concurrency)

    async def _wrapped(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*[_wrapped(c) for c in coros])


class JSONParser:
    @staticmethod
    def parse(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        return json.loads(text)


def env_or_value(value: str | None, env_name: str | None) -> str | None:
    if value:
        return value
    if env_name:
        return os.getenv(env_name)
    return None

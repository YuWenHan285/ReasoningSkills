from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential



class LLMClient:
    def __init__(self, *, model: str, api_base: str | None = None, api_key: str | None = None, temperature: float = 0.0, max_tokens: int | None = None) -> None:
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_base)
        return self._client

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
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": False,
        }
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if response_format is not None:
            request["response_format"] = response_format
        if extra_body is not None:
            request["extra_body"] = extra_body

        response = await self.client.chat.completions.create(**request)
        choice = response.choices[0]
        text = choice.message.content or ""
        usage = response.usage
        usage_dict = usage.model_dump() if usage is not None else {}
        return {
            "text": text,
            "finish_reason": choice.finish_reason,
            "usage": usage_dict,
            "raw": response.model_dump(),
        }


class EmbeddingClient:
    def __init__(self, *, model: str, api_base: str | None = None, api_key: str | None = None, dimensions: int | None = None) -> None:
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.dimensions = dimensions
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_base)
        return self._client

    @retry(
        reraise=True,
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(Exception),
    )
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        request: dict[str, Any] = {
            "model": self.model,
            "input": texts,
        }
        if self.dimensions is not None:
            request["dimensions"] = self.dimensions

        response = await self.client.embeddings.create(**request)
        return [item.embedding for item in response.data]


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

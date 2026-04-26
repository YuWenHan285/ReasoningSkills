from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


def _openai_model_name(model: str) -> str:
    if model.startswith("openai/"):
        return model.removeprefix("openai/")
    return model


def _message_extra(message: Any, key: str) -> Any:
    value = getattr(message, key, None)
    if value is not None:
        return value
    model_extra = getattr(message, "model_extra", None)
    if isinstance(model_extra, dict):
        return model_extra.get(key)
    if isinstance(message, dict):
        return message.get(key)
    return None


def _message_text(message: Any) -> tuple[str, str | None]:
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    reasoning_content = _message_extra(message, "reasoning_content")

    text = content or ""
    if reasoning_content:
        text = f"<think>\n{reasoning_content}\n</think>\n{text}"
    return text, reasoning_content


def _compact_completion_raw(response: Any, choice: Any, reasoning_content: str | None) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "model": getattr(response, "model", None),
        "created": getattr(response, "created", None),
    }
    if reasoning_content:
        raw["has_reasoning_content"] = True
    return {key: value for key, value in raw.items() if value is not None}


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
            "model": _openai_model_name(self.model),
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
        text, reasoning_content = _message_text(choice.message)
        usage = response.usage
        usage_dict = usage.model_dump() if usage is not None else {}
        return {
            "text": text,
            "finish_reason": choice.finish_reason,
            "usage": usage_dict,
            "raw": _compact_completion_raw(response, choice, reasoning_content),
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
            "model": _openai_model_name(self.model),
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


def env_or_value(value: str | None, env_name: str | None) -> str | None:
    if value:
        return value
    if env_name:
        return os.getenv(env_name)
    return None

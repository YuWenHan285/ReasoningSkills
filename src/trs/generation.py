from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from tqdm.asyncio import tqdm_asyncio

from trs.clients import LLMClient
from trs.prompts import DIRECT_PROMPT
from trs.schemas import DatasetExample, GenerationRecord
from trs.utils.io import append_jsonl, read_jsonl


@dataclass
class GenerationConfig:
    output_path: str
    concurrency: int = 8
    overwrite: bool = False


def _existing_ids(path: str | Path) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    return {row["example_id"] for row in read_jsonl(p) if "example_id" in row}


async def generate_trajectories(
    *,
    examples: list[DatasetExample],
    llm: LLMClient,
    output_path: str,
    system_prompt: str | None = None,
    prompt_template: str = DIRECT_PROMPT,
    concurrency: int = 8,
    overwrite: bool = False,
    generation_kwargs: dict | None = None,
) -> list[GenerationRecord]:
    generation_kwargs = generation_kwargs or {}
    done = set() if overwrite else _existing_ids(output_path)

    semaphore = asyncio.Semaphore(concurrency)
    results: list[GenerationRecord] = []

    async def _one(example: DatasetExample) -> GenerationRecord | None:
        if example.id in done:
            return None
        async with semaphore:
            prompt = prompt_template.format(problem=example.question)
            response = await llm.complete(
                user_prompt=prompt,
                system_prompt=system_prompt,
                **generation_kwargs,
            )
            record = GenerationRecord(
                example_id=example.id,
                question=example.question,
                answer=example.answer,
                model=llm.model,
                system_prompt=system_prompt,
                user_prompt=prompt,
                raw_text=response["text"],
                finish_reason=response.get("finish_reason"),
                usage=response.get("usage", {}),
                metadata={"raw": response.get("raw", {})},
            )
            append_jsonl(record.model_dump(), output_path)
            return record

    tasks = [_one(ex) for ex in examples]
    for rec in await tqdm_asyncio.gather(*tasks):
        if rec is not None:
            results.append(rec)
    return results

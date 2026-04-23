from __future__ import annotations

import asyncio
from pathlib import Path

from tqdm.asyncio import tqdm_asyncio

from trs.clients import JSONParser, LLMClient
from trs.prompts import EXTRACTION_SYSTEM, EXTRACTION_USER
from trs.schemas import ExtractedSkill, GenerationRecord, SkillCard
from trs.utils.io import append_jsonl, read_jsonl


def load_generation_records(path: str | Path) -> list[GenerationRecord]:
    return [GenerationRecord.model_validate(x) for x in read_jsonl(path)]


async def extract_skills(
    *,
    records: list[GenerationRecord],
    llm: LLMClient,
    output_path: str,
    concurrency: int = 8,
) -> list[ExtractedSkill]:
    semaphore = asyncio.Semaphore(concurrency)
    extracted: list[ExtractedSkill] = []

    async def _one(record: GenerationRecord) -> ExtractedSkill:
        correctness = record.metadata.get("correctness")
        prompt = EXTRACTION_USER.format(
            question=record.question,
            answer=record.answer,
            trajectory=record.raw_text,
            correctness=correctness,
        )
        async with semaphore:
            response = await llm.complete(
                user_prompt=prompt,
                system_prompt=EXTRACTION_SYSTEM,
                response_format={"type": "json_object"},
            )
        parsed = JSONParser.parse(response["text"])
        skill = ExtractedSkill(
            source_example_id=record.example_id,
            source_question=record.question,
            source_answer=record.answer,
            source_model=record.model,
            correctness=correctness,
            keywords=parsed["keywords"],
            card=SkillCard.model_validate(parsed["card"]),
            raw_response=response["text"],
            metadata={
                "usage": response.get("usage", {}),
                "source_usage": record.usage,
            },
        )
        append_jsonl(skill.model_dump(), output_path)
        return skill

    tasks = [_one(record) for record in records]
    for item in await tqdm_asyncio.gather(*tasks):
        extracted.append(item)
    return extracted

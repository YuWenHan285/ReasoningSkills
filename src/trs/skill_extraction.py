from __future__ import annotations

import asyncio
from pathlib import Path
import xml.etree.ElementTree as ET

from tqdm.asyncio import tqdm_asyncio

from trs.clients import LLMClient
from trs.prompts import EXTRACTION_SYSTEM, EXTRACTION_USER
from trs.schemas import ExtractedSkill, GenerationRecord
from trs.utils.io import append_jsonl, read_jsonl


def load_generation_records(path: str | Path) -> list[GenerationRecord]:
    return [GenerationRecord.model_validate(x) for x in read_jsonl(path)]


def _strip_xml_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("xml"):
            text = text[3:].strip()
    if text.startswith("<think>") and "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text


def parse_extracted_skill_xml(text: str) -> tuple[str, list[str]]:
    wrapped = f"<root>{_strip_xml_fence(text)}</root>"
    try:
        root = ET.fromstring(wrapped)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid skill extraction XML: {exc}") from exc

    learned_heuristic = (root.findtext("learned_heuristic") or "").strip()
    keywords_text = (root.findtext("retrieval_keywords") or "").strip()
    retrieval_keywords = [kw.strip() for kw in keywords_text.split(",") if kw.strip()]
    if not learned_heuristic or not retrieval_keywords:
        return None
    return learned_heuristic, retrieval_keywords


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
            )
        learned_heuristic, retrieval_keywords = parse_extracted_skill_xml(response["text"])
        skill = ExtractedSkill(
            source_example_id=record.example_id,
            source_question=record.question,
            source_answer=record.answer,
            source_model=record.model,
            correctness=correctness,
            learned_heuristic=learned_heuristic,
            retrieval_keywords=retrieval_keywords,
            raw_response=response["text"],
            metadata={
                "usage": response.get("usage", {}),
                "source_usage": record.usage,
            },
        )
        if skill:
            append_jsonl(skill.model_dump(), output_path)
        return skill

    tasks = [_one(record) for record in records]
    for item in await tqdm_asyncio.gather(*tasks):
        extracted.append(item)
    return extracted

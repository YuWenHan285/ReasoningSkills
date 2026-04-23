from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from tqdm.asyncio import tqdm_asyncio

from trs.clients import LLMClient
from trs.prompts import render_trs_prompt
from trs.retrieval.bm25 import BM25Index
from trs.retrieval.dense import DenseIndex
from trs.retrieval.hybrid import hybrid_search
from trs.schemas import DatasetExample, ExtractedSkill, InferenceRecord
from trs.utils.io import append_jsonl


@dataclass
class RetrievalConfig:
    backend: str = "bm25"
    top_k: int = 1
    bm25_weight: float = 0.5
    dense_weight: float = 0.5


class SkillRetriever:
    def __init__(self, skills: list[ExtractedSkill], *, backend: str = "bm25", dense_model_name: str = "BAAI/bge-m3") -> None:
        self.skills = skills
        self.backend = backend
        self.bm25 = BM25Index.build(skills) if backend in {"bm25", "hybrid"} else None
        self.dense = DenseIndex.build_local(skills, dense_model_name) if backend in {"dense", "hybrid"} else None

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
    ) -> list[tuple[ExtractedSkill, float, str]]:
        if self.backend == "bm25":
            assert self.bm25 is not None
            hits = self.bm25.search(query, top_k)
            return [(self.skills[i], score, "bm25") for i, score in hits]
        if self.backend == "dense":
            assert self.dense is not None
            hits = self.dense.search_local(query, top_k)
            return [(self.skills[i], score, "dense") for i, score in hits]
        assert self.bm25 is not None and self.dense is not None
        hits = hybrid_search(
            query=query,
            bm25=self.bm25,
            dense=self.dense,
            top_k=top_k,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )
        return [(self.skills[i], score, "hybrid") for i, score in hits]


def format_solving_hints(hits: list[tuple[ExtractedSkill, float, str]]) -> str:
    if not hits:
        return ""
    blocks = []
    for idx, (skill, _score, _backend) in enumerate(hits, start=1):
        blocks.append(f"Skill {idx}.\n{skill.card.render()}")
    return "\n\n".join(blocks)


async def run_inference(
    *,
    examples: list[DatasetExample],
    retriever: SkillRetriever,
    llm: LLMClient,
    output_path: str,
    prompt_style: str,
    budget: int | None,
    concurrency: int = 8,
    generation_kwargs: dict | None = None,
    top_k: int = 1,
    bm25_weight: float = 0.5,
    dense_weight: float = 0.5,
) -> list[InferenceRecord]:
    generation_kwargs = generation_kwargs or {}
    semaphore = asyncio.Semaphore(concurrency)
    results: list[InferenceRecord] = []

    async def _one(example: DatasetExample) -> InferenceRecord:
        hits = retriever.retrieve(
            example.question,
            top_k=top_k,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )
        solving_hints = format_solving_hints(hits)
        prompt = render_trs_prompt(
            problem=example.question,
            solving_hints=solving_hints,
            style=prompt_style,
            budget=budget,
        )
        async with semaphore:
            response = await llm.complete(user_prompt=prompt, **generation_kwargs)
        record = InferenceRecord(
            example_id=example.id,
            question=example.question,
            prompt_style=prompt_style,
            model=llm.model,
            retrieved=[
                {
                    "source_example_id": skill.source_example_id,
                    "score": score,
                    "backend": backend,
                    "keywords": skill.keywords,
                    "card": skill.card.model_dump(),
                }
                for skill, score, backend in hits
            ],
            final_prompt=prompt,
            response_text=response["text"],
            usage=response.get("usage", {}),
            metadata={"raw": response.get("raw", {})},
        )
        append_jsonl(record.model_dump(), output_path)
        return record

    tasks = [_one(example) for example in examples]
    for item in await tqdm_asyncio.gather(*tasks):
        results.append(item)
    return results

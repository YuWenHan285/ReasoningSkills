from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class DatasetExample(BaseModel):
    id: str
    question: str
    answer: str | None = None
    metadata: dict = Field(default_factory=dict)


class GenerationRecord(BaseModel):
    example_id: str
    question: str
    answer: str | None = None
    model: str
    system_prompt: str | None = None
    user_prompt: str
    raw_text: str
    finish_reason: str | None = None
    usage: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)


class ExtractedSkill(BaseModel):
    source_example_id: str
    source_question: str
    source_answer: str | None = None
    source_model: str
    correctness: int | None = None
    learned_heuristic: str
    retrieval_keywords: list[str] = Field(default_factory=list, min_length=1)
    raw_response: str
    metadata: dict = Field(default_factory=dict)

    @field_validator("learned_heuristic")
    @classmethod
    def clean_learned_heuristic(cls, value: str) -> str:
        return value.strip()

    @field_validator("retrieval_keywords")
    @classmethod
    def clean_retrieval_keywords(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for kw in value:
            kw = kw.strip()
            if not kw:
                continue
            lowered = kw.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            out.append(kw)
        return out

    def key_text(self) -> str:
        return f"{self.source_question}\n{', '.join(self.retrieval_keywords)}"

    def value_text(self) -> str:
        return self.learned_heuristic


class RetrievalMode(str):
    pass


class RetrievalHit(BaseModel):
    skill_id: str
    score: float
    backend: Literal["bm25", "dense", "hybrid"]
    learned_heuristic: str
    key_text: str
    metadata: dict = Field(default_factory=dict)


class InferenceRecord(BaseModel):
    example_id: str
    question: str
    prompt_style: str
    model: str
    retrieved: list[dict] = Field(default_factory=list)
    final_prompt: str
    response_text: str
    usage: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)

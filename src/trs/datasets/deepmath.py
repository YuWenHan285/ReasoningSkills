from __future__ import annotations

from typing import Iterable

from datasets import Dataset, load_dataset

from trs.schemas import DatasetExample


DEEP_MATH_CANDIDATE_COLUMNS = {
    "question": ["question", "prompt", "problem"],
    "answer": ["final_answer", "answer", "solution"],
}


def _pick(record: dict, candidates: list[str]) -> str | None:
    for key in candidates:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def load_deepmath(
    *,
    dataset_name: str = "zwhe99/DeepMath-103K",
    split: str = "train",
    start: int = 0,
    limit: int | None = None,
) -> list[DatasetExample]:
    ds = load_dataset(dataset_name, split=split)
    if start:
        ds = ds.select(range(start, len(ds)))
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    examples: list[DatasetExample] = []
    for i, row in enumerate(ds):
        question = _pick(row, DEEP_MATH_CANDIDATE_COLUMNS["question"])
        answer = _pick(row, DEEP_MATH_CANDIDATE_COLUMNS["answer"])
        if not question:
            continue
        ex_id = str(row.get("id", row.get("uuid", f"deepmath-{i}")))
        metadata = {k: v for k, v in row.items() if k not in {"question", "prompt", "problem", "final_answer", "answer", "solution"}}
        examples.append(DatasetExample(id=ex_id, question=question, answer=answer, metadata=metadata))
    return examples

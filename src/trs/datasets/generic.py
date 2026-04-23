from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset

from trs.schemas import DatasetExample
from trs.utils.io import read_jsonl


QUESTION_CANDIDATES = ["question", "prompt", "problem", "description"]
ANSWER_CANDIDATES = ["answer", "final_answer", "solution", "target"]


def _choose(record: dict, candidates: list[str]) -> str | None:
    for c in candidates:
        value = record.get(c)
        if value not in (None, ""):
            return value
    return None


def load_generic_local(path: str | Path, limit: int | None = None) -> list[DatasetExample]:
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = read_jsonl(path)
    elif path.suffix == ".json":
        import json
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
    elif path.suffix == ".parquet":
        rows = pd.read_parquet(path).to_dict("records")
    else:
        raise ValueError(f"Unsupported file type: {path}")

    if limit is not None:
        rows = rows[:limit]

    out: list[DatasetExample] = []
    for i, row in enumerate(rows):
        question = _choose(row, QUESTION_CANDIDATES)
        if not question:
            continue
        answer = _choose(row, ANSWER_CANDIDATES)
        ex_id = str(row.get("id", row.get("uuid", row.get("example_id", f"local-{i}"))))
        out.append(DatasetExample(id=ex_id, question=question, answer=answer, metadata=row))
    return out


def load_hf_dataset(dataset_name: str, split: str, limit: int | None = None) -> list[DatasetExample]:
    ds = load_dataset(dataset_name, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    out: list[DatasetExample] = []
    for i, row in enumerate(ds):
        question = _choose(row, QUESTION_CANDIDATES)
        if not question:
            continue
        answer = _choose(row, ANSWER_CANDIDATES)
        ex_id = str(row.get("id", row.get("uuid", f"hf-{i}")))
        out.append(DatasetExample(id=ex_id, question=question, answer=answer, metadata=row))
    return out

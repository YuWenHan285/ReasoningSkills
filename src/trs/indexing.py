from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from trs.schemas import ExtractedSkill
from trs.utils.io import read_jsonl, write_json, write_jsonl


def load_skills(path: str | Path) -> list[ExtractedSkill]:
    return [ExtractedSkill.model_validate(x) for x in read_jsonl(path)]


def save_skills(skills: list[ExtractedSkill], path: str | Path) -> None:
    write_jsonl([s.model_dump() for s in skills], path)


def save_faiss_index(index: faiss.Index, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def save_dense_metadata(skills: list[ExtractedSkill], matrix: np.ndarray, path: str | Path, model_name: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix(".npy"), matrix)
    write_json(
        {
            "model_name": model_name,
            "skills": [s.model_dump() for s in skills],
            "matrix_file": path.with_suffix(".npy").name,
        },
        path.with_suffix(".json"),
    )

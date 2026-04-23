from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from trs.schemas import ExtractedSkill
from trs.utils.text import keyword_tokenize


@dataclass
class BM25Index:
    skills: list[ExtractedSkill]
    bm25: BM25Okapi
    corpus_tokens: list[list[str]]

    @classmethod
    def build(cls, skills: list[ExtractedSkill]) -> "BM25Index":
        corpus_tokens = [keyword_tokenize(skill.key_text()) for skill in skills]
        bm25 = BM25Okapi(corpus_tokens)
        return cls(skills=skills, bm25=bm25, corpus_tokens=corpus_tokens)

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        q = keyword_tokenize(query)
        scores = np.asarray(self.bm25.get_scores(q), dtype=float)
        if scores.size == 0:
            return []
        idx = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in idx]

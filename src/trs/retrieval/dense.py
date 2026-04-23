from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from trs.clients import EmbeddingClient
from trs.schemas import ExtractedSkill


@dataclass
class DenseIndex:
    skills: list[ExtractedSkill]
    matrix: np.ndarray
    index: faiss.IndexFlatIP
    model_name: str
    backend: str
    local_model: SentenceTransformer | None = None
    api_client: EmbeddingClient | None = None

    @classmethod
    def build_local(cls, skills: list[ExtractedSkill], model_name: str = "BAAI/bge-m3") -> "DenseIndex":
        model = SentenceTransformer(model_name)
        texts = [s.key_text() for s in skills]
        matrix = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix.astype(np.float32))
        return cls(skills=skills, matrix=matrix.astype(np.float32), index=index, model_name=model_name, backend="local", local_model=model)

    @classmethod
    def build_from_embeddings(cls, skills: list[ExtractedSkill], vectors: np.ndarray, model_name: str) -> "DenseIndex":
        vectors = vectors.astype(np.float32)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(skills=skills, matrix=vectors, index=index, model_name=model_name, backend="api")

    def search_local(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if self.local_model is None:
            raise RuntimeError("Local model not initialized")
        q = self.local_model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        scores, indices = self.index.search(q.astype(np.float32), top_k)
        return [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]

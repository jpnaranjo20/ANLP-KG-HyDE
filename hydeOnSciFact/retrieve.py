from __future__ import annotations

from typing import List, Tuple

import faiss
import numpy as np


def build_faiss_index(corpus_vectors: np.ndarray) -> faiss.Index:
    dim = corpus_vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(corpus_vectors)
    return index


def retrieve_top_k(
    index: faiss.Index,
    query_vectors: np.ndarray,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    scores, indices = index.search(query_vectors, top_k)
    return scores, indices

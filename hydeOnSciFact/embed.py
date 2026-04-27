from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBED_BATCH_SIZE, EMBED_MODEL


class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str], batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

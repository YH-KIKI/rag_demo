from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_demo.model_registry import EMBEDDING_MODELS


@lru_cache(maxsize=len(EMBEDDING_MODELS))
def _get_model(model_key: str) -> SentenceTransformer:
    config = EMBEDDING_MODELS[model_key]
    return SentenceTransformer(config.model_name, trust_remote_code=config.trust_remote_code)


def embed_passages(texts: list[str], model_key: str = "e5") -> np.ndarray:
    config = EMBEDDING_MODELS[model_key]
    model = _get_model(model_key)
    prefixed = [f"{config.passage_prefix}{text}" for text in texts]
    return model.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True)


def embed_query(text: str, model_key: str = "e5") -> np.ndarray:
    config = EMBEDDING_MODELS[model_key]
    model = _get_model(model_key)
    return model.encode(f"{config.query_prefix}{text}", normalize_embeddings=True, convert_to_numpy=True)

from functools import lru_cache

from transformers import AutoTokenizer

from rag_demo.config import settings


@lru_cache(maxsize=1)
def _get_tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(settings.embedding_model_name)


def chunk_text(
    text: str,
    max_tokens: int = settings.chunk_max_tokens,
    overlap_tokens: int = settings.chunk_overlap_tokens,
) -> list[str]:
    text = text.strip()
    if not text:
        return []

    tokenizer = _get_tokenizer()
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return [text]

    stride = max_tokens - overlap_tokens
    chunks = []
    for start in range(0, len(token_ids), stride):
        window = token_ids[start : start + max_tokens]
        if not window:
            break
        chunks.append(tokenizer.decode(window, skip_special_tokens=True))
        if start + max_tokens >= len(token_ids):
            break
    return chunks

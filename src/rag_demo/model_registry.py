from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingModelConfig:
    key: str
    model_name: str
    dim: int
    query_prefix: str
    passage_prefix: str
    table_name: str


EMBEDDING_MODELS: dict[str, EmbeddingModelConfig] = {
    "e5": EmbeddingModelConfig(
        key="e5",
        model_name="intfloat/multilingual-e5-base",
        dim=768,
        query_prefix="query: ",
        passage_prefix="passage: ",
        table_name="document_chunks",
    ),
    "bge_m3": EmbeddingModelConfig(
        key="bge_m3",
        model_name="BAAI/bge-m3",
        dim=1024,
        query_prefix="",
        passage_prefix="",
        table_name="document_chunks_bge_m3",
    ),
}

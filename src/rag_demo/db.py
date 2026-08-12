import psycopg
from pgvector.psycopg import register_vector

from rag_demo.config import settings
from rag_demo.model_registry import EMBEDDING_MODELS

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS {table_name} (
    id BIGSERIAL PRIMARY KEY,
    doc_id TEXT NOT NULL,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR({embedding_dim}) NOT NULL,
    metadata JSONB DEFAULT '{{}}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx
    ON {table_name} USING hnsw (embedding vector_cosine_ops);
"""


def get_connection() -> psycopg.Connection:
    conn = psycopg.connect(settings.database_url, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_schema(model_key: str = "e5") -> None:
    config = EMBEDDING_MODELS[model_key]
    with get_connection() as conn:
        conn.execute(SCHEMA_SQL.format(table_name=config.table_name, embedding_dim=config.dim))

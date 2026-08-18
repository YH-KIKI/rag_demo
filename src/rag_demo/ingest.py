import argparse
import logging
import random
import sys

from datasets import load_dataset
from psycopg.types.json import Jsonb

from rag_demo.chunking import chunk_text
from rag_demo.config import settings
from rag_demo.db import get_connection, init_schema
from rag_demo.embeddings import embed_passages
from rag_demo.model_registry import EMBEDDING_MODELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 32


def _insert_batch(conn, rows: list[dict], model_key: str) -> None:
    if not rows:
        return
    table_name = EMBEDDING_MODELS[model_key].table_name
    contents = [row["content"] for row in rows]
    embeddings = embed_passages(contents, model_key)
    with conn.cursor() as cur:
        cur.executemany(
            f"""
            INSERT INTO {table_name} (doc_id, chunk_index, content, embedding, metadata)
            VALUES (%s, %s, %s, %s, %s)
            """,
            [
                (row["doc_id"], row["chunk_index"], row["content"], embedding, Jsonb(row["metadata"]))
                for row, embedding in zip(rows, embeddings)
            ],
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(EMBEDDING_MODELS), default="e5")
    parser.add_argument(
        "--limit", type=int, default=None, help="무작위 샘플링할 행 수 (미지정 시 전체 적재)"
    )
    parser.add_argument("--seed", type=int, default=42, help="--limit 샘플링 시드")
    args = parser.parse_args()
    model_key = args.model

    if not settings.hf_dataset_path:
        logger.error("HF_DATASET_PATH가 설정되지 않았습니다. .env를 확인하세요.")
        sys.exit(1)

    init_schema(model_key)

    logger.info("데이터셋 로딩: %s (split=%s)", settings.hf_dataset_path, settings.hf_dataset_split)
    dataset = load_dataset(settings.hf_dataset_path, split=settings.hf_dataset_split)

    text_column = settings.hf_dataset_text_column
    if text_column not in dataset.column_names:
        logger.error(
            "텍스트 컬럼 '%s'을(를) 찾을 수 없습니다. 사용 가능한 컬럼: %s",
            text_column,
            dataset.column_names,
        )
        sys.exit(1)

    original_indices = list(range(len(dataset)))
    if args.limit is not None and args.limit < len(dataset):
        original_indices = sorted(random.Random(args.seed).sample(original_indices, args.limit))
        dataset = dataset.select(original_indices)
        logger.info("샘플링된 %d행 적재 (seed=%d)", len(original_indices), args.seed)

    conn = get_connection()
    pending_rows: list[dict] = []
    total_chunks = 0

    try:
        for pos, row in enumerate(dataset):
            row_index = original_indices[pos]
            text = row.get(text_column)
            if not text:
                continue

            other_fields = {k: v for k, v in row.items() if k != text_column}
            metadata = {**other_fields, "source": "huggingface"}
            for chunk_index, chunk in enumerate(chunk_text(text)):
                pending_rows.append(
                    {
                        "doc_id": str(row_index),
                        "chunk_index": chunk_index,
                        "content": chunk,
                        "metadata": metadata,
                    }
                )

            if len(pending_rows) >= BATCH_SIZE:
                _insert_batch(conn, pending_rows, model_key)
                total_chunks += len(pending_rows)
                pending_rows = []
                if row_index % 500 == 0:
                    logger.info("row %d 처리, 누적 청크 %d개", row_index, total_chunks)

        _insert_batch(conn, pending_rows, model_key)
        total_chunks += len(pending_rows)
    finally:
        conn.close()

    logger.info("완료: 총 %d개 청크 적재", total_chunks)


if __name__ == "__main__":
    main()

import argparse
import json
import logging
import sys
import zipfile
from pathlib import Path

from psycopg.types.json import Jsonb

from rag_demo.chunking import chunk_text
from rag_demo.db import get_connection, init_schema
from rag_demo.embeddings import embed_passages
from rag_demo.model_registry import EMBEDDING_MODELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 32

DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "59.반려견 성장 및 질병 관련 말뭉치 데이터"
    / "3.개방데이터"
    / "1.데이터"
    / "Training"
    / "02.라벨링데이터"
)


def _iter_qa_records(data_dir: Path):
    zip_paths = sorted(data_dir.glob("TL_질의응답데이터_*.zip"))
    if not zip_paths:
        logger.error("QA zip 파일을 찾을 수 없습니다: %s", data_dir)
        sys.exit(1)

    for zip_path in zip_paths:
        category = zip_path.stem.replace("TL_질의응답데이터_", "")
        with zipfile.ZipFile(zip_path) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".json")]
            logger.info("%s: %d개 QA 파일", zip_path.name, len(names))
            for name in names:
                with zf.open(name) as f:
                    try:
                        record = json.load(f)
                    except json.JSONDecodeError:
                        logger.warning("JSON 파싱 실패: %s/%s", zip_path.name, name)
                        continue

                qa = record.get("qa") or {}
                meta = record.get("meta") or {}
                question = (qa.get("input") or "").strip()
                answer = (qa.get("output") or "").strip()
                if not question or not answer:
                    continue

                doc_id = Path(name).stem
                content = f"질문: {question}\n답변: {answer}"
                metadata = {
                    "source": "aihub_dog_qa",
                    "category": category,
                    "department": meta.get("department", ""),
                    "disease": meta.get("disease", ""),
                    "life_cycle": meta.get("lifeCycle", ""),
                }
                yield doc_id, content, metadata


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
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--models", nargs="+", choices=list(EMBEDDING_MODELS), default=list(EMBEDDING_MODELS)
    )
    args = parser.parse_args()

    logger.info("QA 레코드 파싱 및 청킹 시작: %s", args.data_dir)
    all_rows: list[dict] = []
    doc_count = 0
    for doc_id, content, metadata in _iter_qa_records(args.data_dir):
        doc_count += 1
        for chunk_index, chunk in enumerate(chunk_text(content)):
            all_rows.append(
                {
                    "doc_id": f"aihub_qa_{doc_id}",
                    "chunk_index": chunk_index,
                    "content": chunk,
                    "metadata": metadata,
                }
            )
        if doc_count % 2000 == 0:
            logger.info("문서 %d개 처리, 누적 청크 %d개", doc_count, len(all_rows))

    logger.info("파싱 완료: 문서 %d개 -> 청크 %d개", doc_count, len(all_rows))
    if not all_rows:
        logger.error("적재할 청크가 없습니다.")
        sys.exit(1)

    for model_key in args.models:
        logger.info("[%s] 스키마 준비", model_key)
        init_schema(model_key)

        conn = get_connection()
        total_inserted = 0
        try:
            for start in range(0, len(all_rows), BATCH_SIZE):
                batch = all_rows[start : start + BATCH_SIZE]
                _insert_batch(conn, batch, model_key)
                total_inserted += len(batch)
                batch_num = start // BATCH_SIZE
                if batch_num % 50 == 0:
                    logger.info("[%s] %d / %d 청크 적재", model_key, total_inserted, len(all_rows))
        finally:
            conn.close()
        logger.info("[%s] 완료: %d개 청크 적재", model_key, total_inserted)


if __name__ == "__main__":
    main()

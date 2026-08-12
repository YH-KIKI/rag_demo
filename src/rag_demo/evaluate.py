import argparse
import json
import random
import statistics
import time

from rag_demo.db import get_connection
from rag_demo.embeddings import embed_query
from rag_demo.model_registry import EMBEDDING_MODELS

TOP_K = 20
RECALL_KS = (1, 5, 10)


def _sample_queries(n: int, seed: int) -> list[dict]:
    with get_connection() as conn:
        conn.execute("SELECT setseed(%s)", ((seed % 1000) / 1000,))
        rows = conn.execute(
            """
            SELECT doc_id, chunk_index, metadata->>'question' AS question
            FROM document_chunks
            WHERE metadata->>'question' IS NOT NULL AND metadata->>'question' != ''
            ORDER BY random()
            LIMIT %s
            """,
            (n,),
        ).fetchall()
    return [{"doc_id": r[0], "chunk_index": r[1], "question": r[2]} for r in rows]


def _rank_of_target(table_name: str, query_embedding, doc_id: str, chunk_index: int) -> int | None:
    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT doc_id, chunk_index
            FROM {table_name}
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (query_embedding, TOP_K),
        ).fetchall()
    for i, (d, c) in enumerate(rows, start=1):
        if d == doc_id and c == chunk_index:
            return i
    return None


def evaluate_model(model_key: str, samples: list[dict]) -> dict:
    table_name = EMBEDDING_MODELS[model_key].table_name
    ranks: list[int | None] = []
    latencies_ms: list[float] = []

    for s in samples:
        t0 = time.perf_counter()
        embedding = embed_query(s["question"], model_key)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        ranks.append(_rank_of_target(table_name, embedding, s["doc_id"], s["chunk_index"]))

    n = len(ranks)
    result = {"model": model_key, "n": n}
    for k in RECALL_KS:
        result[f"recall@{k}"] = sum(1 for r in ranks if r is not None and r <= k) / n
    result[f"mrr@{TOP_K}"] = sum((1 / r) if r is not None else 0 for r in ranks) / n
    result["not_found_rate"] = sum(1 for r in ranks if r is None) / n
    result["avg_query_ms"] = statistics.mean(latencies_ms)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    samples = _sample_queries(args.n, args.seed)

    results = [evaluate_model(key, samples) for key in EMBEDDING_MODELS]
    print(json.dumps({"n_samples": len(samples), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

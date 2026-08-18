import argparse
import json
import random
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from rag_demo.db import get_connection
from rag_demo.embeddings import embed_query
from rag_demo.model_registry import EMBEDDING_MODELS
from rag_demo.report import generate_report

TOP_K = 20
RECALL_KS = (1, 3, 5, 10)
RUNS_DIR = Path(__file__).resolve().parents[2] / "docs" / "eval-runs"

# 대화형 잡담/인사/메타 질문 등 검색 평가 대상이 될 수 없는 "질문"을 걸러낸다.
# (원본 HF 데이터셋이 멀티턴 대화의 한 턴을 question으로 잘못 뽑아온 경우가 많음)
_JUNK_QUESTION_RE = re.compile(
    r"^(hi|hello|hey|hiya|howdy|yo|bonjour|hola|hallo)\b"
    r"|^(thanks?|thank\s?you|thank\s?u|many\s+thanks|thx)\b"
    r"|^(ok|okay|sure|yes|no|got it|good|great|perfect|cool|nice|wow)[.,!\s]*$"
    r"|^(ok|okay)\s*,?\s*(thanks?|thank\s?you)\b"
    r"|^(no\s+thanks|no,?\s+thank\s+you|yes\s+please)"
    r"|^not\s+(at\s+this\s+time|right\s+now|really|sure)\b"
    r"|^(who are you|what model|which model|do you use gpt|do you speak|can you speak|can you answer)\b"
    r"|^(summarize|interpret|analyze|give (me )?more|list \d+ more|more facts"
    r"|write great code|do my homework|new chat)\b",
    re.IGNORECASE,
)


def _is_junk_question(question: str) -> bool:
    normalized = question.strip()
    if not normalized:
        return True
    if re.fullmatch(r"[\W\d_]*", normalized):  # 숫자/기호만 있는 경우
        return True
    alnum_only = re.sub(r"[^0-9A-Za-z가-힣]", "", normalized)
    if len(alnum_only) <= 2:  # "C", "Lg", "et" 같은 의미 없는 짧은 토큰
        return True
    return bool(_JUNK_QUESTION_RE.search(normalized))


def _sample_queries(n: int, seed: int) -> tuple[list[dict], int]:
    """doc_id당 하나의 질문만, 잡담/인사 등 잡음을 제외하고 샘플링한다.

    같은 doc_id의 모든 청크는 동일한 question 메타데이터를 공유하므로(문서 단위 라벨),
    청크 행을 그대로 무작위 추출하면 같은 질문이 여러 번 뽑히거나 특정 chunk_index로
    치우칠 수 있다. doc_id 단위로 먼저 중복 제거한 뒤 샘플링한다.

    반환값: (샘플 목록, 잡음으로 제외된 개수)
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT ON (doc_id) doc_id, metadata->>'question' AS question
            FROM document_chunks
            WHERE metadata->>'question' IS NOT NULL AND metadata->>'question' != ''
            ORDER BY doc_id
            """
        ).fetchall()

    candidates = [{"doc_id": r[0], "question": r[1]} for r in rows]
    valid = [c for c in candidates if not _is_junk_question(c["question"])]
    filtered_count = len(candidates) - len(valid)

    sampled = random.Random(seed).sample(valid, min(n, len(valid)))
    return sampled, filtered_count


def _rank_of_target(table_name: str, query_embedding, doc_id: str) -> int | None:
    """정답 doc_id에 속한 청크가 상위 결과에 처음 등장하는 순위를 반환한다.

    같은 문서의 청크는 전부 같은 질문의 정답으로 간주한다(doc-level relevance) —
    특정 chunk_index 하나만 정답으로 채점하면 라벨이 문서 내 어느 청크로 뽑혔는지에
    따라 결과가 좌우되는 문제가 있다.
    """
    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT doc_id
            FROM {table_name}
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (query_embedding, TOP_K),
        ).fetchall()
    for i, (d,) in enumerate(rows, start=1):
        if d == doc_id:
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
        ranks.append(_rank_of_target(table_name, embedding, s["doc_id"]))

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

    samples, filtered_junk_count = _sample_queries(args.n, args.seed)

    results = [evaluate_model(key, samples) for key in EMBEDDING_MODELS]
    output = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "n_samples": len(samples),
        "filtered_junk_count": filtered_junk_count,
        "results": results,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = RUNS_DIR / f"{run_id}.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved: {out_path}")

    report_path = generate_report(output, run_id)
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()

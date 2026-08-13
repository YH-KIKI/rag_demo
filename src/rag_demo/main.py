from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_demo.config import settings
from rag_demo.db import get_connection
from rag_demo.embeddings import embed_query
from rag_demo.llm import generate_answer, translate_query_to_english
from rag_demo.model_registry import EMBEDDING_MODELS

app = FastAPI(title=settings.app_name, debug=settings.debug)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": f"{settings.app_name} is running"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    model: Literal["e5", "bge_m3"] = "e5"


class SearchResult(BaseModel):
    doc_id: str
    chunk_index: int
    content: str
    metadata: dict
    distance: float


def _search(search_query: str, top_k: int, model_key: str) -> list[SearchResult]:
    table_name = EMBEDDING_MODELS[model_key].table_name

    search_query_en = translate_query_to_english(search_query)
    queries = {search_query}
    queries.add(search_query_en)

    candidates: dict[tuple[str, int], SearchResult] = {}
    with get_connection() as conn:
        for query_text in queries:
            query_embedding = embed_query(query_text, model_key)
            rows = conn.execute(
                f"""
                SELECT doc_id, chunk_index, content, metadata, embedding <=> %s AS distance
                FROM {table_name}
                ORDER BY distance
                LIMIT %s
                """,
                (query_embedding, top_k),
            ).fetchall()
            for row in rows:
                key = (row[0], row[1])
                if key not in candidates or row[4] < candidates[key].distance:
                    candidates[key] = SearchResult(
                        doc_id=row[0],
                        chunk_index=row[1],
                        content=row[2],
                        metadata=row[3],
                        distance=row[4],
                    )

    return sorted(candidates.values(), key=lambda r: r.distance)[:top_k]


@app.post("/search")
async def search(request: SearchRequest) -> list[SearchResult]:
    return _search(request.query, request.top_k, request.model)


class AskResponse(BaseModel):
    answer: str
    sources: list[SearchResult]


@app.post("/ask")
async def ask(request: SearchRequest) -> AskResponse:
    sources = _search(request.query, request.top_k, request.model)
    answer = generate_answer(request.query, [s.content for s in sources])
    return AskResponse(answer=answer, sources=sources)


class CompareRequest(BaseModel):
    query: str
    top_k: int = 5


class ModelResult(BaseModel):
    model: str
    answer: str
    sources: list[SearchResult]


class CompareResponse(BaseModel):
    results: list[ModelResult]


@app.post("/compare")
async def compare(request: CompareRequest) -> CompareResponse:
    results = []
    for model_key in EMBEDDING_MODELS:
        sources = _search(request.query, request.top_k, model_key)
        answer = generate_answer(request.query, [s.content for s in sources])
        results.append(ModelResult(model=model_key, answer=answer, sources=sources))
    return CompareResponse(results=results)

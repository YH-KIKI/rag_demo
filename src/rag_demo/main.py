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


def _search(search_query_en: str, top_k: int, model_key: str) -> list[SearchResult]:
    table_name = EMBEDDING_MODELS[model_key].table_name
    query_embedding = embed_query(search_query_en, model_key)
    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT doc_id, chunk_index, content, metadata, embedding <=> %s AS distance
            FROM {table_name}
            ORDER BY distance
            LIMIT %s
            """,
            (query_embedding, top_k),
        ).fetchall()

    return [
        SearchResult(
            doc_id=row[0],
            chunk_index=row[1],
            content=row[2],
            metadata=row[3],
            distance=row[4],
        )
        for row in rows
    ]


@app.post("/search")
async def search(request: SearchRequest) -> list[SearchResult]:
    search_query = translate_query_to_english(request.query)
    return _search(search_query, request.top_k, request.model)


class AskResponse(BaseModel):
    answer: str
    sources: list[SearchResult]


@app.post("/ask")
async def ask(request: SearchRequest) -> AskResponse:
    search_query = translate_query_to_english(request.query)
    sources = _search(search_query, request.top_k, request.model)
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
    search_query = translate_query_to_english(request.query)
    results = []
    for model_key in EMBEDDING_MODELS:
        sources = _search(search_query, request.top_k, model_key)
        answer = generate_answer(request.query, [s.content for s in sources])
        results.append(ModelResult(model=model_key, answer=answer, sources=sources))
    return CompareResponse(results=results)

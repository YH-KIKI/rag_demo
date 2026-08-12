from fastapi import FastAPI

from rag_demo.config import settings

app = FastAPI(title=settings.app_name, debug=settings.debug)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": f"{settings.app_name} is running"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

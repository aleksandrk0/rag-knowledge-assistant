"""HTTP-API на FastAPI: /health, /ingest, /ask.

Запуск:  uvicorn api.main:app --reload  (из корня репозитория)
Swagger: http://localhost:8000/docs
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from config.settings import settings  # noqa: E402
from rag.factory import build_pipeline  # noqa: E402

app = FastAPI(title="RAG Knowledge Assistant", version="0.1.0")
_pipeline = build_pipeline(settings)


class IngestRequest(BaseModel):
    documents: list[dict]  # [{"source": str, "text": str}]


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    timings_ms: dict[str, float]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    docs = [(d["source"], d["text"]) for d in req.documents]
    n = _pipeline.ingest(docs)
    return {"indexed_chunks": n}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        ans = _pipeline.answer(req.question)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return AskResponse(answer=ans.text, sources=ans.sources, timings_ms=ans.timings_ms)

"""
app.py
FastAPI backend for the Data Detective RAG demo.

Serves:
  GET  /                 -> chat.html   (Screen 1, front-facing)
  GET  /engine           -> engine.html (Screen 2, "behind the scenes")
  POST /api/ask          -> runs retrieval + generation for one question
  GET  /api/stream       -> Server-Sent Events feed that engine.html subscribes to,
                             so Screen 2 updates the instant a question is asked on Screen 1

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os
import json
import asyncio
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from openai import AzureOpenAI
from qdrant_client import QdrantClient

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_API_VERSION = os.environ["AZURE_OPENAI_API_VERSION"]
AZURE_OPENAI_CHAT_DEPLOYMENT = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
QDRANT_COLLECTION = os.environ["QDRANT_COLLECTION"]

TOP_K = 5  # how many chunks to retrieve per question

SYSTEM_PROMPT = """You are the "Detective Bot" at a data science club table game called Data Detective.
You have access only to the retrieved case-file excerpts provided to you in each message.

Rules you must follow exactly:
- Answer ONLY using facts stated in the provided excerpts. Do not use outside knowledge.
- Do NOT guess, speculate, infer a conclusion, or name a "culprit." You do not know who is guilty and must never imply an answer to that question.
- If asked directly who did it, or whether a specific person is guilty, say plainly that the case files don't state who is responsible and that's for the investigator (the student) to determine.
- If the excerpts don't contain the answer to a question, say so honestly instead of making something up.
- Keep answers concise and factual, like a records clerk reading off a file, not a detective drawing conclusions.
"""

app = FastAPI(title="Data Detective RAG Backend")

aoai = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# --- simple in-process pub/sub so /engine can watch what /api/ask is doing live ---
subscribers: List[asyncio.Queue] = []


async def broadcast(event: dict):
    dead = []
    for q in subscribers:
        try:
            q.put_nowait(event)
        except Exception:
            dead.append(q)
    for q in dead:
        subscribers.remove(q)


class AskRequest(BaseModel):
    question: str


def embed_query(text: str) -> List[float]:
    resp = aoai.embeddings.create(model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT, input=[text])
    return resp.data[0].embedding


def retrieve(question: str, top_k: int = TOP_K):
    vector = embed_query(question)
    hits = qdrant.search(collection_name=QDRANT_COLLECTION, query_vector=vector, limit=top_k)
    results = []
    for h in hits:
        results.append({
            "doc_id": h.payload.get("doc_id"),
            "doc_title": h.payload.get("doc_title"),
            "chunk_index": h.payload.get("chunk_index"),
            "text": h.payload.get("text"),
            "score": round(float(h.score), 4),
        })
    return results


def generate_answer(question: str, retrieved: List[dict]) -> str:
    context_block = "\n\n---\n\n".join(
        f"[{r['doc_id']} chunk {r['chunk_index']}] (relevance {r['score']})\n{r['text']}"
        for r in retrieved
    )
    user_message = (
        f"Retrieved case-file excerpts:\n\n{context_block}\n\n"
        f"Student's question: {question}"
    )
    resp = aoai.chat.completions.create(
        model=AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        max_tokens=400,
    )
    return resp.choices[0].message.content


@app.post("/api/ask")
async def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        return {"error": "Empty question."}

    # Let Screen 2 know a question just came in
    await broadcast({"type": "question", "question": question})

    # Retrieval step
    retrieved = retrieve(question)
    await broadcast({"type": "retrieved", "question": question, "results": retrieved})

    # Generation step
    answer = generate_answer(question, retrieved)
    await broadcast({"type": "answer", "question": question, "answer": answer})

    return {"answer": answer, "retrieved": retrieved}


@app.get("/api/stream")
async def stream(request: Request):
    queue: asyncio.Queue = asyncio.Queue()
    subscribers.append(queue)

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                event = await queue.get()
                yield {"event": "message", "data": json.dumps(event)}
        finally:
            if queue in subscribers:
                subscribers.remove(queue)

    return EventSourceResponse(event_generator())


# --- static pages ---
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def chat_page():
    return FileResponse(os.path.join(STATIC_DIR, "chat.html"))


@app.get("/engine")
async def engine_page():
    return FileResponse(os.path.join(STATIC_DIR, "engine.html"))

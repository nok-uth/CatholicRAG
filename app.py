"""FastAPI web server for CatholicRAG."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pinecone import Pinecone
from pydantic import BaseModel

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL = "gpt-4o-mini"
INDEX_NAME = "catholic-rag"
TOP_K = 5

SYSTEM_PROMPT = """You are a knowledgeable guide to Catholic teaching and Scripture.
Answer the user's question using ONLY the context passages provided below.

Rules:
1. Answer using the context passages below. Every claim must cite its source inline:
   - For Catechism passages: [CCC N] where N is the paragraph number
   - For Bible verses: [Book Chapter:Verse]
2. Use all relevant passages in the context — even if they address the topic indirectly.
3. Only if the context is truly unrelated to the question, reply with exactly:
   "The provided context does not address this question."
4. Do not speculate or draw on outside knowledge.

CONTEXT:
{context}
"""

app = FastAPI(title="CatholicRAG")
app.mount("/static", StaticFiles(directory="static"), name="static")

openai_client: OpenAI | None = None
pinecone_index = None


@app.on_event("startup")
def startup():
    global openai_client, pinecone_index
    openai_key = os.environ.get("OPENAI_API_KEY")
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    if not openai_key or not pinecone_key:
        raise RuntimeError("OPENAI_API_KEY and PINECONE_API_KEY must be set.")
    openai_client = OpenAI(api_key=openai_key)
    pc = Pinecone(api_key=pinecone_key)
    pinecone_index = pc.Index(INDEX_NAME)


@app.get("/")
def index():
    return FileResponse(Path("static/index.html"))


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []


class Source(BaseModel):
    label: str
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Embed the question
    emb = openai_client.embeddings.create(
        model=EMBED_MODEL, input=[req.question]
    ).data[0].embedding

    # Retrieve from Pinecone
    results = pinecone_index.query(vector=emb, top_k=TOP_K, include_metadata=True)
    matches = [m["metadata"] for m in results["matches"]]

    # Build context string and source list
    context_parts: list[str] = []
    sources: list[Source] = []
    for m in matches:
        if m.get("source") == "CCC":
            label = f"CCC {m['paragraph']}"
            context_parts.append(f"[{label}] {m['text']}")
        else:
            label = f"{m['book']} {m['chapter']}:{m['verse']}"
            context_parts.append(f"[{label}] {m['text']}")
        sources.append(Source(label=label, text=m["text"]))

    context = "\n\n".join(context_parts)

    # Build messages with conversation history
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        *req.history,
        {"role": "user", "content": req.question},
    ]

    response = openai_client.chat.completions.create(
        model=GEN_MODEL,
        temperature=0.0,
        messages=messages,
    )

    return ChatResponse(
        answer=response.choices[0].message.content,
        sources=sources,
    )

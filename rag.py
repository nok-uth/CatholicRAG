"""RAG over the Catechism of the Catholic Church + NABRE Bible.

Uses OpenAI text-embedding-3-small for retrieval and gpt-4o-mini for generation.
Requires OPENAI_API_KEY and PINECONE_API_KEY in the environment (or .env file).
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

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


def embed_query(client: OpenAI, question: str) -> list[float]:
    response = client.embeddings.create(model=EMBED_MODEL, input=[question])
    return response.data[0].embedding


def retrieve(client: OpenAI, index, question: str, k: int = TOP_K) -> list[dict]:
    q_emb = embed_query(client, question)
    results = index.query(vector=q_emb, top_k=k, include_metadata=True)
    return [match["metadata"] for match in results["matches"]]


def format_context(matches: list[dict]) -> str:
    parts: list[str] = []
    for m in matches:
        if m.get("source") == "CCC":
            parts.append(f"[CCC {m['paragraph']}] {m['text']}")
        else:
            parts.append(f"[{m['book']} {m['chapter']}:{m['verse']}] {m['text']}")
    return "\n\n".join(parts)


def answer(client: OpenAI, question: str, matches: list[dict]) -> str:
    context = format_context(matches)
    response = client.chat.completions.create(
        model=GEN_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


def main() -> int:
    openai_key = os.environ.get("OPENAI_API_KEY")
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    if not openai_key:
        print("error: OPENAI_API_KEY not set.", file=sys.stderr)
        return 1
    if not pinecone_key:
        print("error: PINECONE_API_KEY not set.", file=sys.stderr)
        return 1

    openai_client = OpenAI(api_key=openai_key)
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(INDEX_NAME)

    stats = index.describe_index_stats()
    print(f"Connected to '{INDEX_NAME}' ({stats.total_vector_count:,} vectors)")
    print("\nAsk a question about Catholic teaching or Scripture.")
    print("Press Enter on an empty line or Ctrl+C to quit.\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break

        matches = retrieve(openai_client, index, question)
        print()
        print(answer(openai_client, question, matches))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

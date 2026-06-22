"""Test harness: ingest only the first 10 records and verify round-trip."""
from __future__ import annotations

import itertools
import os
import sys
import time

from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone

import ingest_data as I

load_dotenv()


def main() -> int:
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    if not (gemini_key and pinecone_key):
        print("error: missing GEMINI_API_KEY or PINECONE_API_KEY", file=sys.stderr)
        return 1

    genai_client = genai.Client(api_key=gemini_key)
    pc = Pinecone(api_key=pinecone_key)
    index = I.ensure_index(pc)

    records = (
        rec
        for src in (I.iter_ccc(I.CCC_PATH), I.iter_bible(I.BIBLE_PATH))
        for rec in src
    )
    sample = list(itertools.islice(records, 10))
    print("test records:", [r["id"] for r in sample])

    vectors = I.embed_texts(genai_client, [r["text"] for r in sample])
    print(f"embedded {len(vectors)} vectors, dim={len(vectors[0])}")

    upserts = [
        {"id": r["id"], "values": v, "metadata": r["metadata"]}
        for r, v in zip(sample, vectors)
    ]
    resp = index.upsert(vectors=upserts)
    print(f"upsert response: {resp}")

    print("waiting 3s for serverless index to settle...")
    time.sleep(3)

    fetched = index.fetch(ids=[sample[0]["id"]])
    vec = fetched.vectors[sample[0]["id"]]
    print(f"\nround-trip fetch for {sample[0]['id']}:")
    print(f"  metadata: {vec.metadata}")
    print(f"  values dim: {len(vec.values)}")

    print(f"\nindex stats: {index.describe_index_stats()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

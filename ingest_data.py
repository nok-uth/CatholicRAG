"""Embed the Catechism + NABRE Bible and upsert into a Pinecone serverless index.

Requires OPENAI_API_KEY and PINECONE_API_KEY in the environment (a `.env` file
at the working directory is loaded automatically).

Re-running is safe: vector IDs are deterministic, so upserts overwrite rather
than duplicate. If the run is interrupted, a checkpoint file (ingest_checkpoint.txt)
records how many records were successfully processed so the next run resumes
from that point rather than re-embedding from scratch.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
CCC_PATH = DATA_DIR / "catechism.json"
BIBLE_PATH = DATA_DIR / "nabre.json"

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
INDEX_NAME = "catholic-rag"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

EMBED_BATCH = 100
UPSERT_BATCH = 100
MIN_CYCLE_SECS = 1.0        # OpenAI limits are generous; 1s gap is plenty
MAX_EMBED_ATTEMPTS = 6
DEFAULT_RETRY_DELAY = 10.0

CHECKPOINT_PATH = Path("ingest_checkpoint.txt")


def sanitize_id(s: str) -> str:
    return re.sub(r"\s+", "_", s.strip())


def iter_ccc(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    for entry in data:
        para_id = entry["id"]
        text = entry["text"].strip()
        if not text:
            continue
        yield {
            "id": f"ccc-{para_id}",
            "text": text,
            "metadata": {"source": "CCC", "paragraph": para_id, "text": text},
        }


def iter_bible(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    for book in data:
        book_name = book["book"]
        for chapter in book["chapters"]:
            chap_num = chapter["chapter"]
            for verse in chapter["verses"]:
                verse_num = verse["verse"]
                text = verse["text"].strip()
                if not text:
                    continue
                yield {
                    "id": f"bible-{sanitize_id(book_name)}-{chap_num}-{verse_num}",
                    "text": text,
                    "metadata": {
                        "source": "Bible",
                        "book": book_name,
                        "chapter": chap_num,
                        "verse": verse_num,
                        "text": text,
                    },
                }


def batched(iterable: Iterable[dict], size: int) -> Iterator[list[dict]]:
    batch: list[dict] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def embed_with_backoff(client: OpenAI, texts: list[str]) -> list[list[float]]:
    last_exc: BaseException | None = None
    for attempt in range(1, MAX_EMBED_ATTEMPTS + 1):
        try:
            return embed_texts(client, texts)
        except Exception as exc:
            last_exc = exc
            if attempt == MAX_EMBED_ATTEMPTS:
                break
            delay = DEFAULT_RETRY_DELAY * attempt
            print(
                f"  embed error (attempt {attempt}/{MAX_EMBED_ATTEMPTS}); "
                f"sleeping {delay:.0f}s before retry: {exc}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def ensure_index(pc: Pinecone):
    existing = {i["name"] for i in pc.list_indexes()}
    if INDEX_NAME in existing:
        info = pc.describe_index(INDEX_NAME)
        if info.dimension != EMBED_DIM:
            print(
                f"existing index has dim={info.dimension}, need {EMBED_DIM} — "
                f"deleting and recreating..."
            )
            pc.delete_index(INDEX_NAME)
            existing.discard(INDEX_NAME)

    if INDEX_NAME not in existing:
        print(
            f"creating Pinecone index {INDEX_NAME!r} "
            f"(dim={EMBED_DIM}, cosine, serverless {PINECONE_CLOUD}/{PINECONE_REGION})..."
        )
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
    return pc.Index(INDEX_NAME)


def load_checkpoint() -> int:
    if CHECKPOINT_PATH.exists():
        try:
            n = int(CHECKPOINT_PATH.read_text().strip())
            print(f"checkpoint found: resuming after {n:,} records already processed")
            return n
        except (ValueError, OSError):
            pass
    return 0


def save_checkpoint(n: int) -> None:
    CHECKPOINT_PATH.write_text(str(n))


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
    index = ensure_index(pc)

    initial_stats = index.describe_index_stats()
    print(f"index at startup: {initial_stats.total_vector_count:,} vectors already stored")

    skip = load_checkpoint()

    all_records: Iterator[dict] = (
        rec for src in (iter_ccc(CCC_PATH), iter_bible(BIBLE_PATH)) for rec in src
    )
    if skip:
        all_records = itertools.islice(all_records, skip, None)

    total = skip
    start_batch = skip // EMBED_BATCH + 1

    for batch_num, batch in enumerate(batched(all_records, EMBED_BATCH), start=start_batch):
        cycle_start = time.monotonic()

        texts = [r["text"] for r in batch]
        vectors = embed_with_backoff(openai_client, texts)

        upserts = [
            {"id": r["id"], "values": v, "metadata": r["metadata"]}
            for r, v in zip(batch, vectors)
        ]
        for i in range(0, len(upserts), UPSERT_BATCH):
            index.upsert(vectors=upserts[i : i + UPSERT_BATCH])

        total += len(batch)
        save_checkpoint(total)

        print(
            f"batch {batch_num}: +{len(batch)} -> {total:,}/{37_956} total "
            f"(last={batch[-1]['id']})",
            flush=True,
        )

        elapsed = time.monotonic() - cycle_start
        wait = max(0.0, MIN_CYCLE_SECS - elapsed)
        if wait:
            time.sleep(wait)

    CHECKPOINT_PATH.unlink(missing_ok=True)
    print(f"\nDone. Upserted {total:,} vectors total.")
    print(f"Index stats: {index.describe_index_stats()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

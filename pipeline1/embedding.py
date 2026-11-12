"""Pipeline 1: embed BNS CSV chunks on CPU and store them in local Qdrant.

Install: python -m pip install "qdrant-client[fastembed]"
Run through main.py: python main.py index

This module rebuilds its named Qdrant collection each time you index.
It does not use PyTorch, a GPU, Docker, or a hosted API.
"""

import csv
import uuid
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "bns_chunks"
BATCH_SIZE = 32

TEXT_FIELDS = ("chunk_id", "record_id", "annexure", "section_in_bns", "chunk_text")
INT_FIELDS = (
    "chunk_index", "total_chunks", "section_number",
    "source_page_start", "source_page_end", "pdf_page_start", "pdf_page_end",
    "char_start", "char_end",
)
REQUIRED_FIELDS = set(TEXT_FIELDS + INT_FIELDS)


def embed_chunks(chunks_csv, qdrant_dir, cache_dir=None, batch_size=BATCH_SIZE):
    """Embed all rows in bns_chunks.csv, save vectors and source metadata.

    Returns the number of vectors stored. The CSV must already exist.
    """
    chunks_csv = Path(chunks_csv)
    qdrant_dir = Path(qdrant_dir)
    cache_dir = Path(cache_dir) if cache_dir else qdrant_dir.parent / "model_cache"

    if not chunks_csv.is_file():
        raise FileNotFoundError(f"Chunk CSV not found: {chunks_csv}")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    with chunks_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_FIELDS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing columns from chunk CSV: {sorted(missing)}")
        rows = list(reader)

    if not rows:
        raise ValueError("Chunk CSV is empty")

    ids = [row["chunk_id"].strip() for row in rows]
    if any(not chunk_id for chunk_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("Missing or duplicate chunk_id values")

    payloads = []
    for row in rows:
        if not row["chunk_text"].strip():
            raise ValueError(f"Empty chunk_text for {row['chunk_id']}")
        payload = {field: row[field] for field in TEXT_FIELDS}
        payload.update({field: int(row[field]) for field in INT_FIELDS})
        payloads.append(payload)

    # Generate vectors before modifying the existing database collection.
    cache_dir.mkdir(parents=True, exist_ok=True)
    embedding_model = TextEmbedding(model_name=MODEL_NAME, cache_dir=str(cache_dir))
    vectors = list(embedding_model.embed(
        [item["chunk_text"] for item in payloads], batch_size=batch_size
    ))
    
    # DEBUG: Print the first 3 generated embeddings
    for i in range(min(3, len(vectors))):
        print(f"\n{'=' * 60}")
        print(f"Chunk ID: {payloads[i]['chunk_id']}")
        print(f"Section: {payloads[i]['section_number']}")
        print(f"Chunk text: {payloads[i]['chunk_text'][:200]}")
        print(f"Embedding dimensions: {len(vectors[i])}")
        print(f"Embedding vector:\n{vectors[i].tolist()}")
        print(f"{'=' * 60}")

    if len(vectors) != len(payloads):
        raise RuntimeError("Embedding count does not match chunk count")
    dimension = len(vectors[0])
    if not dimension or any(len(vector) != dimension for vector in vectors):
        raise RuntimeError("Invalid embedding dimensions")

    qdrant_dir.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(qdrant_dir))
    try:
        # Local development: replace only our project's collection.
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=dimension, distance=models.Distance.COSINE
            ),
        )

        for start in range(0, len(payloads), batch_size):
            points = []
            for payload, vector in zip(
                payloads[start:start + batch_size],
                vectors[start:start + batch_size],
            ):
                # Stable UUID: the same chunk gets the same Qdrant ID.
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "bns/" + payload["chunk_id"]))
                points.append(models.PointStruct(
                    id=point_id, vector=vector.tolist(), payload=payload
                ))
            client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)

        count = client.count(collection_name=COLLECTION_NAME, exact=True).count
        if count != len(payloads):
            raise RuntimeError(f"Stored {count} vectors, expected {len(payloads)}")
    finally:
        client.close()  # Important: free the local Qdrant file lock.

    print(f"Embedded {count} chunks with {MODEL_NAME} ({dimension} dimensions)")
    print(f"Qdrant collection: {COLLECTION_NAME}")
    print(f"Local database: {qdrant_dir}")
    return count


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    embed_chunks(root / "data" / "bns_chunks.csv", root / "data" / "qdrant_store")

"""Ingestion pipeline: load AGORA CSVs, embed segment text, persist to Chroma.

Run once before querying:

    python ingest.py            # build the store (skips if already populated)
    python ingest.py --rebuild  # force a clean rebuild

We embed the `Text` field of each segment (the operative paragraph) and store
the document-level metadata needed for citations alongside each vector, so the
query path never has to re-join against documents.csv.
"""

from __future__ import annotations

import argparse
import os

import chromadb
import pandas as pd

from rag import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DATA_DIR,
    EMBED_MODEL_NAME,
    get_embedder,
)

BATCH_SIZE = 256  # Chroma add() batch size


def load_documents(data_dir: str) -> dict[str, dict]:
    """Map AGORA ID -> the metadata we cite (name, authority, status)."""
    path = os.path.join(data_dir, "documents.csv")
    # utf-8-sig strips the BOM on the first column header.
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, low_memory=False)
    lookup: dict[str, dict] = {}
    for _, row in df.iterrows():
        lookup[str(row["AGORA ID"])] = {
            "official_name": (row.get("Official name") or "Unknown document").strip(),
            "authority": (row.get("Authority") or "Unknown authority").strip(),
            "status": (row.get("Most recent activity") or "Unknown").strip(),
        }
    return lookup


def load_segments(data_dir: str) -> pd.DataFrame:
    """Load segments.csv, keeping only rows with non-empty Text."""
    path = os.path.join(data_dir, "segments.csv")
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, low_memory=False)
    df = df[df["Text"].notna() & (df["Text"].str.strip() != "")].reset_index(drop=True)
    return df


def build(data_dir: str, rebuild: bool = False) -> None:
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine distance for normalized text embeddings
    )

    if collection.count() > 0 and not rebuild:
        print(
            f"Collection '{COLLECTION_NAME}' already has {collection.count()} segments. "
            "Use --rebuild to recreate. Nothing to do."
        )
        return

    print(f"Loading data from {os.path.abspath(data_dir)} ...")
    docs = load_documents(data_dir)
    segments = load_segments(data_dir)
    print(f"Loaded {len(segments)} segments across {len(docs)} documents.")

    print(f"Embedding with {EMBED_MODEL_NAME} (CPU is fine for this size) ...")
    embedder = get_embedder()
    texts = segments["Text"].tolist()
    embeddings = embedder.encode(
        texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    )

    print("Writing to Chroma ...")
    ids, metadatas = [], []
    for _, row in segments.iterrows():
        doc_id = str(row["Document ID"])
        position = str(row["Segment position"])
        meta = docs.get(doc_id, {})
        ids.append(f"{doc_id}:{position}")
        metadatas.append(
            {
                "document_id": doc_id,
                "segment_position": position,
                "official_name": meta.get("official_name", "Unknown document"),
                "authority": meta.get("authority", "Unknown authority"),
                "status": meta.get("status", "Unknown"),
            }
        )

    for start in range(0, len(ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=metadatas[start:end],
        )

    print(f"Done. Vector store now holds {collection.count()} segments at {CHROMA_DIR}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the AGORA vector store.")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Path to the agora CSV folder.")
    parser.add_argument("--rebuild", action="store_true", help="Recreate the store from scratch.")
    args = parser.parse_args()
    build(args.data_dir, rebuild=args.rebuild)


if __name__ == "__main__":
    main()

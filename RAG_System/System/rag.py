"""Core RAG logic: retrieve grounded segments and generate a cited answer.

This module is the single source of truth for configuration (paths, model
names, the grounding prompt) so that ``ingest.py``, ``cli.py`` and ``eval.py``
all stay consistent. The embedding model and Chroma collection are loaded
lazily and cached, so importing this module is cheap.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))

# Override with AGORA_DATA_DIR env var if the dataset lives elsewhere.
DATA_DIR = os.environ.get("AGORA_DATA_DIR", os.path.join(_HERE, "..", "dataset", "agora"))

# Persistent vector store — must be built by ingest.py before querying.
CHROMA_DIR = os.path.join(_HERE, "chroma_db")
COLLECTION_NAME = "agora_segments"

# 384-dim, CPU-friendly. To swap models, change this and re-run ingest.py.
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Strong instruction-following on Groq's free tier.
GROQ_MODEL = "llama-3.3-70b-versatile"

# eval.py checks for this exact string to verify grounding behaviour.
REFUSAL = "I don't have enough information in the provided documents to answer that."

SYSTEM_PROMPT = f"""You are a careful assistant answering questions about AI governance \
documents (laws, regulations, policies).

Rules you must follow:
1. Answer ONLY using the numbered sources provided below. Never use outside knowledge.
2. If the sources do not contain the answer, reply with exactly this sentence and nothing else:
   "{REFUSAL}"
3. Cite every claim inline using the source number in square brackets, e.g. [Source 2].
4. Be concise and factual. Do not speculate or add information that is not in the sources.
"""


# ---------------------------------------------------------------------------
# Lazily-loaded singletons
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def get_embedder():
    """Load and cache the sentence-transformers embedding model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL_NAME)


@functools.lru_cache(maxsize=1)
def get_collection():
    """Return the persistent Chroma collection (must be built by ingest.py)."""
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Vector store '{COLLECTION_NAME}' not found in {CHROMA_DIR}. "
            "Run `python ingest.py` first."
        ) from exc


def embed_query(text: str):
    """Embed a single query string, returning a plain Python list."""
    return get_embedder().encode([text])[0].tolist()


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

@dataclass
class Source:
    """One retrieved segment plus the metadata needed to cite it."""

    text: str
    document_id: str
    official_name: str
    authority: str
    status: str
    segment_position: str
    distance: float

    def citation(self, n: int) -> str:
        """Render a one-line citation, e.g.
        `[1] REGULATION (EU) 2024/1689 ... — European Union — segment #1 (Enacted)`.
        """
        return (
            f"[{n}] {self.official_name} — {self.authority} "
            f"— segment #{self.segment_position} ({self.status})"
        )


def retrieve(query: str, k: int = 5) -> list[Source]:
    """Return the top-k most similar segments to ``query`` (cosine distance)."""
    collection = get_collection()
    result = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    sources: list[Source] = []
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]
    for text, meta, dist in zip(docs, metas, dists):
        sources.append(
            Source(
                text=text,
                document_id=str(meta.get("document_id", "")),
                official_name=meta.get("official_name", "Unknown document"),
                authority=meta.get("authority", "Unknown authority"),
                status=meta.get("status", "Unknown"),
                segment_position=str(meta.get("segment_position", "")),
                distance=float(dist),
            )
        )
    return sources


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _build_context(sources: list[Source]) -> str:
    """Format retrieved sources into the numbered block the model reads."""
    blocks = []
    for i, s in enumerate(sources, start=1):
        header = f"[Source {i}] {s.official_name} ({s.authority}), segment #{s.segment_position}"
        blocks.append(f"{header}\n{s.text}")
    return "\n\n".join(blocks)


def generate(query: str, sources: list[Source]) -> str:
    """Call Groq with the grounding prompt and the retrieved context."""
    from groq import Groq  # lazy import so ingest.py doesn't need groq/key

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "(get one free at https://console.groq.com/keys)."
        )

    client = Groq(api_key=api_key)
    user_message = (
        f"Sources:\n\n{_build_context(sources)}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the sources above, citing each claim as [Source N]."
    )
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,  # deterministic, grounded output
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content.strip()


def answer(query: str, k: int = 5) -> tuple[str, list[Source]]:
    """Full RAG step: retrieve -> generate. Returns (answer_text, sources)."""
    sources = retrieve(query, k=k)
    if not sources:
        return REFUSAL, []
    text = generate(query, sources)
    return text, sources

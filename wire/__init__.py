"""Wire Engine — a framework-free RAG engine, built from primitives.

Each module is one brick:
  embeddings  -> embed text to a vector, cosine similarity
  chunking    -> split posts into retrievable passages
  store       -> embed-once persistence (the Wire) + cosine search
  engine      -> grounded generation (answers only from retrieved context)
"""
from .embeddings import embed, cosine
from .chunking import chunk_all, chunk_text
from .store import build_or_load_wire, search
from .engine import answer

__all__ = [
    "embed", "cosine",
    "chunk_all", "chunk_text",
    "build_or_load_wire", "search",
    "answer",
]

"""Embeddings + similarity. The one place the embedding model lives."""
import os
import numpy as np
import voyageai

MODEL = "voyage-3"
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))
    return _client


def embed(text, input_type="document"):
    """Embed a single string -> np.float32 vector.

    input_type="document" when storing corpus chunks,
    input_type="query" when embedding a search query (voyage optimises each).
    """
    result = _get_client().embed([text], model=MODEL, input_type=input_type)
    return np.array(result.embeddings[0], dtype=np.float32)


def cosine(a, b):
    """Cosine similarity between two vectors. A dumb, local ruler."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denom == 0 else float(np.dot(a, b) / denom)

"""The Wire: a list of {title, url, text, vec} dicts. Embed once, cache to disk,
search by brute-force cosine. A vector DB is this + an index (HNSW) for scale."""
import os
import pickle
from .embeddings import embed, cosine


def build_or_load_wire(chunks, path):
    """Embed chunks once and cache. On re-run, load from disk (free).

    NOTE: if you change the source text, delete the cache file so it re-embeds.
    Stale cache is the trap: it will happily serve vectors for the old text.
    """
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    wire = [{**c, "vec": embed(c["text"], input_type="document")} for c in chunks]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(wire, f)
    return wire


def search(query, wire, top_k=4):
    """Return the top_k chunks by cosine, as [(score, chunk_dict), ...]."""
    qvec = embed(query, input_type="query")
    scored = [(cosine(qvec, item["vec"]), item) for item in wire]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

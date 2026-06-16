"""Grounded generation: hand the retrieved chunks to the LLM, fenced to answer
ONLY from them. Refuses honestly when the corpus doesn't cover the question."""
import os
from anthropic import Anthropic
from .store import search

MODEL = "claude-haiku-4-5-20251001"
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _client


SYSTEM = (
    "You answer ONLY from the provided context passages. "
    "If the answer is not in the context, reply exactly: 'Not in the Wire yet.' "
    "Cite the source title for each claim. Never use outside knowledge."
)


def answer(query, wire, top_k=4, model=MODEL):
    """Retrieve, then generate grounded in what was retrieved.

    Returns (text, hits) where hits is [(score, chunk_dict), ...].
    """
    hits = search(query, wire, top_k=top_k)
    context = "\n\n".join(
        f"[{i + 1}] {h['title']}\n{h['text']}" for i, (_, h) in enumerate(hits)
    )
    msg = _get_client().messages.create(
        model=model,
        max_tokens=500,
        system=SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        }],
    )
    return msg.content[0].text, hits

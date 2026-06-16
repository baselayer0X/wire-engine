"""Probe 2 — Does a cross-encoder reranker fix what naive search missed?

Naive vector search buried the correct source on a reworded query. The fix is a
cross-encoder: instead of placing query and chunk separately and measuring
distance (bi-encoder embeddings), it reads query and chunk TOGETHER and scores
their actual relationship.

This VALIDATES the guardrail on the exact query that failed. The before/after is
the point: you don't trust a guardrail, you measure it on the case it should catch.

RESULT (measured): naive search left the correct source out of the top 4 entirely;
after reranking it climbed to rank 1. Good embeddings are necessary; good ranking
is decisive.

Run:  python test_rerank.py      (needs: pip install sentence-transformers)
"""
from dotenv import load_dotenv
load_dotenv()

from wire import chunk_all, build_or_load_wire, search
from demo import DEMO_POSTS, WIRE_PATH
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

QUERY = "why did you lock up your own money on-chain?"


def show(label, ranked):
    print(f"\n{label}")
    for i, (score, c) in enumerate(ranked, 1):
        print(f"  {i}. {round(float(score), 3)}  {c['title']}")


def main():
    wire = build_or_load_wire(chunk_all(DEMO_POSTS), WIRE_PATH)

    # STAGE 1 — naive vector search, wide net (grab 10, not 4)
    candidates = search(QUERY, wire, top_k=10)
    show("STAGE 1 — naive vector search (what the engine did before):", candidates[:4])

    # STAGE 2 — rerank the shortlist: cross-encoder reads query+chunk together
    pairs = [(QUERY, c["text"]) for _, c in candidates]
    rerank_scores = reranker.predict(pairs)
    reranked = sorted(zip(rerank_scores, [c for _, c in candidates]),
                      key=lambda x: x[0], reverse=True)
    show("STAGE 2 — after cross-encoder rerank (the guardrail):", reranked[:4])

    print("\n--- did the guardrail work? ---")
    print("Did the correct source move UP into the top 4 after reranking?")
    print("Scale note: stage-1 scores are cosine (0-1); stage-2 are cross-encoder")
    print("logits (different scale). Read the ORDER and the GAPS, not raw numbers.")


if __name__ == "__main__":
    main()

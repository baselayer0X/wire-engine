"""Probe 3 — Does the engine know when NOT to answer?

A human running a search is a built-in skeptic: sees an off result, re-queries.
An unsupervised agent has none of that — it acts on whatever it got. So the engine
must judge its own confidence and abstain when it's low. The first primitive of
acting without a human in the loop.

METRIC (the honest story — keep it in the writeup): my first instinct was the GAP
between #1 and #2 reranked results. It misfired: with two genuinely strong
passages the gap is tiny, and the gate misread "two good answers" as "unsure." The
signal that worked is the TOP result's ABSOLUTE relevance score.

LIMIT: this catches "I'm lost," NOT "confidently wrong." And the threshold is a
risk decision, not a math fact.

Run:  python test_confidence.py
"""
from dotenv import load_dotenv
load_dotenv()

from wire import chunk_all, build_or_load_wire, search
from demo import DEMO_POSTS, WIRE_PATH
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def answer_with_confidence(query, wire, score_threshold=-2.0):
    candidates = search(query, wire, top_k=10)
    pairs = [(query, c["text"]) for _, c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, [c for _, c in candidates]),
                    key=lambda x: x[0], reverse=True)
    top = float(ranked[0][0])

    print(f"\nQ: {query}")
    print(f"   top reranker score: {round(top, 2)}")
    if top >= score_threshold:
        print(f"   CONFIDENT (>= {score_threshold}) -> act. Source: {ranked[0][1]['title']}")
    else:
        print(f"   LOW CONFIDENCE (< {score_threshold}) -> ABSTAIN. Don't act unsupervised.")


def main():
    wire = build_or_load_wire(chunk_all(DEMO_POSTS), WIRE_PATH)
    # in-corpus, clear answer -> should ACT
    answer_with_confidence("how do stablecoins pull deposits away from banks?", wire)
    # off-topic -> should ABSTAIN
    answer_with_confidence("what's the best running shoe for a marathon?", wire)
    # in-corpus topic, answer never written -> should ABSTAIN (corpus gap)
    answer_with_confidence("what did locking money on-chain teach about immutability?", wire)


if __name__ == "__main__":
    main()

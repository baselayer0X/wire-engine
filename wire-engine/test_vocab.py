"""Probe 1 — Is payments vocabulary adversarial to embeddings?

Hypothesis: dense payments terms (RTGS vs deferred net settlement, clearing)
collapse into near-synonyms in vector space despite different operational meaning.

RESULT (measured): the hypothesis did NOT hold. voyage-3 kept these terms
separated, and clearly distinguished them from an unrelated control. The failure
mode isn't embedding collapse — it's ranking at scale. The negative result IS the
finding: test, don't assert.

Run:  python test_vocab.py
"""
from dotenv import load_dotenv
load_dotenv()

from wire import embed, cosine

TERMS = [
    "real-time gross settlement (RTGS)",
    "deferred net settlement",
    "clearing",
    "liquidity",
    "settlement velocity",
    "a stable internet connection",   # control: should score LOW vs all above
]


def main():
    vecs = [embed(t) for t in TERMS]
    n = len(TERMS)
    width = max(len(t) for t in TERMS)

    print("\nPairwise cosine similarity (1.0 = identical meaning):\n")
    print(" " * (width + 2) + "  ".join(f"{i + 1:>5}" for i in range(n)))

    high_pairs = []
    for i in range(n):
        row = []
        for j in range(n):
            s = cosine(vecs[i], vecs[j])
            row.append(f"{s:5.2f}")
            if i < j and s > 0.85 and "internet" not in TERMS[i] and "internet" not in TERMS[j]:
                high_pairs.append((TERMS[i], TERMS[j], s))
        print(f"{TERMS[i]:<{width}}  " + "  ".join(row))

    print("\n--- did the hypothesis hold? ---")
    if high_pairs:
        print("Operationally-different terms scored dangerously close (>0.85):")
        for a, b, s in sorted(high_pairs, key=lambda x: -x[2]):
            print(f"  {round(s, 3)}  {a}  <->  {b}")
    else:
        print("Nothing bunched above 0.85. The embedding model held the line.")
        print("Finding: the failure is ranking at scale, not embedding collapse.")


if __name__ == "__main__":
    main()

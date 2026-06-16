"""End-to-end demo: build a Wire from a small corpus, ask a question, answer
grounded in the retrieved passages.

REPLACE DEMO_POSTS with your own writing (same {title, url, text} shape).
The text below is short illustrative sample data, not full essays — swap it out.
If you change the text later, delete data/demo_wire.pkl so it re-embeds.
"""
from dotenv import load_dotenv
load_dotenv()

from wire import chunk_all, build_or_load_wire, answer

WIRE_PATH = "data/demo_wire.pkl"

DEMO_POSTS = [
    {
        "title": "Stablecoins Aren't Eliminating Trade-Offs. They're Redistributing Them.",
        "url": "",
        "text": (
            "A stablecoin doesn't remove the trade-offs of moving money. It moves "
            "them somewhere less visible. When a dollar sits in a tokenised form, the "
            "question of who holds the reserves, on what terms, and how quickly they "
            "can be redeemed doesn't disappear. It shifts from a bank's balance sheet "
            "to an issuer's. If stablecoins pull deposits out of banks, the funding "
            "that banks used to lend against has to come from somewhere else, or it "
            "doesn't come at all. The convenience is real. The trade-off didn't vanish; "
            "it changed address."
        ),
    },
    {
        "title": "BIS Project Agorá and the Plumbing of Cross-Border Settlement",
        "url": "",
        "text": (
            "Project Agorá tries to solve cross-border settlement by putting tokenised "
            "commercial bank deposits and central bank money on a shared ledger. The "
            "hard part was never the messaging — it was the deferral of FX and the "
            "sequencing of who gets paid when. That's where the plumbing meets the real "
            "economy. A unified ledger changes the choreography of settlement, but it "
            "inherits every legal and liquidity question the correspondent banking model "
            "spent decades papering over."
        ),
    },
    {
        "title": "I Locked $5 on Ethereum Until 2030. Here's What It Cost and What I Learned.",
        "url": "",
        "text": (
            "I deposited 5 USDC into a time-locked smart contract on Ethereum mainnet, "
            "set to stay frozen until March 2030. Not even I can withdraw it before then. "
            "The total cost to deploy, approve, and deposit was thirty-six cents. The "
            "lesson wasn't about the money. It was about immutability: once the contract "
            "is deployed, there is no admin key, no override, no support ticket. A wrong "
            "address or a bad timestamp can't be patched. On-chain, the spec is the "
            "product, and finality is not a setting you can change later."
        ),
    },
    {
        "title": "The Payments Assumptions We Never Question",
        "url": "",
        "text": (
            "Settlement is T+2 because someone decided it should be, back when batch "
            "processing was the binding constraint. The constraint is gone; the policy "
            "remains, because business models depend on it. Most of what we call 'how "
            "payments work' is a stack of defaults that were once technical limits and "
            "are now just habits. Asking which is which is the most useful thing a "
            "payments product person can do."
        ),
    },
]


def main():
    wire = build_or_load_wire(chunk_all(DEMO_POSTS), WIRE_PATH)
    q = "do stablecoins threaten bank deposits?"
    response, hits = answer(q, wire, top_k=4)

    print(f"Q: {q}\n")
    print(response)
    print("\nSources retrieved:")
    for score, h in hits:
        print(f"  {round(score, 3)}  {h['title']}")


if __name__ == "__main__":
    main()

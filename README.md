# Wire Engine

**A framework-free RAG engine, built from primitives to find where vector search breaks and what that means before an AI agent acts on what it retrieves, with no human in the loop.**

A small reference experiment, not a production system. The point isn't the engine; it's the mechanic the engine makes visible. **v0.2.**

---

## Why this exists

AI agents are moving into financial workflows, researching, scoring risk, eventually touching settlement. The conversation is almost entirely about reasoning. I went one layer beneath that and came out convinced the harder question is upstream.

A human running a search is a built-in skeptic: when retrieval surfaces something off, they notice, reframe, and re-query. An agent has none of that. It acts on whatever got fetched. Removing the human doesn't make retrieval better, it removes the thing that was quietly catching retrieval's mistakes. In finance, where a miss can move money and can't be clawed back, that gap is the risk.

So the thesis this repo backs into: **as agents enter financial systems, the ability to abstain may matter as much as the ability to act.**

---

## What it does

Point it at a body of writing. It splits each piece into passages, embeds them into a store (the **Wire**), retrieves the closest in meaning, reranks them, judges its own confidence, and either answers grounded in the sources, after citing them, or abstains with *"Not in the Wire yet."*

---

## What the probes actually showed

Three probes, three measured results. Two of them refuted something I believed going in, which is the point of testing instead of asserting.

**1. Embeddings are better than I assumed (`test_vocab.py`).** Hypothesis: dense payments terms (RTGS vs deferred net settlement, clearing) would collapse into near-synonyms in vector space. *They didn't.* voyage-3 kept operationally-different terms separated, and clearly distinguished them from an unrelated control. The failure isn't embedding collapse; it's ranking. As a corpus scales, the embeddings stay accurate; the problem becomes too many plausibly-relevant candidates competing for too few retrieval slots.

**2. Reranking is decisive (`test_rerank.py`).** On a query where naive vector search buried the correct source entirely (dropping it out of the top 4), a cross-encoder reranker pulled it to rank 1. The cross-encoder reads query and passage *together*, instead of placing them separately and measuring distance. The mechanism that failed was fixed by the mechanism that reads jointly. *Good embeddings are necessary; good ranking is decisive.*

**3. The engine can know when not to answer (`test_confidence.py`).** A confidence gate reads the top reranked result's absolute relevance score; below a threshold, it abstains instead of acting. It correctly acts on in-corpus questions and refuses both off-topic queries and questions whose answer was never written. **Honest limit:** this catches "I'm lost," not "confidently wrong," and the threshold is a risk decision, not a math fact.

---

## Field notes: where it breaks

- **The two models fail in opposite ways.** The generation model was faithful every time, accurate synthesis or an honest refusal. Every failure lived in retrieval, upstream of it.
- **Retrieval matches phrasing, not topic.** The same question, reworded, surfaced or buried the right passage, because the embedding model places query and chunk separately.
- **You can't retrieve what you never wrote.** The confidence gate flagged a question whose answer simply wasn't in my corpus. Retrieval gaps and corpus gaps look identical from outside.
- **Score-bunching is noise, not ranking.** Cosine scores clustered within 0.01, the model saying "equally relevant," not "this one wins." Reranking un-bunched them.
- **Retrieval is the product.** Across every failure, the breakage was in what got fetched and how it ranked — never in the model that talks.

---

## The real problem: validating the guardrails unsupervised

Reranking and confidence-gating help. They don't *close* it. When you replace the human with software guardrails, you inherit an obligation you didn't have before: proving those guardrails catch what the human used to catch, on your data, before anything acts. Output metrics aren't enough; auditing retrieval decisions in regulated domains has no standard yet. That validation gap — not retrieval itself — is the live problem.

---

## Probes

- `test_vocab.py` — tests whether payments vocabulary collapses in embedding space (it didn't).
- `test_rerank.py` — measures whether a cross-encoder rescues the passage naive search buried (it did).
- `test_confidence.py` — abstains when the top result doesn't clear a relevance bar.

## Deliberate next steps

- [x] Vocabulary adversarial probe
- [x] Cross-encoder reranking
- [x] Confidence gating / abstention
- [ ] **Separation of reasoning and authorization** — so a confident retrieval miss can't reach the execution path unchecked.
- [ ] **A real evaluation harness** — measuring whether retrieval is trustworthy enough to run unsupervised. The actual frontier.

---

## Quickstart

```bash
git clone https://github.com/baselayer0X/wire-engine.git
cd wire-engine
pip install -r requirements.txt
pip install sentence-transformers   # for the reranker + confidence probes
cp .env.example .env                 # add your keys
python demo.py
```

Needs a [Voyage AI](https://www.voyageai.com) key and an [Anthropic](https://console.anthropic.com) key. Small free tiers; a full run costs a fraction of a cent. The cross-encoder runs locally (downloads once, ~80MB).

---

## Stack

- **Embeddings:** Voyage (`voyage-3`)
- **Generation:** Anthropic Claude
- **Reranking:** sentence-transformers cross-encoder (local)
- **Math + storage:** NumPy + pickle
- **Language:** Python, no framework — primitives on purpose

---

*A reference experiment in retrieval and abstention, built to understand payments and AI infrastructure from the inside, not the spec. Not investment advice. MIT licensed.*

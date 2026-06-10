# AGORA RAG — Grounded Q&A over AI-governance documents

A small, CLI-based Retrieval-Augmented Generation (RAG) system that answers
natural-language questions about AI governance using the [AGORA](https://www.kaggle.com/datasets/umerhaddii/ai-governance-documents-data)
corpus (laws, regulations, executive orders, and policies from governments and
companies worldwide). Every answer is grounded strictly in retrieved text and
returns citations showing the **document name, issuing authority, and segment
(paragraph) position**.

## How it works

```
                  ingest.py (run once)                     query time
  ┌───────────────┐   embed    ┌──────────────┐      ┌──────────────────────┐
  │ segments.csv  │ ─────────▶ │  ChromaDB    │ ◀──  │ cli.py / eval.py     │
  │ documents.csv │  MiniLM    │ (persistent) │      │  1. embed question   │
  │ (metadata)    │            │  vectors +   │ ──▶  │  2. top-k cosine      │
  └───────────────┘            │  citation    │      │  3. grounded prompt   │
                               │  metadata    │      │  4. Groq LLM → answer │
                               └──────────────┘      │     + citations       │
                                                     └──────────────────────┘
```

1. **Ingest** (`ingest.py`): load the ~5,400 pre-chunked segments, embed each
   segment's `Text` with `all-MiniLM-L6-v2`, and store the vectors in a
   persistent ChromaDB collection. The document-level metadata needed for
   citations (official name, authority, status) is joined in at this stage and
   stored next to each vector.
2. **Retrieve** (`rag.py`): embed the user's question with the same model and
   fetch the top-k most similar segments by cosine distance.
3. **Generate** (`rag.py`): send the retrieved segments to a Groq-hosted LLM
   under a strict system prompt that forbids outside knowledge, requires inline
   `[Source N]` citations, and mandates a fixed refusal sentence when the answer
   is not in the context.
4. **Cite** (`cli.py`): print the answer followed by a numbered source list.

## Key decisions and tradeoffs

| Decision | Choice | Why |
| --- | --- | --- |
| Chunking | Use the dataset's existing segments as-is | AGORA is already split into coherent paragraph-level segments with stable positions, which double as citation anchors. Re-chunking would add complexity and break those references. |
| What to embed | Segment `Text` (not `Summary`) | `Text` is the operative language users ask about; `Summary` is a lossy, sometimes machine-generated abstraction. Embedding `Text` keeps retrieval faithful to the source. |
| Embedding model | `all-MiniLM-L6-v2` | 384-dim, fast on CPU, strong quality-for-size. The corpus is small (~5.4k segments), so a heavier model isn't justified for a 6–8h build. |
| Vector store | ChromaDB (persistent) | Zero-config local persistence, cosine similarity, metadata stored alongside vectors — no separate database to run. |
| Retrieval | Simple top-k cosine | Transparent and easy to reason about. With a corpus this size it performs well; see *Limitations* for when this breaks down. |
| Citation join | At ingest time | Each vector carries its document name/authority/status, so the query path needs no runtime join back to `documents.csv`. |
| LLM | Groq `llama-3.3-70b-versatile`, `temperature=0` | Free, fast, strong instruction-following (critical for grounding). Temperature 0 keeps answers deterministic and faithful. |
| Grounding | System prompt + fixed refusal string | The model must answer only from sources and emit an exact refusal sentence otherwise. That exact string is what `eval.py` checks. |

## Setup

```bash
# 1. From the System/ directory, install dependencies
pip install -r requirements.txt

# 2. Provide a Groq API key (free: https://console.groq.com/keys)
cp .env.example .env        # then edit .env and paste your key

# 3. Get the dataset (NOT committed — see below) and place it so that
#    ../dataset/agora/ contains segments.csv, documents.csv, etc.
#    Override the location with AGORA_DATA_DIR if your path differs.
```

### Dataset

The AGORA corpus is **not included** in this repo (per the assessment's "do not
commit large datasets"). Download it from
[Kaggle](https://www.kaggle.com/datasets/umerhaddii/ai-governance-documents-data)
and unzip so the `agora/` folder (containing `segments.csv`, `documents.csv`,
`authorities.csv`, `collections.csv`, and `fulltext/`) sits at `../dataset/agora`
relative to this `System/` directory, or point `AGORA_DATA_DIR` at it.

## How to run

```bash
# Build the vector store once (~seconds for the corpus; first run also
# downloads the embedding model)
python ingest.py
python ingest.py --rebuild        # force a clean rebuild

# Ask questions interactively
python cli.py

# Or ask a single question
python cli.py -q "What does the EU AI Act establish?"

# Evaluate retrieval + grounding
python eval.py
python eval.py --skip-generation  # retrieval-only (no API key needed)
```

## Example usage

**Question 1 — cross-jurisdiction governance question**
```
$ python cli.py -q "What does the EU AI Act establish?"

The EU AI Act, also known as Regulation (EU) 2024/1689, lays down harmonised
rules on artificial intelligence [Source 2]. It amends several Regulations and
Directives, including Regulations (EC) No 300/2008, (EU) No 167/2013,
(EU) No 168/2013, (EU) 2018/858, (EU) 2018/1139, and (EU) 2019/2144, and
Directives 2014/90/EU, (EU) 2016/797, and (EU) 2020/1828 [Source 3]. The Act
establishes rules for the development, use, and uptake of artificial
intelligence in the internal market, aiming to foster the development of AI
while ensuring a high level of protection of public interests and fundamental
rights [Source 4]. It also requires statements of conformity with the
Regulation and other relevant Union laws, as well as compliance with
Regulations (EU) 2016/679 and (EU) 2018/1725 and Directive (EU) 2016/680
when AI systems involve the processing of personal data [Source 2].

Sources:
  [1] National AI Commission Act — United States Congress — segment #3 (Proposed)
  [2] REGULATION (EU) 2024/1689 ... (Artificial Intelligence Act) — European Union — segment #234 (Enacted)
  [3] REGULATION (EU) 2024/1689 ... (Artificial Intelligence Act) — European Union — segment #216 (Enacted)
  [4] Commission Decision Establishing the European AI Office — European Union — segment #1 (Enacted)
  [5] Federal AI Governance and Transparency Act — United States Congress — segment #2 (Proposed)
```

**Question 2 — framework/standards question**
```
$ python cli.py -q "What characteristics make an AI system trustworthy according to NIST?"

According to NIST, the characteristics of trustworthy AI systems include:
valid and reliable, safe, secure and resilient, accountable and transparent,
explainable and interpretable, privacy-enhanced, and fair with harmful bias
managed [Source 2].

Sources:
  [1] National Artificial Intelligence Research and Development Strategic Plan 2023 Update — Executive Office of the President — segment #26 (Enacted)
  [2] NIST AI Risk Management Framework 1.0 — National Institute of Standards and Technology — segment #1 (Enacted)
  [3] NIST AI Risk Management Framework 1.0 — National Institute of Standards and Technology — segment #4 (Enacted)
  [4] Governance Principles for a New Generation of Artificial Intelligence — Chinese central government — segment #5 (Enacted)
  [5] Memorandum on Advancing the United States' Leadership in Artificial Intelligence — Executive Office of the President — segment #12 (Enacted)
```

**Question 3 — grounding check (answer not in corpus)**
```
$ python cli.py -q "What is the capital of France?"

I don't have enough information in the provided documents to answer that.

Sources:
  [1] REGULATION (EU) 2024/1689 ... (Artificial Intelligence Act) — European Union — segment #246 (Enacted)
  [2] Council of Europe Framework Convention on Artificial Intelligence — Other multinational — segment #1 (Enacted)
  ...
```

*The sources list shows what was retrieved — the model correctly determined none
of it answers the question and refused rather than guessing.*

**Evaluation results**
```
$ python eval.py

=== Retrieval quality (hit@5) ===
  [PASS] expected doc  757 | What does the EU AI Act (Regulation 2024/1689) establish?
  [PASS] expected doc  772 | What characteristics make an AI system trustworthy according...
  [PASS] expected doc 1163 | What is California's Safe and Secure Innovation for Frontier...
  [PASS] expected doc  307 | What does the US Executive Order on Safe, Secure, and Trustw...
  [PASS] expected doc 1377 | What is the legislative intent of China's draft Artificial I...
  [PASS] expected doc    1 | What does the law require regarding a digital development in...
  -> hit@5: 6/6 = 100%

=== Grounding (refusal on out-of-corpus questions) ===
  [PASS] What is the capital of France?
  [PASS] What was Apple's revenue in the third quarter of 2023?
  -> refused: 2/2 = 100%
```

## Evaluating answer quality

`eval.py` checks the two distinct ways the system can fail:

- **Retrieval quality (`hit@k`)** — for a gold set of questions with a known
  source document, does that document appear in the top-k retrieved segments?
  This isolates retrieval and needs no LLM call.
- **Grounding** — for questions whose answers are *not* in the corpus, does the
  model emit the exact refusal sentence instead of hallucinating?

## Limitations and known issues

- **No metadata pre-filtering.** The 77 binary taxonomy columns in
  `documents.csv` (Applications, Harms, Risk factors, …) and the authority
  hierarchy are not yet used to pre-filter retrieval. For broad or
  jurisdiction-specific queries this would improve precision.
- **Pure dense retrieval.** Semantic-only search can miss exact-keyword or
  acronym matches (e.g. a specific article number). A hybrid BM25 + dense
  approach with a reranker would be more robust.
- **Faithfulness is not auto-graded.** `eval.py` verifies that the *right
  documents are retrieved* and that out-of-corpus questions are refused, but it
  does not yet score whether every sentence of an answer is supported by its
  cited source (e.g. via an LLM-as-judge faithfulness check).
- **Segment-level context only.** Long documents (the EU AI Act spans 254
  segments) are retrieved as isolated paragraphs; answers spanning many
  segments may miss context. Parent-document expansion would help.
- **Top-k is fixed.** No adaptive cutoff by similarity score; a low-relevance
  query still returns k segments (though the grounding prompt mitigates this).

## What I'd do next

1. Add optional metadata filters (authority/jurisdiction, taxonomy tags) parsed
   from the query, as an opt-in pre-filter before semantic search.
2. Add hybrid retrieval (BM25 + dense) and a cross-encoder reranker.
3. Add an LLM-as-judge faithfulness metric to `eval.py` and expand the gold set.
4. Parent-document / neighbouring-segment expansion for long laws.

## Project layout

```
System/
  ingest.py            # build the vector store from the AGORA CSVs
  rag.py               # config, retrieval, and grounded generation (shared core)
  cli.py               # interactive + single-shot CLI
  eval.py              # retrieval hit@k + grounding/refusal checks
  eval_questions.json  # gold questions (answerable + unanswerable)
  requirements.txt
  .env.example
  README.md
```

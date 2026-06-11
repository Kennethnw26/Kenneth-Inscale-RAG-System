# AGORA RAG - Grounded Q&A over AI Governance Documents

A CLI-based Retrieval-Augmented Generation (RAG) system that answers questions about AI governance documents. The knowledge base is the AGORA corpus which contains laws, regulations, executive orders, and policies from governments and companies around the world. Every answer is grounded strictly in the retrieved text and comes with citations showing the document name, issuing authority, and the exact paragraph it came from.

## How it works

```
                  ingest.py (run once)                     query time
  +--------------+   embed    +--------------+      +----------------------+
  | segments.csv | ---------> |  ChromaDB    | <--  | cli.py / eval.py     |
  | documents.csv|  MiniLM    | (persistent) |      |  1. embed question   |
  | (metadata)   |            |  vectors +   | -->  |  2. top-k cosine     |
  +--------------+            |  citation    |      |  3. grounded prompt  |
                              |  metadata    |      |  4. Groq LLM -> answer|
                              +--------------+      |     + citations       |
                                                    +----------------------+
```

1. **Ingest** (`ingest.py`): loads the pre-chunked segments, embeds each segment's text using `all-MiniLM-L6-v2`, and stores the vectors in a persistent ChromaDB collection. Document-level metadata needed for citations is joined in at this stage so the query path never needs to look it up again.
2. **Retrieve** (`rag.py`): embeds the user's question using the same model and fetches the top-k most similar segments by cosine distance.
3. **Generate** (`rag.py`): sends the retrieved segments to a Groq-hosted LLM under a strict system prompt that forbids outside knowledge, requires inline `[Source N]` citations, and returns a fixed refusal message when the answer is not in the context.
4. **Cite** (`cli.py`): prints the answer followed by a numbered source list.

## Assumptions

These are the decisions I made upfront that shaped how the system works.

- **Segments are used as-is.** The dataset is already split into paragraph-level chunks with a stable `Segment position` identifier per row. I used these directly as the retrieval unit and as the citation anchor instead of re-chunking, since re-chunking would add complexity without clear benefit given the data is already well-structured.

- **I embed the `Text` field, not the `Summary` field.** The text is the actual operative content that users ask about. The summary is a lossy abstraction and the dataset itself notes that summaries may include unreviewed machine output, so I did not trust it as a retrieval target.

- **The system only retrieves from `segments.csv`, not the raw `fulltext/` files.** The pre-chunked segments cover the same content and already have structured metadata attached. Using the raw text files would require re-chunking and re-building the metadata join.

- **Out-of-context questions get a fixed refusal.** If the retrieved segments do not contain enough information to answer the question, the system returns a specific refusal message instead of guessing. This is enforced through the system prompt and verified in `eval.py`.

- **Each query is independent.** There is no conversation history between questions. The system is designed for single-turn factual lookups, not back-and-forth dialogue.

- **Citations show document name, authority, and segment position.** This is enough for someone to trace an answer back to the exact paragraph in the original source.

- **Evaluation covers retrieval quality and grounding, but not faithfulness.** `eval.py` checks whether the right documents are retrieved and whether out-of-corpus questions are refused. It does not yet score whether every sentence in an answer is actually supported by the cited source. That would require an LLM-as-judge setup which is outside the scope of this project for now.

## Key decisions and tradeoffs

| Decision | Choice | Why |
| --- | --- | --- |
| Chunking | Use existing segments as-is | Already split into coherent paragraph-level chunks with stable position identifiers that double as citation anchors |
| What to embed | Segment `Text` only | Text is the operative content; Summary is a lossy abstraction that may include unreviewed machine output |
| Embedding model | `all-MiniLM-L6-v2` | 384-dim, fast on CPU, good quality for its size. The corpus is small at around 5,400 segments so a larger model is not needed |
| Vector store | ChromaDB (persistent) | Zero-config local persistence, cosine similarity, metadata stored alongside vectors with no separate database to run |
| Retrieval | Simple top-k cosine | Transparent and easy to reason about. Works well at this corpus size |
| Citation join | At ingest time | Each vector carries its document name, authority, and status so the query path needs no runtime join back to documents.csv |
| LLM | Groq `llama-3.3-70b-versatile`, `temperature=0` | Free, fast, and follows instructions reliably which is important for grounding. Temperature 0 keeps answers deterministic |
| Grounding | System prompt + fixed refusal string | The model must answer only from retrieved sources and emit an exact refusal sentence otherwise. That exact string is what `eval.py` checks against |

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add a Groq API key (free at https://console.groq.com/keys)
cp .env.example .env
# then open .env and paste your key

# 3. Place the dataset so that ../dataset/agora/ contains
#    segments.csv, documents.csv, authorities.csv, collections.csv, and fulltext/
#    You can override the path with the AGORA_DATA_DIR environment variable
```

### Dataset

The AGORA corpus is not included in this repo. Download it from [Kaggle](https://www.kaggle.com/datasets/umerhaddii/ai-governance-documents-data) and unzip it so the `agora/` folder sits at `../dataset/agora` relative to the `System/` directory.

## How to run

```bash
# Build the vector store (first run also downloads the embedding model)
python ingest.py
python ingest.py --rebuild        # force a clean rebuild

# Ask questions interactively
python cli.py

# Ask a single question
python cli.py -q "What does the EU AI Act establish?"

# Run the evaluation
python eval.py
python eval.py --skip-generation  # retrieval check only, no API key needed
```

## Example usage

**Question 1 - cross-jurisdiction governance question**
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
rights [Source 4].

Sources:
  [1] National AI Commission Act - United States Congress - segment #3 (Proposed)
  [2] REGULATION (EU) 2024/1689 ... (Artificial Intelligence Act) - European Union - segment #234 (Enacted)
  [3] REGULATION (EU) 2024/1689 ... (Artificial Intelligence Act) - European Union - segment #216 (Enacted)
  [4] Commission Decision Establishing the European AI Office - European Union - segment #1 (Enacted)
  [5] Federal AI Governance and Transparency Act - United States Congress - segment #2 (Proposed)
```

**Question 2 - framework/standards question**
```
$ python cli.py -q "What characteristics make an AI system trustworthy according to NIST?"

According to NIST, the characteristics of trustworthy AI systems include:
valid and reliable, safe, secure and resilient, accountable and transparent,
explainable and interpretable, privacy-enhanced, and fair with harmful bias
managed [Source 2].

Sources:
  [1] National Artificial Intelligence R&D Strategic Plan 2023 Update - Executive Office of the President - segment #26 (Enacted)
  [2] NIST AI Risk Management Framework 1.0 - National Institute of Standards and Technology - segment #1 (Enacted)
  [3] NIST AI Risk Management Framework 1.0 - National Institute of Standards and Technology - segment #4 (Enacted)
  [4] Governance Principles for a New Generation of Artificial Intelligence - Chinese central government - segment #5 (Enacted)
  [5] Memorandum on Advancing the United States' Leadership in Artificial Intelligence - Executive Office of the President - segment #12 (Enacted)
```

**Question 3 - grounding check (answer not in corpus)**
```
$ python cli.py -q "What specific penalties does the UK's AI Liability Framework impose on companies that deploy discriminatory hiring algorithms?"

I don't have enough information in the provided documents to answer that.
```

The system retrieves documents but correctly determines none of them answer the question and refuses rather than guessing.

## Evaluating answer quality

`eval.py` checks the two main ways the system can fail.

- **Retrieval quality (hit@k):** for a set of questions with a known source document, does that document appear in the top-k retrieved segments? This isolates retrieval from generation and needs no LLM call.
- **Grounding:** for questions whose answers are not in the corpus, does the model return the refusal message instead of hallucinating?

```
=== Retrieval quality (hit@5) ===
  [PASS] expected doc  757 | What does the EU AI Act (Regulation 2024/1689) establish?
  [PASS] expected doc  772 | What characteristics make an AI system trustworthy according...
  [PASS] expected doc 1163 | What is California's Safe and Secure Innovation for Frontier...
  [PASS] expected doc  307 | What does the US Executive Order on Safe, Secure, and Trustw...
  [PASS] expected doc 1377 | What is the legislative intent of China's draft Artificial I...
  [PASS] expected doc    1 | What does the law require regarding a digital development in...
  -> hit@5: 6/6 = 100%

=== Grounding (refusal on out-of-corpus questions) ===
  [PASS] What specific penalties does the UK's AI Liability Framework impose on companies that deploy discriminatory hiring algorithms?
  [PASS] What was Apple's revenue in the third quarter of 2023?
  -> refused: 2/2 = 100%
```

## Limitations

- **Top-k is fixed.** There is no adaptive cutoff based on similarity score, so a low-relevance query still returns k segments. The grounding prompt reduces the risk of the LLM using them but it is still not ideal.
- **Segment-level context only.** Long documents like the EU AI Act span 254 segments. Answers that need information spread across multiple segments can feel incomplete since each segment is retrieved in isolation.
- **Faithfulness is not measured.** `eval.py` checks that the right documents are retrieved and that out-of-corpus questions are refused, but it does not check whether every sentence in the answer is actually supported by the source it cites.

## What I would do next

1. Try different values of k and see how it affects answer quality for different types of questions.
2. Look into whether scoring answers against their cited sources (faithfulness) is feasible to add to the evaluation.
3. Expand the gold question set in `eval.py` to cover more document types and edge cases.

## Project layout

```
System/
  ingest.py            # builds the vector store from the AGORA CSVs
  rag.py               # config, retrieval, and grounded generation
  cli.py               # interactive and single-shot CLI
  eval.py              # retrieval hit@k and grounding checks
  eval_questions.json  # gold questions (answerable and unanswerable)
  requirements.txt
  .env.example
```

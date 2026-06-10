"""Lightweight evaluation harness — the answer-quality check.

Two things we measure, mapping to the two ways this system can fail:

1. Retrieval quality (hit@k): for each gold question, does the expected source
   document appear in the top-k retrieved segments? This isolates retrieval from
   generation (no LLM call, fast, deterministic).

2. Grounding / refusal: for questions whose answer is NOT in the corpus, does the
   model correctly refuse instead of hallucinating? This needs a Groq call, so it
   is skipped automatically if GROQ_API_KEY is unset (use --skip-generation to
   force retrieval-only).

Run:
    python eval.py
    python eval.py --k 5
    python eval.py --skip-generation
"""

from __future__ import annotations

import argparse
import json
import os

from dotenv import load_dotenv

from rag import REFUSAL, answer, retrieve

# Load .env from this file's folder regardless of the IDE working directory.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

QUESTIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_questions.json")


def evaluate_retrieval(answerable: list[dict], k: int) -> float:
    """Print a hit@k table for the answerable questions and return the hit rate."""
    print(f"\n=== Retrieval quality (hit@{k}) ===")
    hits = 0
    for item in answerable:
        sources = retrieve(item["question"], k=k)
        retrieved_ids = {s.document_id for s in sources}
        hit = item["expected_doc_id"] in retrieved_ids
        hits += hit
        mark = "PASS" if hit else "FAIL"
        print(f"  [{mark}] expected doc {item['expected_doc_id']:>4} | {item['question'][:60]}")
    rate = hits / len(answerable) if answerable else 0.0
    print(f"  -> hit@{k}: {hits}/{len(answerable)} = {rate:.0%}")
    return rate


def evaluate_grounding(unanswerable: list[dict], k: int) -> float:
    """Check that out-of-corpus questions trigger the refusal. Returns pass rate."""
    print("\n=== Grounding (refusal on out-of-corpus questions) ===")
    passed = 0
    for item in unanswerable:
        answer_text, _ = answer(item["question"], k=k)
        refused = REFUSAL.lower() in answer_text.lower()
        passed += refused
        mark = "PASS" if refused else "FAIL"
        print(f"  [{mark}] {item['question'][:60]}")
        if not refused:
            print(f"         got: {answer_text[:80]}")
    rate = passed / len(unanswerable) if unanswerable else 0.0
    print(f"  -> refused: {passed}/{len(unanswerable)} = {rate:.0%}")
    return rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and grounding.")
    parser.add_argument("--k", type=int, default=5, help="top-k for retrieval (default 5).")
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Only run the retrieval check (no Groq calls).",
    )
    args = parser.parse_args()

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        gold = json.load(f)

    evaluate_retrieval(gold["answerable"], args.k)

    if args.skip_generation or not os.environ.get("GROQ_API_KEY"):
        print("\n(Skipping grounding check — set GROQ_API_KEY to run it.)")
    else:
        evaluate_grounding(gold["unanswerable"], args.k)


if __name__ == "__main__":
    main()

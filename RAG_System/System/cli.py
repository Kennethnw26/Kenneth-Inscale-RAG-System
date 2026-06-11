"""Command-line interface for the AGORA RAG system.

Interactive:
    python cli.py

Single question (handy for scripting / reproducible README examples):
    python cli.py --question "What does the EU AI Act establish?" --k 5
"""

from __future__ import annotations

import argparse

import os

from dotenv import load_dotenv

from rag import Source, answer

# Ensures .env is found regardless of IDE working directory.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))


def render(answer_text: str, sources: list[Source]) -> str:
    """Format an answer plus its citation list for the terminal."""
    lines = ["", answer_text, ""]
    if sources:
        lines.append("Sources:")
        for i, s in enumerate(sources, start=1):
            lines.append("  " + s.citation(i))
    return "\n".join(lines)


def ask(question: str, k: int) -> None:
    try:
        answer_text, sources = answer(question, k=k)
    except RuntimeError as exc:
        print(f"\nError: {exc}")
        return
    print(render(answer_text, sources))


def interactive(k: int) -> None:
    print("AGORA AI-governance Q&A. Type a question, or 'exit' to quit.\n")
    while True:
        try:
            question = input("Ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        ask(question, k)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask grounded questions over the AGORA corpus.")
    parser.add_argument("--question", "-q", help="Ask a single question and exit.")
    parser.add_argument("--k", type=int, default=5, help="Number of segments to retrieve (default 5).")
    args = parser.parse_args()

    if args.question:
        ask(args.question, args.k)
    else:
        interactive(args.k)


if __name__ == "__main__":
    main()

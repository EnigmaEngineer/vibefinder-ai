"""Command-line front end for the VibeFinder 2.0 agent.

    python -m src.assistant "lofi beats for studying"
    python -m src.assistant "music for a rainy day" --show-trace
    python -m src.assistant "music for a rainy day" --no-rag      # baseline for comparison
    python -m src.assistant --interactive                         # chat loop with feedback
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from tabulate import tabulate

from src.agent import ROOT, VibeAgent, save_trace, trace_to_markdown
from src.explainer import get_explainer
from src.main import HEADERS, build_rows
from src.recommender import STRATEGIES

LOG_DIR = ROOT / "logs"


def setup_logging(verbose: bool) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    handlers = [logging.FileHandler(LOG_DIR / "assistant.log", encoding="utf-8")]
    if verbose:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def render(result, show_trace: bool = False) -> str:
    out = [f"Request: {result.query}", f"Status:  {result.status}"]
    if result.status != "ok":
        out.append(f"Message: {result.message}")
        for note in result.notes:
            out.append(f"Note:    {note}")
    else:
        out.append(f"Sources: {', '.join(result.sources) or 'none (explicit cues only)'}")
        out.append(f"Mode:    {result.mode or 'default'} | attempts: {result.attempts} | "
                   f"confidence: {result.confidence:.2f} ({result.confidence_label})"
                   + ("  ** flagged for human review **" if result.needs_review else ""))
        out.append(tabulate(build_rows(result.recommendations), headers=HEADERS, tablefmt="grid",
                            floatfmt=".2f", colalign=("right", "left", "left", "right", "left")))
        out.append(f"Answer ({result.explainer}):")
        out.append(result.explanation)
        out.append(f"Output guardrail: {'passed' if result.guardrail.get('ok') else result.guardrail}")
    if show_trace:
        out.append("")
        out.append(trace_to_markdown(result))
    return "\n".join(out)


def ask_feedback(result) -> None:
    """Human-in-the-loop: a 1-5 rating is appended to logs/feedback.jsonl."""
    if result.status != "ok":
        return
    answer = input("Rate these picks 1-5 (Enter to skip): ").strip()
    if answer in {"1", "2", "3", "4", "5"}:
        comment = input("Anything wrong with them? (optional): ").strip()
        LOG_DIR.mkdir(exist_ok=True)
        with open(LOG_DIR / "feedback.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"query": result.query, "rating": int(answer), "comment": comment,
                                "titles": result.titles(), "confidence": result.confidence}) + "\n")
        print("Thanks, saved to logs/feedback.jsonl")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="VibeFinder 2.0: describe what you want to hear.")
    parser.add_argument("query", nargs="?", help="what you want to listen to")
    parser.add_argument("--no-rag", action="store_true", help="skip retrieval (baseline behaviour)")
    parser.add_argument("--no-llm", action="store_true", help="skip Gemini even if a key is set (template only)")
    parser.add_argument("--mode", choices=sorted(STRATEGIES), help="force a scoring mode")
    parser.add_argument("-k", type=int, default=5, help="number of songs (1-10)")
    parser.add_argument("--show-trace", action="store_true", help="print the agent's step-by-step trace")
    parser.add_argument("--save-trace", action="store_true", help="write the trace to logs/traces/*.json")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    parser.add_argument("--interactive", action="store_true", help="keep asking; rate each answer")
    parser.add_argument("--verbose", action="store_true", help="also print log lines to stderr")
    args = parser.parse_args(argv)
    if not 1 <= args.k <= 10:
        parser.error("-k must be between 1 and 10")
    if not args.query and not args.interactive:
        parser.error("give a query or use --interactive")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    agent = VibeAgent(use_rag=not args.no_rag, k=args.k, explainer=get_explainer(prefer_llm=not args.no_llm))

    def handle(query: str):
        result = agent.run(query, mode=args.mode)
        print(json.dumps(result.to_dict(), indent=2, default=str) if args.json
              else render(result, args.show_trace))
        if args.save_trace:
            print(f"trace saved to {Path(save_trace(result)).relative_to(ROOT)}")
        return result

    if not args.interactive:
        result = handle(args.query)
        return 0 if result.status != "blocked" else 2
    print("VibeFinder 2.0. Describe what you want to hear, or 'quit'.")
    while True:
        try:
            query = input("\n> ")
        except EOFError:
            break
        if query.strip().lower() in {"quit", "exit", "q"}:
            break
        ask_feedback(handle(query))
    return 0


if __name__ == "__main__":
    sys.exit(main())

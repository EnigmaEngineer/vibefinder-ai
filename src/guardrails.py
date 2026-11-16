"""Input and output guardrails.

Input checks run before anything else and can block a request outright.
Output checks run on the final answer: every recommended song must be a real
catalog row, nothing may repeat, and the explanation may only name catalog
songs and cite documents that were actually retrieved. A failed output check
is never shown to the user; the agent falls back to the template explainer.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

MAX_QUERY_CHARS = 300
MAX_EXPLANATION_CHARS = 1800

INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
    r"(reveal|print|show|repeat) (me )?(your|the) (system )?(prompt|instructions)",
    r"\bsystem prompt\b",
    r"\byou are now\b",
    r"\bdeveloper mode\b",
    r"\bjailbreak\b",
]


@dataclass
class InputCheck:
    ok: bool
    cleaned: str
    reason: Optional[str] = None   # machine-readable code when blocked
    message: str = ""


def check_input(query) -> InputCheck:
    """Validate and clean a user request. Never raises."""
    if not isinstance(query, str):
        return InputCheck(False, "", "not_text", "The request must be text.")
    # Drop control characters, collapse whitespace.
    cleaned = "".join(ch for ch in query if unicodedata.category(ch)[0] != "C" or ch == " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return InputCheck(False, "", "empty", "Tell me what you want to listen to, e.g. 'calm music for studying'.")
    if len(cleaned) > MAX_QUERY_CHARS:
        return InputCheck(False, cleaned[:MAX_QUERY_CHARS], "too_long",
                          f"Please keep the request under {MAX_QUERY_CHARS} characters.")
    if not re.search(r"[a-zA-Z]", cleaned):
        return InputCheck(False, cleaned, "no_words", "I could not find any words in that request.")
    lowered = cleaned.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return InputCheck(False, cleaned, "prompt_injection",
                              "That looks like an attempt to change my instructions. I only recommend music.")
    return InputCheck(True, cleaned)


@dataclass
class OutputCheck:
    ok: bool
    problems: List[str] = field(default_factory=list)


def check_output(
    recommendations: Sequence,
    catalog: Iterable[Dict],
    explanation: str,
    retrieved_ids: Iterable[str],
    k: int,
) -> OutputCheck:
    """Verify the final answer against the catalog and the retrieved context."""
    catalog = list(catalog)
    by_id = {song["id"]: song for song in catalog}
    titles = {song["title"].lower() for song in catalog}
    problems: List[str] = []

    if len(recommendations) > k:
        problems.append(f"returned {len(recommendations)} songs, more than k={k}")
    seen = set()
    for song, _score, _why in recommendations:
        real = by_id.get(song.get("id"))
        if real is None or real["title"] != song.get("title"):
            problems.append(f"song not in catalog: {song.get('title')!r}")
        if song.get("id") in seen:
            problems.append(f"duplicate song: {song.get('title')!r}")
        seen.add(song.get("id"))

    if len(explanation) > MAX_EXPLANATION_CHARS:
        problems.append("explanation too long")
    for quoted in re.findall(r'"([^"]{2,60})"', explanation):
        if quoted.lower() not in titles and not _is_user_quote(quoted, explanation):
            problems.append(f"explanation names a song that is not in the catalog: {quoted!r}")
    allowed = set(retrieved_ids)
    for cited in re.findall(r"\[([a-z0-9-]+)\]", explanation):
        if cited not in allowed:
            problems.append(f"explanation cites a source that was not retrieved: [{cited}]")
    return OutputCheck(not problems, problems)


def _is_user_quote(quoted: str, explanation: str) -> bool:
    """The template echoes the request as: You asked for "...". That quote is allowed."""
    return f'You asked for "{quoted}"' in explanation

"""Turns the agent's picks into a short written answer.

Two explainers share one interface:

* TemplateExplainer (default): deterministic, offline, always available.
* GeminiExplainer: used when GEMINI_API_KEY is set and the google-genai package
  is installed. It gets the retrieved guides and the ranked songs as context and
  must cite guides as [doc-id]. Its output still goes through the output
  guardrail, and the agent falls back to the template if it fails.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Sequence

logger = logging.getLogger(__name__)

PROFILE_LABELS = {
    "genre": "genre", "mood": "mood", "energy": "energy", "valence": "positivity",
    "danceability": "danceability", "acousticness": "acousticness",
    "instrumentalness": "instrumental", "speechiness": "spoken word",
    "popularity": "popularity", "release_decade": "decade",
}


def describe_profile(profile: Dict) -> str:
    parts = []
    for key, label in PROFILE_LABELS.items():
        if key in profile:
            value = profile[key]
            parts.append(f"{label} {value:.2f}" if isinstance(value, float) else f"{label} {value}")
    tags = profile.get("detailed_mood_tags")
    if tags:
        parts.append("tags " + "/".join(tags))
    return ", ".join(parts) or "no preferences"


def _top_reasons(reasons: str, n: int = 2) -> str:
    """First n scoring reasons, shortened: 'genre match: 'lofi' (+3.00)' -> 'genre match'."""
    picked = []
    for reason in reasons.split("; "):
        if reason.startswith("diversity penalty"):
            continue
        head = reason.split(" (+")[0]
        if " vs target " in head:
            picked.append(f"{head.split(' ')[0]} close to target")
        else:
            picked.append(head.split(":")[0])
        if len(picked) == n:
            break
    return ", ".join(picked) or "closest overall fit"


class TemplateExplainer:
    name = "template"

    def explain(self, query: str, profile: Dict, recommendations: Sequence, sources: List[str],
                notes: List[str], context: Dict[str, str] | None = None) -> str:
        safe_query = query.replace('"', "'")
        lines = [f'You asked for "{safe_query}".']
        cite = " ".join(f"[{s}]" for s in sources)
        lines.append(f"I read that as: {describe_profile(profile)}" + (f" (from {cite})." if cite else "."))
        for i, (song, score, reasons) in enumerate(recommendations, start=1):
            lines.append(f'{i}. "{song["title"]}" by {song.get("artist", "unknown")} '
                         f'({song.get("genre")}, {song.get("mood")}) - {_top_reasons(reasons)}.')
        for note in notes:
            lines.append(f"Note: {note}")
        return "\n".join(lines)


class GeminiExplainer:
    name = "gemini"

    TIMEOUT_MS = 12_000  # a live demo can't wait on a slow API; past this we use the template

    def __init__(self, api_key: str, model: str):
        from google import genai  # imported lazily: optional dependency
        try:
            # One try, short timeout. The SDK's default retries can hang for minutes on a 503.
            self._client = genai.Client(api_key=api_key, http_options={
                "timeout": self.TIMEOUT_MS, "retry_options": {"attempts": 1}})
        except Exception:  # older SDKs without these options
            self._client = genai.Client(api_key=api_key, http_options={"timeout": self.TIMEOUT_MS})
        self._model = model

    def explain(self, query: str, profile: Dict, recommendations: Sequence, sources: List[str],
                notes: List[str], context: Dict[str, str] | None = None) -> str:
        songs = "\n".join(
            f'- "{s["title"]}" by {s.get("artist")} | {s.get("genre")}, {s.get("mood")}, energy {s.get("energy")} | why: {r}'
            for s, _score, r in recommendations
        )
        guides = "\n".join(f"[{doc_id}] {text}" for doc_id, text in (context or {}).items())
        prompt = (
            "You are a music recommendation assistant. Explain the picks below in under 150 words.\n"
            "Rules: only mention the songs listed, put double quotes around song titles and nothing else, "
            "cite guides only by their [id], do not invent facts about artists.\n\n"
            f"Request: {query}\nTaste profile: {describe_profile(profile)}\n"
            f"Guides:\n{guides or '(none)'}\nRanked songs:\n{songs}\n"
            f"Caveats to mention: {'; '.join(notes) or 'none'}"
        )
        response = self._client.models.generate_content(model=self._model, contents=prompt)
        return (response.text or "").strip()


ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_env_file(path: Path = ENV_FILE) -> None:
    """Read KEY=value lines from .env into os.environ (real env vars win). No dependency needed."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_explainer(prefer_llm: bool = True):
    """Gemini when configured, otherwise the template. Never raises."""
    if not prefer_llm:
        return TemplateExplainer()
    load_env_file()
    key = os.environ.get("GEMINI_API_KEY")
    if prefer_llm and key:
        try:
            return GeminiExplainer(key, os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"))
        except Exception as exc:  # missing package, bad key format, etc.
            logger.warning("Gemini explainer unavailable (%s); using template", exc)
    return TemplateExplainer()

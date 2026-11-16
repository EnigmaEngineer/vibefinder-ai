"""VibeFinder 2.0 agent: plan -> retrieve -> build profile -> recommend -> self-check -> explain -> verify.

The agent wraps the Module 3 scoring engine (src/recommender.py) without
changing it. Every step is written to a trace so a run can be audited later
(see save_trace and logs/agent_traces.md).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.explainer import TemplateExplainer, get_explainer
from src.guardrails import check_input, check_output
from src.query_parser import parse_query
from src.recommender import (TAGS_COLUMN, get_strategy, get_top_recommendations, load_songs)
from src.retriever import NUMERIC_HINTS, Hit, Retriever, load_knowledge_base, song_documents

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATHS = (ROOT / "data" / "songs.csv", ROOT / "data" / "songs_extra.csv")
KNOWLEDGE_DIR = ROOT / "knowledge"

MIN_DOC_SCORE = 0.12     # a guide below this similarity is ignored
SEED_SONG_SCORE = 0.45   # a song/artist match above this is treated as "more like this"
LOW_CONFIDENCE = 0.5     # below this the agent revises, then flags for human review
HIGH_CONFIDENCE = 0.75
MAX_REVISIONS = 2
MAX_TAGS = 4
UNSUPPORTED_CAP = 0.45  # confidence ceiling when the user asked for a genre we don't carry


def load_catalog(paths=CATALOG_PATHS) -> List[Dict]:
    """Load and merge every catalog CSV. A repeated id keeps the first row."""
    songs, seen = [], set()
    for path in paths:
        if not Path(path).exists():
            logger.warning("catalog file missing: %s", path)
            continue
        for song in load_songs(str(path)):
            if song["id"] in seen:
                logger.warning("duplicate song id %s in %s skipped", song["id"], path)
                continue
            seen.add(song["id"])
            songs.append(song)
    return songs


@dataclass
class AgentResult:
    query: str
    status: str                      # "ok", "blocked" or "clarify"
    message: str = ""
    plan: List[str] = field(default_factory=list)
    profile: Dict = field(default_factory=dict)
    mode: Optional[str] = None
    recommendations: List = field(default_factory=list)   # (song, score, reasons)
    explanation: str = ""
    explainer: str = "template"
    confidence: float = 0.0
    confidence_label: str = "none"
    needs_review: bool = False
    notes: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    seed_song: Optional[str] = None
    attempts: int = 0
    use_rag: bool = True
    guardrail: Dict = field(default_factory=dict)
    trace: List[Dict] = field(default_factory=list)

    def titles(self) -> List[str]:
        return [song["title"] for song, _, _ in self.recommendations]

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["recommendations"] = [
            {"title": s["title"], "artist": s.get("artist"), "genre": s.get("genre"),
             "mood": s.get("mood"), "score": round(score, 2), "reasons": why}
            for s, score, why in self.recommendations
        ]
        return data


def max_possible_score(profile: Dict, mode) -> float:
    """Most points any song could earn for this profile under this strategy."""
    weights = get_strategy(mode).weights()
    total = 0.0
    for key in ("genre", "mood", "energy", "valence", "danceability", "acousticness",
                "instrumentalness", "speechiness", "popularity", "release_decade"):
        if key in profile:
            total += weights[key]
    total += weights["tag_match"] * len(profile.get(TAGS_COLUMN) or [])
    return total


def confidence_for(profile: Dict, recs: List, mode) -> float:
    """fit x specificity.

    fit: average of (base score / max possible) over the top 3 picks, using the
    score before any diversity penalty. specificity: a profile with fewer than
    4 signals can't be trusted much even when it matches perfectly ("pop" alone
    matches every pop song), so it scales the result down.
    """
    if not recs:
        return 0.0
    ceiling = max_possible_score(profile, mode)
    if ceiling <= 0:
        return 0.0
    strategy = get_strategy(mode)
    top = recs[:3]
    fit = sum(strategy.score(profile, song)[0] / ceiling for song, _, _ in top) / len(top)
    specificity = min(1.0, len(profile) / 4)
    return round(min(1.0, fit) * specificity, 2)


def label_for(confidence: float) -> str:
    if confidence >= HIGH_CONFIDENCE:
        return "high"
    if confidence >= LOW_CONFIDENCE:
        return "medium"
    return "low"


class VibeAgent:
    def __init__(self, catalog: Optional[List[Dict]] = None, knowledge=None, explainer=None,
                 use_rag: bool = True, k: int = 5):
        self.catalog = catalog if catalog is not None else load_catalog()
        self.knowledge = knowledge if knowledge is not None else load_knowledge_base(KNOWLEDGE_DIR)
        self.retriever = Retriever(self.knowledge + song_documents(self.catalog))
        self.explainer = explainer if explainer is not None else get_explainer()
        self.use_rag = use_rag
        self.k = k
        self.catalog_genres = {s["genre"].lower() for s in self.catalog}
        self.catalog_moods = {s["mood"].lower() for s in self.catalog}

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _step(result: AgentResult, name: str, detail: str, **data) -> None:
        result.trace.append({"step": len(result.trace) + 1, "name": name, "detail": detail, **data})
        logger.info("[%s] %s", name, detail)

    def _profile_from_hits(self, hits: List[Hit]) -> Dict:
        """Merge the taste hints of the retrieved guides, weighted by similarity."""
        profile: Dict = {}
        sums: Dict[str, float] = {}
        weights: Dict[str, float] = {}
        tags: List[str] = []
        for hit in hits:
            hints = hit.doc.profile
            for key in ("genre", "mood"):
                if key in hints and key not in profile:
                    profile[key] = hints[key]       # best-scoring guide wins
            for key in NUMERIC_HINTS:
                if key in hints:
                    sums[key] = sums.get(key, 0.0) + hints[key] * hit.score
                    weights[key] = weights.get(key, 0.0) + hit.score
            for tag in hints.get(TAGS_COLUMN, []):
                if tag not in tags and len(tags) < MAX_TAGS:
                    tags.append(tag)
        for key, total in sums.items():
            value = total / weights[key]
            profile[key] = round(value) if key == "popularity" else round(value, 2)
        if tags:
            profile[TAGS_COLUMN] = tags
        return profile

    @staticmethod
    def _profile_from_song(song: Dict) -> Dict:
        keys = ("genre", "mood", "energy", "valence", "danceability", "acousticness")
        profile = {k: song[k] for k in keys if song.get(k) not in (None, "")}
        if song.get(TAGS_COLUMN):
            profile[TAGS_COLUMN] = list(song[TAGS_COLUMN])[:3]
        return profile

    def _recommend(self, profile: Dict, mode, exclude_id=None):
        pool = [s for s in self.catalog if s["id"] != exclude_id]
        recs = get_top_recommendations(profile, pool, n=self.k, mode=mode)
        return recs, confidence_for(profile, recs, mode)

    def _revise(self, profile: Dict, mode, recs, revision: int):
        """Pick one way to loosen the request. Returns (profile, mode, why) or None."""
        impossible = [k for k, have in (("genre", self.catalog_genres), ("mood", self.catalog_moods))
                      if k in profile and str(profile[k]).lower() not in have]
        if impossible:
            new = {k: v for k, v in profile.items() if k not in impossible}
            return new, mode, f"dropped {', '.join(impossible)} (no catalog song has that label)"
        top = [song for song, _, _ in recs[:3]]
        for key in ("mood", "genre"):
            if key in profile and not any(str(s.get(key)).lower() == str(profile[key]).lower() for s in top):
                new = {k: v for k, v in profile.items() if k != key}
                return new, mode, f"dropped {key}={profile[key]!r} (none of the top 3 matched it)"
        if mode != "energy_focused":
            return profile, "energy_focused", "switched to energy_focused scoring"
        return None

    # -- main loop ---------------------------------------------------------
    def run(self, query: str, mode: Optional[str] = None) -> AgentResult:
        started = time.perf_counter()
        result = AgentResult(query=query if isinstance(query, str) else repr(query), status="ok",
                             use_rag=self.use_rag)

        # 1. Input guardrail
        check = check_input(query)
        self._step(result, "input_guardrail", "passed" if check.ok else f"blocked: {check.reason}",
                   reason=check.reason)
        if not check.ok:
            result.status, result.message = "blocked", check.message
            return result
        text = check.cleaned
        result.query = text

        # 2. Read explicit cues
        parsed = parse_query(text)
        self._step(result, "parse", f"explicit cues: {', '.join(parsed.matched) or 'none'}",
                   explicit=parsed.prefs, unsupported=parsed.unsupported_genres)

        # 3. Retrieve from both sources
        doc_hits: List[Hit] = []
        seed: Optional[Dict] = None
        if self.use_rag:
            doc_hits = self.retriever.search(text, k=3, min_score=MIN_DOC_SCORE, source="knowledge")
            song_hits = self.retriever.search(text, k=1, min_score=SEED_SONG_SCORE, source="catalog")
            if song_hits:
                seed = song_hits[0].doc.song
            self._step(result, "retrieve",
                       f"{len(doc_hits)} guide(s), seed song: {seed['title'] if seed else 'none'}",
                       guides=[{"id": h.doc.id, "score": h.score} for h in doc_hits],
                       seed=seed["title"] if seed else None)
        else:
            self._step(result, "retrieve", "skipped (RAG disabled)")
        # Sources are what the profile is built from: the seed song, or the guides.
        result.sources = [f"song-{seed['id']}"] if seed else [h.doc.id for h in doc_hits]

        # 4. Plan
        if mode is None:
            if seed:
                mode = None
            elif "genre" in parsed.prefs:
                mode = "genre_first"
            elif "mood" in parsed.prefs or any("mood" in h.doc.profile for h in doc_hits):
                mode = "mood_first"
        result.plan = [
            "start from seed song features" if seed else
            ("merge guide hints with explicit cues" if doc_hits else "use explicit cues only"),
            f"score catalog with {mode or 'default'} weights + diversity penalty",
            f"self-check confidence (revise up to {MAX_REVISIONS}x if below {LOW_CONFIDENCE})",
            f"explain with {self.explainer.name} explainer, then verify against catalog",
        ]
        self._step(result, "plan", " -> ".join(result.plan), mode=mode)

        # 5. Build the taste profile
        if seed:
            profile = self._profile_from_song(seed)
            result.seed_song = seed["title"]
        else:
            profile = self._profile_from_hits(doc_hits)
        profile.update(parsed.prefs)          # the user's own words beat retrieved hints
        for genre in parsed.unsupported_genres:
            result.notes.append(f"the catalog has no {genre} songs, so genre was left out")
        self._step(result, "build_profile", f"{len(profile)} signal(s)", profile=profile)
        if not profile:
            result.status = "clarify"
            result.message = ("I couldn't tell what kind of music you want. Try naming a genre, a mood, "
                              "or an activity (e.g. 'upbeat songs for a run').")
            result.needs_review = True
            self._step(result, "clarify", "no usable signals; asking the user to rephrase")
            return result

        # 6. Act, check, revise
        best = None
        current_profile, current_mode = dict(profile), mode
        changes: List[str] = []          # revisions that led to the current attempt
        for attempt in range(MAX_REVISIONS + 1):
            recs, conf = self._recommend(current_profile, current_mode, seed["id"] if seed else None)
            self._step(result, "recommend", f"attempt {attempt + 1}: confidence {conf:.2f}",
                       mode=current_mode, top=[s["title"] for s, _, _ in recs[:3]], confidence=conf)
            if best is None or conf > best[2]:
                best = (current_profile, current_mode, conf, recs, list(changes))
            if conf >= LOW_CONFIDENCE or attempt == MAX_REVISIONS:
                break
            revision = self._revise(current_profile, current_mode, recs, attempt)
            if revision is None:
                self._step(result, "self_check", "low confidence but nothing left to relax")
                break
            current_profile, current_mode, why = revision
            changes.append(why)
            self._step(result, "self_check", f"confidence {conf:.2f} < {LOW_CONFIDENCE}: {why}")
        result.attempts = sum(1 for t in result.trace if t["name"] == "recommend")
        result.profile, result.mode, result.confidence, result.recommendations, kept = best
        result.notes += [f"to find closer matches I {why}" for why in kept]
        if result.attempts > 1:
            self._step(result, "self_check", f"kept the best of {result.attempts} attempts "
                       f"(confidence {result.confidence:.2f}, mode {result.mode or 'default'})")
        if parsed.unsupported_genres and result.confidence >= LOW_CONFIDENCE:
            # The picks can match the hints well and still not be what was asked for.
            self._step(result, "self_check", f"confidence capped from {result.confidence:.2f} to "
                       f"{UNSUPPORTED_CAP}: requested genre is not in the catalog")
            result.confidence = UNSUPPORTED_CAP
        result.confidence_label = label_for(result.confidence)
        result.needs_review = result.confidence < LOW_CONFIDENCE
        if result.needs_review:
            result.notes.append("low confidence: these are the closest songs, not strong matches")

        # 7. Explain, then verify the final answer
        context = ({f"song-{seed['id']}": f"seed song {seed['title']} by {seed.get('artist')}"} if seed
                   else {h.doc.id: h.doc.text for h in doc_hits})
        try:
            result.explanation = self.explainer.explain(text, result.profile, result.recommendations,
                                                        result.sources, result.notes, context=context)
            result.explainer = self.explainer.name
        except Exception as exc:
            self._step(result, "explain", f"{self.explainer.name} failed ({exc}); using template")
            result.explanation = ""
        verdict = check_output(result.recommendations, self.catalog, result.explanation,
                               result.sources, self.k)
        if not result.explanation or not verdict.ok:
            if result.explanation:
                self._step(result, "output_guardrail", f"rejected {self.explainer.name} text",
                           problems=verdict.problems)
            result.explanation = TemplateExplainer().explain(text, result.profile, result.recommendations,
                                                             result.sources, result.notes)
            result.explainer = "template (fallback)" if self.explainer.name != "template" else "template"
            verdict = check_output(result.recommendations, self.catalog, result.explanation,
                                   result.sources, self.k)
        result.guardrail = {"ok": verdict.ok, "problems": verdict.problems}
        self._step(result, "output_guardrail", "passed" if verdict.ok else "FAILED", problems=verdict.problems)
        if not verdict.ok:   # should never happen with the template; never show an unverified answer
            result.status, result.message = "blocked", "Internal check failed; no answer shown."
            result.recommendations = []
        self._step(result, "done", f"{result.confidence_label} confidence in "
                   f"{(time.perf_counter() - started) * 1000:.1f} ms")
        return result


def save_trace(result: AgentResult, folder: str | Path = ROOT / "logs" / "traces") -> Path:
    """Write one run as JSON (for tools) and return the path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = folder / f"trace-{stamp}-{abs(hash(result.query)) % 10_000:04d}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
    return path


def trace_to_markdown(result: AgentResult) -> str:
    """Human-readable version of one run's reasoning trace."""
    lines = [f"### Query: `{result.query}`", "",
             f"- status: **{result.status}**, RAG: {'on' if result.use_rag else 'off'}, "
             f"confidence: {result.confidence:.2f} ({result.confidence_label}), "
             f"needs human review: {result.needs_review}", ""]
    lines.append("| # | step | detail |")
    lines.append("|---|---|---|")
    for step in result.trace:
        extra = {k: v for k, v in step.items() if k not in ("step", "name", "detail") and v not in (None, [], {})}
        detail = step["detail"] + (f" `{json.dumps(extra, default=str)}`" if extra else "")
        lines.append(f"| {step['step']} | {step['name']} | {detail.replace('|', '/')} |")
    if result.titles():
        lines += ["", "Final picks: " + ", ".join(result.titles())]
    if result.message:
        lines += ["", f"Message: {result.message}"]
    return "\n".join(lines) + "\n"

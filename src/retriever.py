"""Retrieval layer for VibeFinder 2.0.

Two sources are indexed side by side:

* the knowledge base: short markdown guides in knowledge/ (one per genre or
  listening activity), each with a `profile:` line of taste hints;
* the song catalog itself, so a query that names a song or an artist can be
  matched to that song ("more like Night Drive Loop").

Retrieval is TF-IDF with cosine similarity, written in plain Python so it runs
anywhere with no model download. Keywords in a doc's front matter count twice.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "for", "from",
    "give", "i", "im", "in", "is", "it", "its", "me", "my", "of", "on", "or",
    "please", "play", "some", "something", "song", "songs", "that", "the", "this",
    "to", "track", "tracks", "want", "with", "you", "music", "playlist", "need",
    "like", "good", "get", "put", "make", "while", "just", "really", "very",
}

NUMERIC_HINTS = ("energy", "valence", "danceability", "acousticness",
                 "instrumentalness", "speechiness", "popularity")


def _stem(word: str) -> str:
    """Tiny suffix stripper: studying -> study, running -> run, beats -> beat."""
    if len(word) > 5 and word.endswith("ing"):
        word = word[:-3]
        if len(word) > 2 and word[-1] == word[-2] and word[-1] not in "ls":
            word = word[:-1]
    elif len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return word


def tokenize(text: str) -> List[str]:
    """Lowercase words (stemmed, stopwords dropped) plus adjacent-word bigrams."""
    text = text.lower().replace("hip-hop", "hip hop").replace("lo-fi", "lofi").replace("r&b", "rnb")
    words = [_stem(w) for w in re.findall(r"[a-z0-9]+", text) if w not in STOPWORDS]
    words = [w for w in words if w]
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


@dataclass
class Document:
    """One retrievable item: a knowledge-base guide or a catalog song."""
    id: str
    source: str            # "knowledge" or "catalog"
    title: str
    text: str
    keywords: List[str] = field(default_factory=list)
    profile: Dict = field(default_factory=dict)   # taste hints this doc implies
    song: Optional[Dict] = None                   # set for catalog docs


@dataclass
class Hit:
    doc: Document
    score: float


def parse_profile_line(line: str) -> Dict:
    """Parse 'genre=lofi, energy=0.4, tags=focused, late night' into a dict.

    Numbers become floats. Everything after 'tags=' is a list of tags.
    """
    profile: Dict = {}
    if "tags=" in line:
        line, tag_part = line.split("tags=", 1)
        profile["detailed_mood_tags"] = [t.strip() for t in tag_part.split(",") if t.strip()]
    for part in line.split(","):
        if "=" not in part:
            continue
        key, value = (s.strip() for s in part.split("=", 1))
        if key in NUMERIC_HINTS:
            profile[key] = float(value)
        elif key:
            profile[key] = value
    return profile


def load_knowledge_base(folder: str | Path = "knowledge") -> List[Document]:
    """Read every knowledge/*.md file with a small key: value front-matter block."""
    docs: List[Document] = []
    for path in sorted(Path(folder).glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        match = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.S)
        if not match:
            logger.warning("skipping %s: no front matter", path.name)
            continue
        meta = {}
        for line in match.group(1).splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip()
        docs.append(Document(
            id=meta.get("id", path.stem),
            source="knowledge",
            title=meta.get("title", path.stem),
            text=match.group(2).strip(),
            keywords=[k.strip() for k in meta.get("keywords", "").split(",") if k.strip()],
            profile=parse_profile_line(meta.get("profile", "")),
        ))
    logger.info("loaded %d knowledge docs from %s", len(docs), folder)
    return docs


def song_documents(songs: Iterable[Dict]) -> List[Document]:
    """Turn catalog rows into documents so songs and artists can be retrieved by name."""
    # Only title and artist are indexed. Genre, mood and tags belong to the guides;
    # indexing them here would make every "chill" query look like a song request.
    return [
        Document(
            id=f"song-{song['id']}",
            source="catalog",
            title=song["title"],
            text=f"{song['title']} {song.get('artist', '')}",
            keywords=[song["title"], song.get("artist", "")],
            song=song,
        )
        for song in songs
    ]


class Retriever:
    """TF-IDF index over any list of Documents."""

    def __init__(self, docs: List[Document]):
        self.docs = docs
        self._vectors: List[Dict[str, float]] = []
        df: Counter = Counter()
        bags = []
        for doc in docs:
            bag = Counter(tokenize(f"{doc.title} {doc.text}"))
            for kw in doc.keywords:
                for tok in tokenize(kw):
                    bag[tok] += 2
            bags.append(bag)
            df.update(bag.keys())
        n = max(len(docs), 1)
        self.idf = {tok: math.log((1 + n) / (1 + count)) + 1.0 for tok, count in df.items()}
        self._vectors = [self._weigh(bag) for bag in bags]

    def _weigh(self, bag: Counter) -> Dict[str, float]:
        vec = {tok: (1 + math.log(tf)) * self.idf.get(tok, 0.0) for tok, tf in bag.items() if tf > 0}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {tok: v / norm for tok, v in vec.items()}

    def search(self, query: str, k: int = 3, min_score: float = 0.0, source: Optional[str] = None) -> List[Hit]:
        """Top-k documents by cosine similarity, best first. Ties go to the lower doc id."""
        qvec = self._weigh(Counter(t for t in tokenize(query) if t in self.idf))
        hits = []
        for doc, dvec in zip(self.docs, self._vectors):
            if source and doc.source != source:
                continue
            score = sum(w * dvec.get(tok, 0.0) for tok, w in qvec.items())
            if score > min_score:
                hits.append(Hit(doc, round(score, 4)))
        hits.sort(key=lambda h: (-h.score, h.doc.id))
        return hits[:k]

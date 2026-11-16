"""Rule-based reading of explicit cues in a free-text request.

This is the "no retrieval" baseline: it only understands words that name a
feature directly (a genre, a mood, "high energy", "acoustic", "90s"). It knows
nothing about activities such as studying or sleeping; that knowledge comes
from the retrieved guides in knowledge/.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

# Phrase -> catalog genre. Longer phrases are checked first ("indie pop" before "pop").
GENRE_SYNONYMS = {
    "indie pop": "indie pop", "hip hop": "hip hop", "hip-hop": "hip hop", "rap": "hip hop",
    "lo-fi": "lofi", "lofi": "lofi", "r&b": "r&b", "rnb": "r&b", "synthwave": "synthwave",
    "edm": "edm", "techno": "edm", "house music": "edm", "classical": "classical",
    "ambient": "ambient", "jazz": "jazz", "rock": "rock", "folk": "folk", "pop": "pop",
}
# Genres people ask for that the catalog does not carry. Spotting them lets the
# agent say so instead of silently returning something else.
UNSUPPORTED_GENRES = ("heavy metal", "metal", "country", "reggae", "punk", "k-pop", "kpop",
                      "opera", "blues", "salsa", "bluegrass")

MOOD_SYNONYMS = {
    "happy": "happy", "chill": "chill", "intense": "intense", "relaxed": "relaxed",
    "moody": "moody", "focused": "focused", "calm": "calm", "peaceful": "calm",
    "sad": "sad", "melancholy": "sad", "romantic": "romantic", "energetic": "energetic",
    "euphoric": "euphoric", "confident": "confident", "warm": "warm", "angry": "intense",
}

ENERGY_PHRASES = [
    (r"\b(high[- ]energy|hype|pumped|fast|banger|bangers|loud)\b", 0.9),
    (r"\b(medium[- ]energy|moderate|mid[- ]tempo)\b", 0.55),
    (r"\b(low[- ]energy|slow|quiet|soft|mellow|gentle)\b", 0.25),
]


@dataclass
class ParsedQuery:
    prefs: Dict = field(default_factory=dict)
    unsupported_genres: List[str] = field(default_factory=list)
    matched: List[str] = field(default_factory=list)   # human-readable cues found

    @property
    def signal_count(self) -> int:
        return len(self.prefs)


def _has(phrase: str, text: str) -> bool:
    return re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", text) is not None


def parse_query(query: str) -> ParsedQuery:
    """Pull explicit preferences out of the request text."""
    text = query.lower()
    out = ParsedQuery()

    for phrase in UNSUPPORTED_GENRES:
        if _has(phrase, text):
            name = "metal" if "metal" in phrase else phrase.replace("kpop", "k-pop")
            if name not in out.unsupported_genres:
                out.unsupported_genres.append(name)
            text = text.replace(phrase, " ")  # so "heavy metal" doesn't also hit "heavy"

    for phrase in sorted(GENRE_SYNONYMS, key=len, reverse=True):
        if _has(phrase, text):
            out.prefs["genre"] = GENRE_SYNONYMS[phrase]
            out.matched.append(f"genre '{phrase}'")
            break

    for word, mood in MOOD_SYNONYMS.items():
        if _has(word, text):
            out.prefs["mood"] = mood
            out.matched.append(f"mood '{word}'")
            break

    for pattern, value in ENERGY_PHRASES:
        found = re.search(pattern, text)
        if found:
            out.prefs["energy"] = value
            out.matched.append(f"energy '{found.group(0)}'")
            break

    if _has("acoustic", text) or _has("unplugged", text):
        out.prefs["acousticness"] = 0.85
        out.matched.append("acoustic")
    elif _has("electronic", text):
        out.prefs["acousticness"] = 0.1
        out.matched.append("electronic")

    if re.search(r"\b(instrumental|no vocals|no lyrics|without lyrics)\b", text):
        out.prefs["instrumentalness"] = 0.85
        out.matched.append("instrumental")

    if re.search(r"\b(danceable|dance)\b", text):
        out.prefs["danceability"] = 0.85
        out.matched.append("danceable")

    decade = re.search(r"\b(?:19|20)?([0-9])0s\b", text)
    if decade:
        digit = int(decade.group(1))
        out.prefs["release_decade"] = (1900 if digit >= 5 else 2000) + digit * 10
        out.matched.append(f"decade {decade.group(0)}")

    if re.search(r"\b(popular|hits|top 40|mainstream)\b", text):
        out.prefs["popularity"] = 80
        out.matched.append("popular")
    elif re.search(r"\b(underground|deep cuts|obscure)\b", text):
        out.prefs["popularity"] = 40
        out.matched.append("underground")

    return out

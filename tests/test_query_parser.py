import pytest
from src.query_parser import parse_query


@pytest.mark.parametrize("text,key,value", [
    ("some lo-fi please", "genre", "lofi"),
    ("indie pop for a picnic", "genre", "indie pop"),
    ("rap for the commute", "genre", "hip hop"),
    ("high energy stuff", "energy", 0.9),
    ("slow songs", "energy", 0.25),
    ("acoustic covers", "acousticness", 0.85),
    ("instrumental only, no vocals", "instrumentalness", 0.85),
    ("90s throwbacks", "release_decade", 1990),
    ("2010s hits", "release_decade", 2010),
    ("sad songs", "mood", "sad"),
])
def test_explicit_cues(text, key, value):
    assert parse_query(text).prefs[key] == value


def test_indie_pop_beats_pop():
    assert parse_query("indie pop").prefs["genre"] == "indie pop"


def test_unsupported_genre_is_flagged_not_used():
    parsed = parse_query("heavy metal for lifting")
    assert parsed.unsupported_genres == ["metal"]
    assert "genre" not in parsed.prefs


def test_word_inside_other_word_is_not_a_match():
    assert "genre" not in parse_query("wrap it up").prefs   # 'rap' inside 'wrap'


def test_activity_words_are_not_understood_without_retrieval():
    assert parse_query("music for studying").prefs == {}

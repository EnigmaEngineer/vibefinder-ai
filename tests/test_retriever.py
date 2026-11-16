from src.retriever import Retriever, load_knowledge_base, parse_profile_line, song_documents, tokenize
from src.agent import KNOWLEDGE_DIR, load_catalog


def test_tokenize_stems_and_bigrams():
    toks = tokenize("Studying with lo-fi beats")
    assert "study" in toks and "lofi" in toks and "beat" in toks
    assert "study_lofi" in toks


def test_parse_profile_line_numbers_and_tags():
    p = parse_profile_line("genre=lofi, energy=0.4, tags=focused, late night")
    assert p == {"genre": "lofi", "energy": 0.4, "detailed_mood_tags": ["focused", "late night"]}


def test_knowledge_base_loads_every_guide_with_hints():
    docs = load_knowledge_base(KNOWLEDGE_DIR)
    assert len(docs) >= 18
    assert all(d.profile for d in docs), "every guide needs a profile line"
    assert len({d.id for d in docs}) == len(docs)


def test_activity_query_finds_the_activity_guide():
    r = Retriever(load_knowledge_base(KNOWLEDGE_DIR))
    assert r.search("something to help me fall asleep", k=1)[0].doc.id == "activity-sleep"
    assert r.search("songs for the gym", k=1)[0].doc.id == "activity-workout"


def test_song_titles_are_retrievable():
    r = Retriever(song_documents(load_catalog()))
    hit = r.search("more like Night Drive Loop", k=1)[0]
    assert hit.doc.song["title"] == "Night Drive Loop"


def test_unrelated_query_scores_nothing():
    r = Retriever(load_knowledge_base(KNOWLEDGE_DIR))
    assert r.search("what is the capital of France", k=3, min_score=0.12) == []

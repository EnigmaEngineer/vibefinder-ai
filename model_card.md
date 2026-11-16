# Model Card: VibeFinder 2.0

## Model details

- **Name:** VibeFinder 2.0 (extends VibeFinder 1.0, my Module 3 Music Recommender Simulation)
- **Type:** content-based recommender wrapped in a retrieval-augmented agent loop
- **Components:** TF-IDF retriever over 20 custom guides + the song catalog, rule-based cue parser, the Module 3 weighted scorer, a self-checking agent with a confidence score, input and output guardrails, an optional Gemini explainer
- **Default mode:** fully deterministic and offline. No model weights, no training.
- **Original card:** [docs/module3_model_card.md](docs/module3_model_card.md)

## Intended use

Suggesting songs from a small demo catalog based on a free-text request ("something for a rainy day", "more like Night Drive Loop"). It is a learning project and a portfolio piece. It is not meant for a real music service. The catalog is 22 made-up songs.

## Data

- `data/songs.csv`: the original 10 songs from Module 3 (8 of them still have blank optional columns)
- `data/songs_extra.csv`: 12 new songs I wrote with every column filled in, covering genres the first file lacked (hip hop, classical, EDM, folk, R&B)
- `knowledge/*.md`: 20 short guides I wrote, 8 for activities (study, workout, sleep, party, night drive, rainy day, date night, heartbreak) and 12 for genres. Each has keywords and a `profile:` line of taste hints like `energy=0.15, instrumentalness=0.9`.

The hint numbers are my own judgment, not measured from listening data. That is the biggest source of bias in the system (see below).

## How it works

1. Input guardrail blocks empty, overlong, wordless and prompt-injection requests.
2. The parser pulls explicit cues (genre names, "high energy", "acoustic", "90s").
3. The retriever finds up to 3 guides above a 0.12 similarity cutoff, and checks whether a song title or artist was named.
4. The agent plans a scoring mode and builds a taste profile: copy the seed song, or merge the guides' hints weighted by similarity. Explicit cues override hints. No signals at all means it asks the user to rephrase.
5. The Module 3 scorer ranks the catalog with a diversity penalty.
6. Self-check: confidence = average fit of the top 3 x `min(1, signals/4)`. Below 0.5 it revises (up to 2 times) and keeps the best attempt. A request for an unsupported genre is capped at 0.45.
7. The explainer writes the answer. The output guardrail checks that every song is real, nothing repeats, no quoted title is invented and every `[citation]` was actually retrieved. Failures fall back to the template.

## Evaluation

### Automated tests

`pytest`: 487 passed, 6 xfailed. 48 tests are new for this project. The 6 xfails are Module 3 bugs I documented back then and left as markers.

### Evaluation harness (`python scripts/evaluate.py`)

17 predefined cases, each with checks on status, top-pick genre/mood/title, feature thresholds and confidence.

| Config | Passed | Avg confidence (answered) | Revisions used |
|---|---|---|---|
| RAG + agent | 15/17 (88%) | 0.63 | 1 |
| Baseline, no RAG | 5/17 (29%) | 0.26 | 7 |

Both failures in the full system are known limitations kept in the suite on purpose (negation and conflicting preferences, below). Per-case results: [eval/results.md](eval/results.md).

### Before/after: what retrieval changed

| Request | Without RAG | With RAG |
|---|---|---|
| "something to help me fall asleep" | `clarify`, nothing returned | Tidepool Drift (energy 0.15), Nocturne in Glass. Confidence 0.77 from [activity-sleep] |
| "music for a rainy day" | `clarify` | Window Seat (folk, tags rainy/reflective). Confidence 0.86 |
| "lofi beats for studying" | Profile is just `genre=lofi`. Confidence 0.25, flagged | 7-signal profile (lofi, chill, energy 0.39, instrumental 0.82, few words). Confidence 0.71 |
| "romantic r&b for date night" | Velvet Static, but confidence 0.17 | Velvet Static, confidence 0.60 |
| "more songs like Night Drive Loop" | `clarify` | Profile copied from the song, the song itself excluded |

Adding the second catalog file mattered too. In the original 10 songs, Spacewalk Thoughts was the only real sleep candidate and no song had a romantic mood at all.

### Human review

I read every output in `logs/agent_traces.md` and the eval tables by hand. The ones I disagreed with are the limitations below. The interactive CLI logs 1-5 ratings to `logs/feedback.jsonl` for future review, but I don't have enough ratings yet to report a number.

## Limitations and biases

- **My taste is baked in.** The guides say sleep music is ambient and classical, party music is popular and danceable, sad music is acoustic. Plenty of people sleep to rain sounds or work out to metal. The system will keep steering them to my defaults.
- **No negation.** "I'm not sad, I want something upbeat" returns sad songs. The parser sees the word "sad" and the user's words win. Kept as a failing eval case.
- **Conflicting asks lose a side.** "Loud acoustic jazz for working out" returns quiet jazz at confidence 0.67. Genre-first scoring outweighs the energy request, and the confidence score doesn't notice the contradiction. Same bug as the Module 3 "Conflicted Listener" profile.
- **Tiny catalog.** 22 songs, one synthwave track, no metal, country or k-pop. The diversity penalty means a 5-song list almost always includes a weak pick.
- **Artist requests.** "songs by Ada Verlaine" treats the first matching song as a seed and excludes it, so you get one Ada Verlaine song instead of all of them.
- **Keyword retrieval.** No synonyms. "I'm exhausted" won't find the sleep guide.
- **English only**, and the energy words ("slow", "loud") are crude.
- **Gemini path is untested against the live API.** The code and the fallback are unit-tested with fake explainers, but every result in this repo was produced in template mode.

## Misuse and prevention

The realistic risks are small for a song recommender, but the patterns transfer:

- **Prompt injection** (aimed at the optional LLM): blocked at input by pattern checks. The LLM also only ever sees retrieved guides and the ranked list, never the raw catalog or instructions it could leak.
- **Hallucinated songs or sources:** the output guardrail rejects any answer that names a song outside the catalog or cites a document that wasn't retrieved, and falls back to the template.
- **Overconfident answers:** low confidence and unsupported genres are flagged for human review instead of presented as good matches.
- **Abuse as a free general chatbot:** off-topic requests get a `clarify` response, not an answer.
- **Not covered:** the pattern list is easy to rephrase around, and I don't filter offensive song titles because the catalog is mine. A real deployment would need both.

## What surprised me while testing

- **My confidence score was confidently wrong.** "heavy metal for headbanging" came back at 0.93 "high" while returning zero metal songs. The profile it built (rock, high energy) was matched almost perfectly, so fit looked great. I added the unsupported-genre cap after seeing that. A similar thing happened with one-word profiles: "pop" alone scored a perfect 1.0, which is why specificity is in the formula now.
- **My first eval passed 14/14.** That felt good for about a minute. I wrote the cases while building the system, so of course they passed. I went looking for requests that should break it and added negation and conflict as permanent failing cases. 15/17 is a more honest number.
- **The revision loop barely runs with RAG on.** Only "chill but not boring" triggers it. Retrieval does most of the heavy lifting, and the loop mostly matters for vague or impossible requests.
- **A logging bug hid in the notes.** Early on, the agent told users "I switched to energy-focused scoring" even when it threw that attempt away and kept the first one. The trace showed the truth. The note didn't. I changed it so notes only describe the attempt that was kept.

## AI collaboration

I built this with Claude as a pair programmer. I used it to plan the extension from the rubric, write most of the first-draft code and tests, render the Mermaid diagram, and run the eval and traces. I made the scope calls and reviewed the outputs against the rubric and against what I'd actually want from a recommender.

**A suggestion I accepted:** wrap the Module 3 scorer instead of rewriting it. The agent calls `get_top_recommendations` and `get_strategy().weights()` and never edits `recommender.py`. That kept all 439 original tests valid as proof the scoring still works, and it made the diagram honest because the "Recommend" box is literally the old code.

**A suggestion I changed:** the first confidence formula was plain fit (top-3 score / max possible). It was simple and passed the early cases. When I tested it on vague and impossible requests it gave "pop" a perfect score and "heavy metal" 0.93. I kept the idea but added the specificity factor and the unsupported-genre cap.

**A suggestion I rejected:** using sentence-transformer embeddings for retrieval, and building a Streamlit UI (the starter already listed `streamlit` in requirements). Both are reasonable, but embeddings mean a model download that graders would have to repeat, and the UI would add a dependency without changing how the AI behaves. TF-IDF over 20 docs is easy to debug and reproduce. I also dropped `pandas` and `streamlit` from requirements since nothing imports them. I skipped the fine-tuning stretch for the same reason: I didn't want a "specialized vs baseline" comparison without a real model behind it.

**Where AI output was wrong:** a draft README line said "6 of the 13 real requests can't be answered without retrieval." When I checked against `eval/results.md` it was 5 of 10. Numbers in docs get checked against generated output now.

## Future improvements

- Handle negation and "but" clauses in the parser ("chill but not boring", "not sad")
- Detect contradictory profiles (high energy + high acousticness) and ask which matters more
- Artist mode: when the query names an artist, return their songs instead of using one as a seed
- Swap TF-IDF for embeddings once the knowledge base is big enough to need synonyms
- Use the 1-5 feedback log to tune guide hints instead of my guesses

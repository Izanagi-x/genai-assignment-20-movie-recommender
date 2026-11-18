# Assignment 20 — Movie recommendations

This is a content-based recommender using all 4,803 movies in the **TMDB 5000
Movie Dataset**: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata.
The original `tmdb_5000_movies.csv` is included. Credits are not needed.
Source download: Kaggle dataset API, 18 November 2026.

## Run

Use Python 3.14 and run these commands from this folder:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
streamlit run app.py
```

The supplied compressed artifact was built with the pinned requirements. Run
`python main.py` to regenerate it after changing data or dependencies. The UI
loads this artifact once using Streamlit's resource cache. The CSV is also
included so a missing artifact can be rebuilt locally. No API key is required.

## Tasks 1–4: data and vectors

`main.py` prints shape, columns, five rows, types, missing values and duplicate
IDs. Overview, parsed genre names and parsed keyword names are the text inputs.
Three missing overviews become empty strings. `clean_text` lowercases, removes
punctuation/special characters, collapses whitespace and removes scikit-learn's
English stopwords. Cleaned text is retained in `prepared_movies.csv`.

`TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)` creates sparse
vectors. A 5,000-feature cap limits vocabulary size; unigrams and bigrams retain
single themes and short phrases; `min_df=2` drops one-off noise. Float32 vectors
reduce memory use. The cosine matrix is calculated from these vectors, then
stored at float16 precision to keep the archive below the portal's 50 MB limit.
This rounding may tie scores that differ by less than roughly 0.001; tied scores
use original row order. `dataset_summary.json` records the actual matrix dimensions.
The full all-movie cosine matrix is stored in `recommendation_model.pkl.gz`
alongside the fitted vectorizer, vectors and prepared data.

Cosine similarity measures the angle between text vectors, so matching word
patterns matter more than description length. Scores here are content similarity
scores, not probabilities that a viewer will enjoy a movie.

## Tasks 5–6: recommendations and UI

`recommend(item_name, top_n=5)` finds the selected movie, excludes itself, sorts
all other movies by descending cosine score and returns the first N. Equal scores
use original row order for reproducible results. Missing names and invalid N
raise useful errors. `Avatar`, `Toy Story` and `Fight Club` are tested by main;
their actual results are saved in `example_recommendations.csv`.

Three duplicate titles are preserved and disambiguated in dropdown labels using
year and numeric ID. An ambiguous plain title asks for the full label. `app.py`
provides one dropdown and one button, then displays five titles, genres, scores
and overviews. The simple UI needs no login or external image service.

## Tasks 7–9: GitHub and Render

A local Git repository is initialized for this project. The GitHub repository is
https://github.com/Izanagi-x/genai-assignment-20-movie-recommender
(code push is pending Git authentication). Deployment uses the
included `render.yaml` or the equivalent dashboard settings:

- Build: `pip install -r requirements.txt`
- Start: `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true`
- Python version: 3.14.2
- Health path: `/_stcore/health`
- Free web service, with no paid resources requested.

**Current status: local implementation and tests; remote GitHub/Render steps
pending Git authentication and Render account access. No live deployment URL
has been produced yet.**
After a repository is pushed and Render is connected, deploy it, open the live
URL, and test at least the same three queries before marking Tasks 7–9 complete.

## Observations

Recommendations are explainable through overlapping descriptions and keywords,
with no collaborative filtering. Rare text, generic overviews and missing
metadata can weaken results. The default English stopword list removes negation;
these vectors do not understand plot logic or personal taste. A dense cosine
matrix grows quadratically, so a larger catalogue would need nearest-neighbour
retrieval instead. The small score differences should not be over-interpreted.
The examples provide a basic sanity chek rather than a relevance benchmark.

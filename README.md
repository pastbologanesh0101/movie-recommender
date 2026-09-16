# AI Movie Recommendation Engine

A small, self-contained **content-based movie recommender**. It represents
each movie as a TF-IDF vector built from its genres and short descriptive
tags, then ranks other movies by **cosine similarity** to recommend titles
with a similar "content fingerprint."

Everything runs fully offline: no external APIs, no dataset downloads at
runtime. The bundled dataset is bundled with the code.

## About the dataset

`data/movies.csv` is a **synthetic, hand-authored toy dataset** created for
this portfolio project (211 rows: title, genres, tags). The titles,
genres, and tags are made up for demonstration purposes -- they are not
scraped or downloaded from any real movie database or third-party service.
It includes a few deliberate "franchise" style series (e.g. `Star Voyager`,
`Star Voyager: New Frontier`, `Star Voyager: The Return`) with matching
genres/tags so the recommender has some obviously-correct nearest
neighbors to demonstrate and test against.

## How it works

1. `recommender.load_movies()` reads `data/movies.csv` into `Movie` records
   (`title`, `genres`, `tags`).
2. Each movie's `genres` + `tags` are combined into one text string.
3. `sklearn.feature_extraction.text.TfidfVectorizer` turns that corpus into
   TF-IDF vectors (a weighted bag-of-words representation).
4. `sklearn.metrics.pairwise.cosine_similarity` computes a full
   movie-by-movie similarity matrix.
5. `MovieRecommender.recommend(title, top_n=5)` looks up the movie's row,
   sorts every other movie by similarity score, and returns the top N.

## Requirements

- Python 3.10+
- `scikit-learn`, `numpy`, `scipy` (installed via `requirements.txt`)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python recommend.py "Star Voyager"
```

### Example output

```
Top 5 movies similar to "Star Voyager":

1. Star Voyager: New Frontier  (similarity: 1.0000)
2. Star Voyager: The Return  (similarity: 1.0000)
3. Broken Horizon  (similarity: 0.6774)
4. Hidden Legacy  (similarity: 0.6774)
5. Distant Horizon  (similarity: 0.6774)
```

Titles must match a row in `data/movies.csv` exactly (matching is
case-insensitive). If a title isn't found, the CLI prints an error and
exits with a non-zero status:

```bash
python recommend.py "A Movie That Does Not Exist"
# Error: Movie titled 'A Movie That Does Not Exist' was not found in the
# dataset. Titles must match exactly (case-insensitive).
```

## Project layout

```
movie-recommender/
├── data/
│   └── movies.csv          # bundled toy dataset (title, genres, tags)
├── recommender.py           # core TF-IDF + cosine similarity logic
├── recommend.py              # CLI entry point
├── tests/
│   └── test_recommend.py    # unit tests (pytest)
├── requirements.txt
├── .github/workflows/tests.yml  # CI: runs pytest on push/PR
├── LICENSE
└── README.md
```

## Running the tests

```bash
pip install -r requirements.txt
pytest -v
```

The test suite covers: dataset integrity, that recommendations come back
sorted by descending similarity, that a movie never recommends itself,
that known "franchise" pairs rank each other highly, case-insensitive
exact-title lookups, and that an unknown title raises a clear error.

Continuous integration (`.github/workflows/tests.yml`) runs this same test
suite on every push and pull request against Python 3.10, 3.11, and 3.12.

## License

MIT - see [LICENSE](LICENSE).

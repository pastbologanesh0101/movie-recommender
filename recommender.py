"""
Core logic for the content-based AI Movie Recommendation Engine.

Loads a small bundled dataset of movies (title, genres, tags), builds a
TF-IDF vector space over the combined "genres + tags" text for each movie,
and uses cosine similarity between those vectors to find movies that are
most similar in content/theme to a given title.

No network calls are made anywhere in this module -- everything works
offline against the bundled data/movies.csv file.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "movies.csv")


@dataclass(frozen=True)
class Movie:
    title: str
    genres: List[str]
    tags: List[str]

    @property
    def content(self) -> str:
        """Combined bag-of-words text used for vectorization."""
        genre_text = " ".join(g.replace("-", " ") for g in self.genres)
        tag_text = " ".join(t.replace("-", " ") for t in self.tags)
        return f"{genre_text} {tag_text}".strip()


def load_movies(path: str = DEFAULT_DATA_PATH) -> List[Movie]:
    """Load the bundled CSV dataset into a list of Movie records."""
    movies: List[Movie] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row["title"].strip()
            genres = [g.strip() for g in row["genres"].split("|") if g.strip()]
            tags = [t.strip() for t in row["tags"].split() if t.strip()]
            movies.append(Movie(title=title, genres=genres, tags=tags))
    if not movies:
        raise ValueError(f"No movies loaded from {path!r}; dataset appears to be empty.")
    return movies


class MovieRecommender:
    """Content-based recommender using TF-IDF + cosine similarity."""

    def __init__(self, movies: List[Movie]):
        if not movies:
            raise ValueError("MovieRecommender requires a non-empty list of movies.")
        self.movies = movies
        self._title_index = {m.title.lower(): i for i, m in enumerate(movies)}

        corpus = [m.content for m in movies]
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    @classmethod
    def from_csv(cls, path: str = DEFAULT_DATA_PATH) -> "MovieRecommender":
        return cls(load_movies(path))

    def _index_for_title(self, title: str) -> int:
        key = title.strip().lower()
        if key not in self._title_index:
            raise ValueError(
                f"Movie titled {title!r} was not found in the dataset. "
                "Titles must match exactly (case-insensitive)."
            )
        return self._title_index[key]

    def recommend(self, title: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """Return up to `top_n` (title, similarity_score) pairs most similar
        to the given movie title, sorted by descending similarity. The
        queried movie itself is excluded from the results."""
        idx = self._index_for_title(title)
        scores = list(enumerate(self.similarity_matrix[idx]))
        # Exclude the movie itself.
        scores = [(i, s) for i, s in scores if i != idx]
        scores.sort(key=lambda pair: pair[1], reverse=True)
        top = scores[:top_n]
        return [(self.movies[i].title, float(round(s, 4))) for i, s in top]

    def known_titles(self) -> List[str]:
        return [m.title for m in self.movies]

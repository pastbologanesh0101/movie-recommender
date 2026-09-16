"""
Unit tests for the AI Movie Recommendation Engine.

Run with:
    pytest
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender import MovieRecommender, load_movies, DEFAULT_DATA_PATH  # noqa: E402


@pytest.fixture(scope="module")
def recommender():
    return MovieRecommender.from_csv()


def test_dataset_loads_and_has_enough_rows():
    movies = load_movies(DEFAULT_DATA_PATH)
    assert len(movies) >= 150, "Bundled dataset should have at least 150 movies."
    # Every row should have a title and at least one genre.
    for movie in movies:
        assert movie.title
        assert len(movie.genres) >= 1


def test_recommend_returns_five_results_sorted_by_similarity(recommender):
    results = recommender.recommend("Star Voyager", top_n=5)
    assert len(results) == 5
    scores = [score for _, score in results]
    # Results must be sorted in descending order of similarity.
    assert scores == sorted(scores, reverse=True)


def test_recommend_excludes_the_queried_movie_itself(recommender):
    results = recommender.recommend("Star Voyager", top_n=5)
    titles = [title for title, _ in results]
    assert "Star Voyager" not in titles


def test_known_similar_pair_ranks_highly(recommender):
    # "Star Voyager" and its direct sequels share identical genres/tags,
    # so they should be near-perfect matches for one another.
    results = recommender.recommend("Star Voyager", top_n=5)
    titles = [title for title, _ in results]
    assert "Star Voyager: New Frontier" in titles
    assert "Star Voyager: The Return" in titles

    top_title, top_score = results[0]
    assert top_title in ("Star Voyager: New Frontier", "Star Voyager: The Return")
    assert top_score > 0.9


def test_recommend_is_case_insensitive_on_exact_title(recommender):
    results_lower = recommender.recommend("star voyager", top_n=3)
    results_upper = recommender.recommend("STAR VOYAGER", top_n=3)
    results_mixed = recommender.recommend("Star Voyager", top_n=3)
    assert results_lower == results_upper == results_mixed


def test_recommend_unknown_title_raises_value_error(recommender):
    with pytest.raises(ValueError):
        recommender.recommend("This Movie Definitely Does Not Exist 42")


def test_recommend_respects_top_n(recommender):
    results = recommender.recommend("Haunted Hollow", top_n=3)
    assert len(results) == 3


def test_recommender_rejects_empty_movie_list():
    with pytest.raises(ValueError):
        MovieRecommender([])

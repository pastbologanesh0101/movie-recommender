#!/usr/bin/env python3
"""
CLI entry point for the AI Movie Recommendation Engine.

Usage:
    python recommend.py "Movie Title"

Prints the top 5 movies most similar (by TF-IDF + cosine similarity over
genres and tags) to the given title, along with a similarity score.
"""
import sys

from recommender import MovieRecommender


def main(argv: list) -> int:
    if len(argv) != 2:
        print('Usage: python recommend.py "Movie Title"', file=sys.stderr)
        return 1

    title = argv[1]
    recommender = MovieRecommender.from_csv()

    try:
        results = recommender.recommend(title, top_n=5)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f'Top 5 movies similar to "{title}":\n')
    for rank, (rec_title, score) in enumerate(results, start=1):
        print(f"{rank}. {rec_title}  (similarity: {score:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

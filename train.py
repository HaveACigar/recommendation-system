import io
import os
import zipfile
from urllib.request import urlopen

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
TOP_K = 10


def download_dataset():
    os.makedirs("data", exist_ok=True)
    with urlopen(DATA_URL) as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as zf:
            zf.extractall("data")


def weighted_rating(frame, m_quantile=0.9):
    stats = frame.groupby("movieId").agg(avg_rating=("rating", "mean"), rating_count=("rating", "count")).reset_index()
    c = stats["avg_rating"].mean()
    m = stats["rating_count"].quantile(m_quantile)
    qualified = stats[stats["rating_count"] >= m].copy()
    qualified["weighted_score"] = (
        qualified["rating_count"] / (qualified["rating_count"] + m) * qualified["avg_rating"]
        + m / (qualified["rating_count"] + m) * c
    )
    return qualified.sort_values(["weighted_score", "rating_count"], ascending=False)


def build_holdout(ratings):
    ratings = ratings.sort_values(["userId", "timestamp"])
    test_idx = ratings.groupby("userId").tail(1).index
    test = ratings.loc[test_idx].copy()
    train = ratings.drop(index=test_idx).copy()
    return train, test


def precision_at_k(train_matrix, model, test, movie_idx_map, user_idx_map, k=10):
    user_factors = model.transform(train_matrix)
    item_factors = model.components_.T
    predictions = user_factors @ item_factors.T

    hits = 0
    total = 0
    for _, row in test.iterrows():
        user = row["userId"]
        movie = row["movieId"]
        if user not in user_idx_map or movie not in movie_idx_map:
            continue
        user_idx = user_idx_map[user]
        movie_idx = movie_idx_map[movie]
        seen = train_matrix[user_idx].toarray().ravel() > 0
        scores = predictions[user_idx].copy()
        scores[seen] = -np.inf
        top_items = np.argpartition(scores, -k)[-k:]
        if movie_idx in top_items:
            hits += 1
        total += 1
    return hits / total if total else 0.0


def main():
    download_dataset()
    base = "data/ml-latest-small"
    ratings = pd.read_csv(f"{base}/ratings.csv")
    movies = pd.read_csv(f"{base}/movies.csv")

    popularity = weighted_rating(ratings).merge(movies, on="movieId", how="left")

    content_df = movies.copy()
    content_df["genres_clean"] = content_df["genres"].str.replace("|", " ", regex=False)
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(content_df["title"] + " " + content_df["genres_clean"])
    content_sim = cosine_similarity(tfidf, dense_output=False)

    train, test = build_holdout(ratings)
    user_ids = np.sort(train["userId"].unique())
    movie_ids = np.sort(train["movieId"].unique())
    user_idx_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
    movie_idx_map = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

    rows = train["userId"].map(user_idx_map).values
    cols = train["movieId"].map(movie_idx_map).values
    values = train["rating"].values
    user_item = sparse.csr_matrix((values, (rows, cols)), shape=(len(user_ids), len(movie_ids)))

    nmf = NMF(n_components=20, init="nndsvda", random_state=42, max_iter=300)
    nmf.fit(user_item)
    user_factors = nmf.transform(user_item)
    item_factors = nmf.components_.T
    reconstructed = user_factors @ item_factors.T

    prec_k = precision_at_k(user_item, nmf, test, movie_idx_map, user_idx_map, k=TOP_K)

    item_factor_norm = item_factors / np.clip(np.linalg.norm(item_factors, axis=1, keepdims=True), 1e-8, None)
    collaborative_sim = item_factor_norm @ item_factor_norm.T

    movie_lookup = movies.set_index("movieId")
    top_popular = popularity[["movieId", "title", "genres", "weighted_score", "rating_count"]].head(25)

    os.makedirs("models", exist_ok=True)
    joblib.dump(
        {
            "ratings": ratings,
            "movies": movies,
            "top_popular": top_popular,
            "popularity": popularity,
            "content_sim": content_sim,
            "collaborative_sim": collaborative_sim,
            "movie_lookup": movie_lookup,
            "movie_ids": movie_ids,
            "movie_idx_map": movie_idx_map,
            "user_ids": user_ids,
            "user_idx_map": user_idx_map,
            "user_item": user_item,
            "nmf_user_factors": user_factors,
            "nmf_item_factors": item_factors,
            "reconstructed": reconstructed,
            "precision_at_10": prec_k,
            "top_k": TOP_K,
        },
        "models/artifacts.pkl",
        compress=3,
    )


if __name__ == "__main__":
    main()

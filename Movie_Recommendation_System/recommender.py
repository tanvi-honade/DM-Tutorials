from pathlib import Path
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- File Paths ----------------
BASE_DIR = Path(__file__).resolve().parent

movies = pd.read_csv(BASE_DIR / "dataset" / "movies.csv")
ratings = pd.read_csv(BASE_DIR / "dataset" / "ratings.csv")

# ---------------- User-Movie Matrix ----------------
movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

# ---------------- Cosine Similarity ----------------
similarity = cosine_similarity(movie_matrix.T)

similarity_df = pd.DataFrame(
    similarity,
    index=movie_matrix.columns,
    columns=movie_matrix.columns
)

# ---------------- Recommendation Function ----------------
def recommend(movie_name):

    # Find selected movie
    movie = movies[movies["title"] == movie_name]

    if movie.empty:
        return pd.DataFrame()

    movie_id = movie.iloc[0]["movieId"]

    if movie_id not in similarity_df.columns:
        return pd.DataFrame()

    # Similarity scores
    scores = similarity_df[movie_id].sort_values(ascending=False)

    recommendations = []

    # Skip first movie (itself)
    for mid in scores.index[1:7]:

        movie_info = movies[movies["movieId"] == mid]

        if movie_info.empty:
            continue

        movie_info = movie_info.iloc[0]

        recommendations.append({
            "Movie": movie_info["title"],
            "Genre": movie_info["genre"] if "genre" in movie_info.index else "N/A",
            "Similarity": round(float(scores[mid]), 2)
        })

    return pd.DataFrame(recommendations)
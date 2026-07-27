from pathlib import Path
import streamlit as st
from recommender import recommend, movies, ratings

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# ---------------- CSS ----------------
BASE_DIR = Path(__file__).resolve().parent
css_file = BASE_DIR / "style.css"

if css_file.exists():
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

# ---------------- Title ----------------
st.title("🎬 Movie Recommendation System")
st.write("Find similar movies using Collaborative Filtering (Cosine Similarity)")

# ---------------- Metrics ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🎬 Movies", len(movies))

with col2:
    st.metric("👤 Users", ratings["userId"].nunique())

with col3:
    st.metric("⭐ Ratings", len(ratings))

st.divider()

# ---------------- Movie Selection ----------------
movie = st.selectbox(
    "Select a Movie",
    sorted(movies["title"].unique())
)

# ---------------- Recommendation ----------------
if st.button("🎯 Recommend Movies"):

    result = recommend(movie)

    if result.empty:
        st.warning("No recommendations found.")
    else:
        st.success("Top Recommended Movies")
        st.dataframe(
            result,
            width="stretch",
            hide_index=True
        )

st.divider()

# ---------------- Dataset ----------------
st.subheader("📂 Movie Dataset")

st.dataframe(
    movies,
    width="stretch",
    hide_index=True
)
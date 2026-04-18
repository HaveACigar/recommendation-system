import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Recommendation System", page_icon="🎥", layout="wide", initial_sidebar_state="collapsed")
TEMPLATE = "plotly_dark"


@st.cache_resource
def load_artifacts():
    return joblib.load("models/artifacts.pkl")


def get_content_recommendations(movie_id, arts, n=10):
    idx = arts["movie_idx_map"].get(movie_id)
    if idx is None:
        movie_position = arts["movies"].index[arts["movies"]["movieId"] == movie_id]
        if len(movie_position) == 0:
            return pd.DataFrame()
        idx = movie_position[0]
        scores = np.asarray(arts["content_sim"][idx].todense()).ravel()
        order = np.argsort(scores)[::-1]
        rec_ids = arts["movies"].iloc[order]["movieId"].tolist()
    else:
        movie_position = arts["movies"].index[arts["movies"]["movieId"] == movie_id][0]
        scores = np.asarray(arts["content_sim"][movie_position].todense()).ravel()
        order = np.argsort(scores)[::-1]
        rec_ids = arts["movies"].iloc[order]["movieId"].tolist()
    rec_ids = [mid for mid in rec_ids if mid != movie_id][:n]
    recs = arts["movies"].set_index("movieId").loc[rec_ids].reset_index()
    return recs


def get_personalized_recommendations(user_id, arts, n=10):
    if user_id not in arts["user_idx_map"]:
        return pd.DataFrame()
    user_idx = arts["user_idx_map"][user_id]
    scores = arts["reconstructed"][user_idx].copy()
    seen = arts["user_item"][user_idx].toarray().ravel() > 0
    scores[seen] = -np.inf
    top_idx = np.argpartition(scores, -n)[-n:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    movie_ids = arts["movie_ids"][top_idx]
    recs = arts["movie_lookup"].loc[movie_ids].reset_index()
    recs["predicted_score"] = scores[top_idx]
    return recs


def main():
    arts = load_artifacts()
    st.title("🎥 Recommendation System")
    st.markdown(
        "MovieLens recommendation demo combining **popularity ranking**, **content-based similarity**, and "
        "**collaborative filtering** via low-rank matrix factorization."
    )

    tabs = st.tabs(["Overview", "Popularity", "Content-Based", "Personalized"])

    with tabs[0]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Users", f"{arts['ratings']['userId'].nunique():,}")
        c2.metric("Movies", f"{arts['movies']['movieId'].nunique():,}")
        c3.metric("Ratings", f"{len(arts['ratings']):,}")
        c4.metric("Precision@10", f"{arts['precision_at_10']:.3f}")

        genre_counts = (
            arts["movies"]["genres"].str.split("|").explode().value_counts().head(12).reset_index()
        )
        genre_counts.columns = ["genre", "count"]
        fig = px.bar(genre_counts, x="genre", y="count", color="count", color_continuous_scale="Blues", title="Top Genres", template=TEMPLATE)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.subheader("Bayesian Popularity Ranking")
        st.dataframe(arts["top_popular"], use_container_width=True, hide_index=True)

    with tabs[2]:
        st.subheader("Content-Based Similarity")
        movie_options = arts["movies"].sort_values("title")["title"].tolist()
        selected_title = st.selectbox("Choose a movie", movie_options)
        movie_id = int(arts["movies"].loc[arts["movies"]["title"] == selected_title, "movieId"].iloc[0])
        recs = get_content_recommendations(movie_id, arts, n=10)
        st.dataframe(recs[["title", "genres"]], use_container_width=True, hide_index=True)

    with tabs[3]:
        st.subheader("Personalized Recommendations")
        user_id = st.selectbox("Choose a user", sorted(arts["user_ids"].tolist()))
        recs = get_personalized_recommendations(user_id, arts, n=10)
        st.dataframe(recs[["title", "genres", "predicted_score"]], use_container_width=True, hide_index=True)

        liked = arts["ratings"].query("userId == @user_id and rating >= 4").merge(arts["movies"], on="movieId").sort_values("rating", ascending=False).head(10)
        st.markdown("**User's highly rated movies**")
        st.dataframe(liked[["title", "genres", "rating"]], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

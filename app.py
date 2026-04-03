# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# st.write("🚀 App is starting...")
# # ------------------ APP START DEBUG ------------------
# st.set_page_config(page_title="Movie Recommender", layout="wide")
# st.title("🎬 Movie Recommendation System")

# # ------------------ LOAD DATA ------------------
# @st.cache_data
# def load_data():
#     movies = pd.read_csv("movies.csv", encoding="latin-1")
#     ratings = pd.read_csv("ratings.csv")

#     # Save original genres
#     movies['genres_original'] = movies['genres']

#     # Average rating
#     avg_ratings = ratings.groupby("movieId")["rating"].mean().reset_index()

#     # Merge
#     movies = pd.merge(movies, avg_ratings, on="movieId")

#     # Clean genres for vectorization
#     movies['genres_clean'] = movies['genres'].str.replace('|', ' ', regex=False)

#     return movies

# movies = load_data()

# # ------------------ GENRE LIST ------------------
# all_genres = sorted({
#     genre
#     for g in movies['genres_original']
#     for genre in g.split('|')
# })

# # ------------------ VECTORIZATION ------------------
# @st.cache_resource
# def compute_similarity(data):
#     cv = CountVectorizer()
#     genre_matrix = cv.fit_transform(data['genres_clean'])
#     return cv, genre_matrix

# cv, genre_matrix = compute_similarity(movies)

# # ------------------ RECOMMEND FUNCTION ------------------
# def recommend_movies(user_genre, top_n=10, low_n=2):
#     user_vec = cv.transform([user_genre])
#     scores = cosine_similarity(user_vec, genre_matrix).flatten()

#     data = movies.copy()
#     data['score'] = scores

#     filtered = data[data['rating'].notna()]
#     filtered = filtered[filtered['score'] > 0]

#     # Top movies
#     top_movies = filtered.sort_values(
#         by=['score', 'rating'], ascending=False
#     ).head(top_n)

#     # Low-rated movies
#     poor_movies = filtered.sort_values(
#         by=['score', 'rating'], ascending=[False, True]
#     ).head(low_n)

#     return top_movies, poor_movies

# # ------------------ SIDEBAR UI ------------------
# st.sidebar.header("🎭 Select Genres")

# selected_genres = st.sidebar.multiselect(
#     "Choose genres:",
#     all_genres
# )

# # ------------------ BUTTON ------------------
# if st.sidebar.button("Recommend Movies"):

#     if not selected_genres:
#         st.warning("⚠️ Please select at least one genre")
#     else:
#         user_genre = " ".join(selected_genres)

#         top_movies, poor_movies = recommend_movies(user_genre)

#         # ------------------ TOP MOVIES ------------------
#         st.subheader("🎬 Top Recommended Movies")

#         for _, row in top_movies.iterrows():
#             st.markdown(f"### 🎥 {row['title']}")
#             st.write(f"⭐ Rating: {round(row['rating'], 2)}")

#             genres = row['genres_original'].split('|')
#             st.write("🎭 Genres:", ", ".join(genres))

#             st.markdown("---")

#         # ------------------ LOW RATED ------------------
#         st.subheader("⚠️ Low Rated But Similar Movies")

#         for _, row in poor_movies.iterrows():
#             st.markdown(f"### 🎥 {row['title']}")
#             st.write(f"⭐ Rating: {round(row['rating'], 2)}")

#             genres = row['genres_original'].split('|')
#             st.write("🎭 Genres:", ", ".join(genres))

#             st.markdown("---") 


import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.write("🚀 App is starting...")
# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv", encoding="latin-1")
    ratings = pd.read_csv("ratings.csv")

    # Save original genres
    movies['genres_original'] = movies['genres']

    # Extract year from title
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
    movies['year'] = pd.to_numeric(movies['year'], errors='coerce')

    # Average rating
    avg_ratings = ratings.groupby("movieId")["rating"].mean().reset_index()

    # Merge
    movies = pd.merge(movies, avg_ratings, on="movieId")

    # Clean genres for vectorization
    movies['genres_clean'] = movies['genres'].str.replace('|', ' ', regex=False)

    return movies

movies = load_data()

# ------------------ GENRE LIST ------------------
all_genres = sorted({
    genre
    for g in movies['genres_original']
    for genre in g.split('|')
})

# ------------------ YEAR RANGE ------------------
min_year = int(movies['year'].min())
max_year = int(movies['year'].max())

# ------------------ VECTORIZATION ------------------
@st.cache_resource
def compute_similarity(data):
    cv = CountVectorizer()
    genre_matrix = cv.fit_transform(data['genres_clean'])
    return cv, genre_matrix

cv, genre_matrix = compute_similarity(movies)

# ------------------ RECOMMEND FUNCTION ------------------
def recommend_movies(user_genre, year_range, top_n=10, low_n=2):
    user_vec = cv.transform([user_genre])
    scores = cosine_similarity(user_vec, genre_matrix).flatten()

    data = movies.copy()
    data['score'] = scores

    # Apply year filter
    data = data[
        (data['year'] >= year_range[0]) &
        (data['year'] <= year_range[1])
    ]

    filtered = data[data['rating'].notna()]
    filtered = filtered[filtered['score'] > 0]

    # Top movies
    top_movies = filtered.sort_values(
        by=['score', 'rating'], ascending=False
    ).head(top_n)

    # Low-rated movies
    poor_movies = filtered.sort_values(
        by=['score', 'rating'], ascending=[False, True]
    ).head(low_n)

    return top_movies, poor_movies

# ------------------ SIDEBAR ------------------
st.sidebar.header("🎭 Select Genres")

selected_genres = st.sidebar.multiselect(
    "Choose genres:",
    all_genres
)

st.sidebar.header("📅 Select Year Range")

year_range = st.sidebar.slider(
    "Select year range:",
    min_value=min_year,
    max_value=max_year,
    value=(2000, 2020)
)

# ------------------ BUTTON ------------------
if st.sidebar.button("Recommend Movies"):

    if not selected_genres:
        st.warning("⚠️ Please select at least one genre")
    else:
        user_genre = " ".join(selected_genres)

        top_movies, poor_movies = recommend_movies(user_genre, year_range)

        # ------------------ TOP MOVIES ------------------
        st.subheader("🎬 Top Recommended Movies")

        for _, row in top_movies.iterrows():
            st.markdown(f"### 🎥 {row['title']} ({int(row['year'])})")
            st.write(f"⭐ Rating: {round(row['rating'], 2)}")

            genres = row['genres_original'].split('|')
            st.write("🎭 Genres:", ", ".join(genres))

            st.markdown("---")

        # ------------------ LOW RATED ------------------
        st.subheader("⚠️ Low Rated But Similar Movies")

        for _, row in poor_movies.iterrows():
            st.markdown(f"### 🎥 {row['title']} ({int(row['year'])})")
            st.write(f"⭐ Rating: {round(row['rating'], 2)}")

            genres = row['genres_original'].split('|')
            st.write("🎭 Genres:", ", ".join(genres))

            st.markdown("---")
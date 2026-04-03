import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ LOAD DATA ------------------
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Save original genres
movies['genres_original'] = movies['genres']

# Average rating
avg_ratings = ratings.groupby("movieId")["rating"].mean().reset_index()
movies = pd.merge(movies, avg_ratings, on="movieId")

# Clean for vectorization
movies['genres'] = movies['genres'].str.replace('|', ' ')

# ------------------ GENRE LIST ------------------
all_genres = set()
for g in movies['genres_original']:
    for genre in g.split('|'):
        all_genres.add(genre)

all_genres = sorted(all_genres)

# ------------------ VECTORIZATION ------------------
cv = CountVectorizer()
genre_matrix = cv.fit_transform(movies['genres'])

# ------------------ RECOMMEND FUNCTION ------------------
def recommend_movies(user_genre, top_n=10, low_n=2):
    user_vec = cv.transform([user_genre])
    scores = cosine_similarity(user_vec, genre_matrix).flatten()
    
    movies['score'] = scores
    filtered = movies[movies['rating'].notna()]
    
    top_movies = filtered[filtered['score'] > 0].sort_values(
        by=['score', 'rating'], ascending=False
    ).head(top_n)
    
    poor_movies = filtered[filtered['score'] > 0].sort_values(
        by=['score', 'rating'], ascending=[False, True]
    ).head(low_n)
    
    return top_movies, poor_movies

# ------------------ UI ------------------
st.title("🎬 Movie Recommendation System")

st.sidebar.header("🎭 Select Genres")

selected_genres = st.sidebar.multiselect(
    "Choose genres:",
    all_genres
)

user_genre = " ".join(selected_genres)

# ------------------ BUTTON ------------------
if st.sidebar.button("Recommend Movies"):
    
    if not user_genre:
        st.warning("⚠️ Please select at least one genre")
    else:
        top_movies, poor_movies = recommend_movies(user_genre)
        
        st.subheader("🎬 Top Recommended Movies")
        
        for _, row in top_movies.iterrows():
            st.markdown(f"### 🎥 {row['title']}")
            st.write(f"⭐ Rating: {round(row['rating'],2)}")
            
            genres = row['genres_original'].split('|')
            st.write("🎭 Genres:", ", ".join(genres))
            st.markdown("---")
        
        st.subheader("⚠️ Low Rated But Similar Movies")
        
        for _, row in poor_movies.iterrows():
            st.markdown(f"### 🎥 {row['title']}")
            st.write(f"⭐ Rating: {round(row['rating'],2)}")
            
            genres = row['genres_original'].split('|')
            st.write("🎭 Genres:", ", ".join(genres))
            st.markdown("---")
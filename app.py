import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from numpy.linalg import norm
import requests

# --- PAGE CONFIG MUST BE FIRST ---
st.set_page_config(
                    page_title="Movie Recommendation System - Complete",
                    page_icon="🎬",
                    layout="wide",
                    initial_sidebar_state="expanded"
                    )

# =============================================================================
# SHARED FUNCTIONS
# =============================================================================

def setup_tmdb_api():
    """Configura l'API TMDB se disponibile"""
    try:
        api_key = st.secrets["API_KEY"]
        # Il token è già completo con "Bearer", usiamolo direttamente
        headers = {
                    "accept": "application/json",
                    "Authorization": api_key
                    }
        
        BASE_URL = "https://api.themoviedb.org/3"
        test_url = f"{BASE_URL}/search/movie?query=avatar"
        test_response = requests.get(test_url, headers=headers)
        
        if test_response.status_code == 200:
            return True, headers
        else:
            st.sidebar.error(f"⚠️ TMDB API Error: {test_response.status_code}")
            st.sidebar.error(f"Response: {test_response.text[:200]}...")
    except Exception as e:
        st.sidebar.error(f"⚠️ TMDB API Connection Error: {str(e)}")

def search_movie_tmdb(query, tmdb_available, headers):
    """Cerca un film usando l'API TMDB"""
    if not tmdb_available or not headers:
        return []
    
    BASE_URL = "https://api.themoviedb.org/3"
    url = f"{BASE_URL}/search/movie?query={query}"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json().get("results", [])
        else:
            st.error(f"TMDB API Error: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error searching TMDB: {str(e)}")
        return []


def apply_custom_css():
    """Apply custom CSS styling."""
    st.markdown("""
    <style>
    :root { --brand: #1e40af; }

    .block-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        box-sizing: border-box;
    }

    .sidebar .sidebar-content { background-color: var(--brand); }
    .sidebar .sidebar-content, .sidebar .sidebar-content * { color: #ffffff !important; }

    .hero{
        background-color: var(--brand);
        color: #ffffff;
        display:flex;
        justify-content:center;
        align-items:center;
        margin: 28px auto 18px;
        padding: 18px 26px;
        min-height:76px;
        border-radius: 40px;
        box-shadow:0 10px 30px rgba(30,64,175,0.12);
        text-align:center;
        width: min(620px, 80%);
        box-sizing: border-box;
    }
    .hero h1 { margin:0; font-size:20px; }
    .hero p { margin:6px 0 0 0; font-size:14px; color: #ffffff; }

    .rec-header {
        background-color: var(--brand);
        color: #ffffff;
        padding: 12px 20px;
        margin: 20px auto;
        border-radius: 20px;
        text-align: center;
        width: min(600px, 80%);
        box-shadow: 0 4px 12px rgba(30,64,175,0.15);
    }
    .rec-header h2 { margin: 0; font-size: 18px; }

    .movie-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
        background-color: #fafafa;
        text-align: center;
        transition: transform 0.2s;
    }
    .movie-card:hover { transform: translateY(-2px); }

    .movie-rating {
        background-color: var(--brand);
        color: #ffffff;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 12px;
        margin: 8px auto 4px;
        display: inline-block;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px;
        color: var(--brand);
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--brand);
        color: white;
    }

    div[data-testid="stSelectbox"] label {
        color: #ffffff !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# SENTENCE TRANSFORMER APP
# =============================================================================

@st.cache_resource
def load_sentence_transformer_model():
    """Carica il modello SentenceTransformer"""
    try:
        model_path = os.path.join("Sentence-Transformer", "pickle_model", "sentence_transformer_model.pkl")
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading SentenceTransformer model: {e}")

@st.cache_data
def load_sentence_transformer_data():
    """Carica il dataset CSV e gli embeddings per SentenceTransformer"""
    try:
        movies_path = os.path.join("Sentence-Transformer", "imdb_movies.csv")
        embeddings_path = os.path.join("Sentence-Transformer", "pickle_model", "movie_embeddings.pkl")
        
        # Carica il dataset CSV con pandas
        movies = pd.read_csv(movies_path)
        movies = movies.dropna(subset=["names", "overview"])
        
        # Carica gli embeddings con joblib
        embeddings = joblib.load(embeddings_path)
        
        return movies, embeddings
    except Exception as e:
        st.error(f"Error loading SentenceTransformer data: {e}")

def recommend_sentence_transformer(movies, embeddings, movie_name, top_k=15):
    """Raccomandazioni basate su embeddings precomputati"""
    idx = movies[movies["names"].str.lower() == movie_name.lower()].index
    if len(idx) == 0:
        return None
    idx = idx[0]

    query_embedding = embeddings[idx]
    
    # Convert to numpy if it's a tensor
    if hasattr(query_embedding, 'cpu'):
        query_embedding = query_embedding.cpu().numpy()
    if hasattr(embeddings, 'cpu'):
        embeddings_np = embeddings.cpu().numpy()
    else:
        embeddings_np = embeddings
    # cosine similarity (NumPy)
    scores = np.dot(embeddings_np, query_embedding) / (norm(embeddings_np, axis=1) * norm(query_embedding))
    # top results (skip the same movie)
    top_indices = np.argsort(scores)[::-1][1:top_k+1]
    return movies.iloc[top_indices]["names"].tolist()


def recommend_with_model_st(model, movies, embeddings, query_text, top_k=15):
    """Raccomandazioni usando il modello per nuove query testuali"""
    # Calcola embedding per la query usando il modello
    query_embedding = model.encode([query_text], convert_to_tensor=True)[0]
    
    # Convert to numpy
    if hasattr(query_embedding, 'cpu'):
        query_embedding = query_embedding.cpu().numpy()
    if hasattr(embeddings, 'cpu'):
        embeddings_np = embeddings.cpu().numpy()
    else:
        embeddings_np = embeddings
    # cosine similarity
    scores = np.dot(embeddings_np, query_embedding) / (norm(embeddings_np, axis=1) * norm(query_embedding))
    # top results
    top_indices = np.argsort(scores)[::-1][:top_k]

    return movies.iloc[top_indices]["names"].tolist()


# =============================================================================
# TFIDF-KNN APP
# =============================================================================

@st.cache_resource
def load_tfidf_model():
    """Load the trained KNN model and TF-IDF vectorizer."""
    try:
        # Load saved models and data
        model_path = os.path.join("TFIDF-KNN", "pickle_model", "knn_model.pkl")
        vectorizer_path = os.path.join("TFIDF-KNN", "pickle_model", "tfidf_vectorizer.pkl")
        
        knn_model = joblib.load(model_path)
        tfidf_vectorizer = joblib.load(vectorizer_path)
        
        # Load movie dataset (include overview column as it's used for recommendations)
        movies_df = pd.read_csv(os.path.join("TFIDF-KNN", "imdb_movies.csv"))
        movies_df = movies_df.dropna(subset=["names", "overview"])
        
        return knn_model, tfidf_vectorizer, movies_df
    except Exception as e:
        st.error(f"Error loading TFIDF model: {e}")

def recommend_tfidf(movie_name, knn_model, tfidf_vectorizer, movies_df, n_recommendations=10):
    """Generate movie recommendations using TF-IDF + KNN."""
    try:
        # Check if movie exists in dataset
        if movie_name not in movies_df["names"].values:
            return []
        
        # Get movie index
        movie_index = movies_df[movies_df["names"] == movie_name].index[0]
        
        # Get the movie's TF-IDF vector using the overview column
        movie_vector = tfidf_vectorizer.transform([movies_df.iloc[movie_index]["overview"]])
        
        # Find similar movies using KNN
        distances, indices = knn_model.kneighbors(movie_vector, n_neighbors=n_recommendations + 1)
        
        # Get recommended movie names (excluding the input movie itself)
        recommended_movies = []
        for idx in indices[0][1:]:  # Skip the first one as it's the input movie itself
            recommended_movies.append(movies_df.iloc[idx]["names"])
        
        return recommended_movies
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return []


# =============================================================================
# TMDB API APP
# =============================================================================

@st.cache_data
def load_tmdb_movies():
    """Load movie dataset for TMDB app."""
    try:
        return pd.read_csv(os.path.join("TMDB-API", "imdb_movies.csv"))
    except Exception as e:
        st.error(f"Error loading TMDB movies: {e}")
        return pd.DataFrame()


def get_recommendations_tmdb(movie_id, tmdb_available, headers):
    """Ottieni raccomandazioni da TMDB per un film specifico"""
    if not tmdb_available or not headers:
        return []
    
    BASE_URL = "https://api.themoviedb.org/3"
    url = f"{BASE_URL}/movie/{movie_id}/recommendations?language=en-US"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json().get("results", [])
        else:
            st.error(f"TMDB API Error: {response.status_code}")
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")


# =============================================================================
# SHARED UI COMPONENTS
# =============================================================================

def display_movie_card_with_rating(movie_title, tmdb_available, headers, image_url, key, tab_key):
    """Display a movie card with rating stars."""
    st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
    
    search_results = search_movie_tmdb(movie_title, tmdb_available, headers)
    if search_results and search_results[0].get("poster_path"):
        st.image(image_url + search_results[0]["poster_path"], use_container_width=True)
        st.markdown(f"<div class='movie-rating'>⭐ {search_results[0].get('vote_average','N/A')}/10</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='height:300px; background: linear-gradient(135deg,#ddd,#f8f8f8); "
            "display:flex; align-items:center; justify-content:center; border-radius:8px; color:#666;'>"
            "🎬 No Image</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='movie-rating'>⭐ N/A</div>", unsafe_allow_html=True)
    
    if st.button(movie_title, key=key):
        st.session_state[f"active_movie_{tab_key}"] = movie_title
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application function."""
    # Apply custom CSS
    apply_custom_css()
    
    # Setup TMDB API
    tmdb_available, headers = setup_tmdb_api()
    IMAGE_URL = "https://image.tmdb.org/t/p/w500"
    
    # Main title
    st.markdown(
        "<div class='hero'><div><h1>🎬 Complete Movie Recommendation System</h1>"
        "<p>Explore three different AI approaches to movie recommendations!</p></div></div>",
        unsafe_allow_html=True,
    )
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "🧠 Sentence Transformer", 
        "📊 TF-IDF + KNN", 
        "🎭 TMDB API"
    ])
    
    # Initialize session state for current tab
    if "current_tab" not in st.session_state:
        st.session_state.current_tab = "st"
    
    # Determine which tab is active by checking which tab content is being displayed
    # We'll use a different approach - manage sidebar based on tab selection
    
    # Tab 1: Sentence Transformer
    with tab1:
        st.session_state.current_tab = "st"
        
        st.markdown("### 🧠 Semantic Movie Recommendations")
        st.info("Using pre-trained Sentence Transformers for semantic similarity between movie descriptions.")
        
        # Load models and data
        model = load_sentence_transformer_model()
        movies, embeddings = load_sentence_transformer_data()
        
        if model is None or movies is None or embeddings is None:
            st.error("Failed to load Sentence Transformer components.")
            return
        
        # Initialize session state for this tab
        if "active_movie_st" not in st.session_state:
            st.session_state["active_movie_st"] = ""
        
        # Movie search in main area
        st.subheader("🔍 Search for a Movie")
        search_query = st.text_input("Type a movie name:", key="search_st")
        
        # Filter movies based on search or show all
        if search_query:
            filtered_movies = movies[movies["names"].str.contains(search_query, case=False, na=False)]["names"].tolist()
        else:
            filtered_movies = movies["names"].dropna().head(50).tolist()  # Show first 50 movies by default
        
        if filtered_movies:
            selected_movie = st.selectbox(
                "Select a movie:",
                options=[""] + filtered_movies,
                format_func=lambda x: "Choose a movie..." if x == "" else x,
                key="selectbox_st"
            )
            if selected_movie and selected_movie != "":
                st.session_state["active_movie_st"] = selected_movie
        else:
            st.warning("No movies found matching your search.")
        
        active_movie = st.session_state.get("active_movie_st", "")
        
        # Show currently selected movie
        if active_movie:
            st.success(f"🎬 Selected Movie: **{active_movie}**")
        
        if not active_movie:
            st.subheader("🔥 Popular Movies")
            popular_movies = movies["names"].dropna().head(10).tolist()
            
            for row_start in range(0, min(len(popular_movies), 10), 5):
                cols = st.columns(5)
                for i, movie_title in enumerate(popular_movies[row_start:row_start+5]):
                    with cols[i]:
                        display_movie_card_with_rating(
                            movie_title, tmdb_available, headers, IMAGE_URL, 
                            f"st_pop_{row_start + i}", "st"
                        )
        else:
            if st.button("🔄 Clear Selection", key="clear_st"):
                st.session_state["active_movie_st"] = ""
                st.rerun()
            
            st.markdown(f"<div class='rec-header'><h2>🎞️ Semantic AI Recommendations for {active_movie}</h2></div>", unsafe_allow_html=True)
            
            # Get recommendations
            recs = recommend_sentence_transformer(movies, embeddings, active_movie)
            if recs:
                for row_start in range(0, min(len(recs), 15), 5):
                    cols = st.columns(5)
                    for i, rec_title in enumerate(recs[row_start:row_start+5]):
                        with cols[i]:
                            display_movie_card_with_rating(
                                rec_title, tmdb_available, headers, IMAGE_URL, 
                                f"st_rec_{row_start + i}", "st"
                            )
            else:
                st.warning("No recommendations found.")
    
    # Tab 2: TF-IDF + KNN
    with tab2:
        st.session_state.current_tab = "tfidf"
        
        st.markdown("### 📊 TF-IDF + K-Nearest Neighbors")
        st.info("Using TF-IDF vectorization and KNN for content-based filtering on movie overviews.")
        
        # Load models and data
        knn_model, tfidf_vectorizer, movies_df = load_tfidf_model()
        
        if knn_model is None or tfidf_vectorizer is None or movies_df is None:
            st.error("Failed to load TF-IDF components.")
            return
        
        # Initialize session state for this tab
        if "active_movie_tfidf" not in st.session_state:
            st.session_state["active_movie_tfidf"] = ""
        
        # Movie search in main area
        st.subheader("🔍 Search for a Movie")
        search_query = st.text_input("Type a movie name:", key="search_tfidf")
        
        # Filter movies based on search or show all
        if search_query:
            filtered_movies = movies_df[movies_df["names"].str.contains(search_query, case=False, na=False)]["names"].tolist()
        else:
            filtered_movies = movies_df["names"].dropna().head(50).tolist()  # Show first 50 movies by default
        
        if filtered_movies:
            selected_movie = st.selectbox(
                "Select a movie:",
                options=[""] + filtered_movies,
                format_func=lambda x: "Choose a movie..." if x == "" else x,
                key="selectbox_tfidf"
            )
            if selected_movie and selected_movie != "":
                st.session_state["active_movie_tfidf"] = selected_movie
        else:
            st.warning("No movies found matching your search.")
        
        active_movie = st.session_state.get("active_movie_tfidf", "")
        
        # Show currently selected movie
        if active_movie:
            st.success(f"🎬 Selected Movie: **{active_movie}**")
        
        if not active_movie:
            st.subheader("🔥 Popular Movies")
            popular_movies = movies_df["names"].dropna().head(10).tolist()
            
            for row_start in range(0, min(len(popular_movies), 10), 5):
                cols = st.columns(5)
                for i, movie_title in enumerate(popular_movies[row_start:row_start+5]):
                    with cols[i]:
                        display_movie_card_with_rating(
                            movie_title, tmdb_available, headers, IMAGE_URL, 
                            f"tfidf_pop_{row_start + i}", "tfidf"
                        )
        else:
            if st.button("🔄 Clear Selection", key="clear_tfidf"):
                st.session_state["active_movie_tfidf"] = ""
                st.rerun()
            
            st.markdown(f"<div class='rec-header'><h2>🎞️ TF-IDF Recommendations for {active_movie}</h2></div>", unsafe_allow_html=True)
            
            # Get recommendations
            recs = recommend_tfidf(active_movie, knn_model, tfidf_vectorizer, movies_df)
            if recs:
                for row_start in range(0, min(len(recs), 15), 5):
                    cols = st.columns(5)
                    for i, rec_title in enumerate(recs[row_start:row_start+5]):
                        with cols[i]:
                            display_movie_card_with_rating(
                                rec_title, tmdb_available, headers, IMAGE_URL, 
                                f"tfidf_rec_{row_start + i}", "tfidf"
                            )
            else:
                st.warning("No recommendations found.")
    
    # Tab 3: TMDB API
    with tab3:
        st.session_state.current_tab = "tmdb"
        
        st.markdown("### 🎭 TMDB API Recommendations")
        st.info("Using The Movie Database (TMDB) API for official movie recommendations.")
        
        # Load data
        movies_df = load_tmdb_movies()
        
        if movies_df.empty:
            st.error("Failed to load TMDB movie data.")
            return
        
        # Initialize session state for this tab
        if "active_movie_tmdb" not in st.session_state:
            st.session_state["active_movie_tmdb"] = ""
        
        # Movie search in main area
        st.subheader("🔍 Search for a Movie")
        search_query = st.text_input("Type a movie name:", key="search_tmdb")
        
        # Filter movies based on search or show all
        if search_query:
            filtered_movies = movies_df[movies_df["names"].str.contains(search_query, case=False, na=False)]["names"].tolist()
        else:
            filtered_movies = movies_df["names"].dropna().head(50).tolist()  # Show first 50 movies by default
        
        if filtered_movies:
            selected_movie = st.selectbox(
                                        "Select a movie:",
                                        options=[""] + filtered_movies,
                                        format_func=lambda x: "Choose a movie..." if x == "" else x,
                                        key="selectbox_tmdb"
                                        )
            if selected_movie and selected_movie != "":
                st.session_state["active_movie_tmdb"] = selected_movie
        else:
            st.warning("No movies found matching your search.")
        
        active_movie = st.session_state.get("active_movie_tmdb", "")
        
        # Show currently selected movie
        if active_movie:
            st.success(f"🎬 Selected Movie: **{active_movie}**")
        
        if not active_movie:
            st.subheader("🔥 Popular Movies")
            
            # Default popular movies with known posters
            default_movies = [
                            {"title":"The Shawshank Redemption","poster":"/q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg"},
                            {"title":"The Godfather","poster":"/rPdtLWNsZmAtoZl9PK7S2wE3qiS.jpg"},
                            {"title":"The Dark Knight","poster":"/qJ2tW6WMUDux911r6m7haRef0WH.jpg"},
                            {"title":"Money Heist","poster":"/reEMJA1uzscCbkpeRJeTT2bjqUp.jpg"},
                            {"title":"3 Idiots","poster":"/66A9MqXOyVFCssoloscw79z8Tew.jpg"}
                            ]
            
            cols = st.columns(5)
            for i, movie in enumerate(default_movies):
                with cols[i]:
                    st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
                    st.image(IMAGE_URL + movie["poster"], use_container_width=True)
                    if st.button(movie["title"], key=f"tmdb_default_{i}"):
                        st.session_state["active_movie_tmdb"] = movie["title"]
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            if st.button("🔄 Clear Selection", key="clear_tmdb"):
                st.session_state["active_movie_tmdb"] = ""
                st.rerun()
                
            movies = search_movie_tmdb(active_movie, tmdb_available, headers)
            if movies:
                selected_movie_data = movies[0]
                movie_id = selected_movie_data["id"]

                st.markdown(f"<div class='rec-header'><h2>🎞️ TMDB Recommendations for {active_movie}</h2></div>", unsafe_allow_html=True)

                recs = get_recommendations_tmdb(movie_id, tmdb_available, headers)
                if recs:
                    for row_start in range(0, min(len(recs), 15), 5):
                        cols = st.columns(5)
                        for i, rec in enumerate(recs[row_start:row_start+5]):
                            with cols[i]:
                                poster_path = rec.get("poster_path")
                                st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
                                if poster_path:
                                    st.image(IMAGE_URL + poster_path, use_container_width=True)
                                else:
                                    st.markdown(
                                        "<div style='height:300px; background: linear-gradient(135deg,#ddd,#f8f8f8); "
                                        "display:flex; align-items:center; justify-content:center; border-radius:8px; color:#666;'>"
                                        "🎬 No Image</div>",
                                        unsafe_allow_html=True,
                                    )
                                st.markdown(f"<div class='movie-rating'>⭐ {rec.get('vote_average','N/A')}/10</div>", unsafe_allow_html=True)
                                if st.button(rec["title"], key=f"tmdb_rec_{movie_id}_{row_start + i}"):
                                    st.session_state["active_movie_tmdb"] = rec["title"]
                                    st.rerun()
                                st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.warning("No recommendations found for this movie.")
            else:
                st.error(f"Movie '{active_movie}' not found on TMDB 😢")
                if st.button("🔄 Try Another Movie", key="retry_tmdb"):
                    st.session_state["active_movie_tmdb"] = ""
                    st.rerun()


if __name__ == "__main__":
    main()

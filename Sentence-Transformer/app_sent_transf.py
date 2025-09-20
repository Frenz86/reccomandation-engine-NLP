import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from numpy.linalg import norm
import requests

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_URL = "https://api.themoviedb.org/3"
IMAGE_URL = "https://image.tmdb.org/t/p/w500"

@st.cache_resource
def load_model():
    """Carica il modello SentenceTransformer"""
    model_path = os.path.join(SCRIPT_DIR, "pickle_model", "sentence_transformer_model.pkl")
    return joblib.load(model_path)

@st.cache_data
def load_data():
    """Carica il dataset CSV e gli embeddings"""
    movies_path = os.path.join(SCRIPT_DIR, "imdb_movies.csv")
    embeddings_path = os.path.join(SCRIPT_DIR, "pickle_model", "movie_embeddings.pkl")
    
    # Carica il dataset CSV con pandas
    movies = pd.read_csv(movies_path)
    movies = movies.dropna(subset=["names", "overview"])  # Pulisce i dati come nel notebook
    
    # Carica gli embeddings con joblib
    embeddings = joblib.load(embeddings_path)
    
    return movies, embeddings

def setup_tmdb_api():
    """Configura l'API TMDB se disponibile"""
    try:
        api_key = st.secrets["API_KEY"]
        # Il token è già completo con "Bearer", usiamolo direttamente
        headers = {
            "accept": "application/json",
            "Authorization": api_key
        }
        
        # Test rapido dell'API
        test_url = f"{BASE_URL}/search/movie?query=avatar"
        test_response = requests.get(test_url, headers=headers)
        
        if test_response.status_code == 200:
            return True, headers
        else:
            st.sidebar.error(f"⚠️ TMDB API Error: {test_response.status_code}")
            st.sidebar.error(f"Response: {test_response.text[:200]}...")
            return False, None
            
    except (KeyError, FileNotFoundError):
        return False, None
    except Exception as e:
        st.sidebar.error(f"⚠️ TMDB API Connection Error: {str(e)}")
        return False, None

def search_movie_tmdb(query, tmdb_available, headers):
    """Cerca un film usando l'API TMDB"""
    if not tmdb_available:
        return []
    try:
        url = f"{BASE_URL}/search/movie?query={query}"
        response = requests.get(url, headers=headers)
        
        # Debug: stampa status code se diverso da 200
        if response.status_code != 200:
            st.error(f"TMDB API Error: {response.status_code} - {response.text}")
            return []
        
        return response.json().get("results", [])
    except Exception as e:
        st.error(f"Error searching TMDB: {str(e)}")
        return []

# -----------------------------
# RECOMMENDATION FUNCTIONS
# -----------------------------
def recommend(movies, embeddings, movie_name, top_k=15):
    """Raccomandazioni basate su embeddings precomputati (più veloce)"""
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
    scores = np.dot(embeddings_np, query_embedding) / (
        norm(embeddings_np, axis=1) * norm(query_embedding)
    )

    # top results (skip the same movie)
    top_indices = np.argsort(scores)[::-1][1:top_k+1]

    return movies.iloc[top_indices]["names"].tolist()

def recommend_with_model(model, movies, embeddings, query_text, top_k=15):
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
    scores = np.dot(embeddings_np, query_embedding) / (
        norm(embeddings_np, axis=1) * norm(query_embedding)
    )

    # top results
    top_indices = np.argsort(scores)[::-1][:top_k]

    return movies.iloc[top_indices]["names"].tolist()

# -----------------------------
# UI FUNCTIONS
# -----------------------------
def setup_css():
    """Configura il CSS personalizzato"""
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
        display:flex;
        justify-content:center;
        align-items:center;
        margin: 20px auto;
        padding: 14px 26px;
        min-height:64px;
        border-radius: 36px;
        box-shadow: 0 10px 30px rgba(30,64,175,0.12);
        text-align:center;
        width: min(620px, 80%);
        box-sizing: border-box;
    }
    .rec-header h2 { margin:0; font-size:18px; font-weight:600; color:#ffffff; }

    .movie-card{
        text-align:center;
        padding:10px;
        border-radius:12px;
        background-color:#ffffff;
        transition: transform 0.18s ease-in-out;
        margin-bottom:15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    .movie-card:hover { transform: scale(1.06); }

    .movie-card .stButton>button{
        background-color: var(--brand) !important;
        color: #ffffff !important;
        font-weight:600;
        width:100%;
        border-radius:8px;
        padding:7px 0;
        transition: transform 0.12s;
        border: none !important;
        white-space: normal;
    }
    .movie-card .stButton>button:hover{ transform: scale(1.03); }

    .movie-rating {
        font-size: 14px;
        color: #666;
        margin: 8px 0;
        font-weight: 500;
    }

    .stButton>button { border-radius:8px; padding:6px 8px; }
    </style>
    """, unsafe_allow_html=True)

def setup_sidebar(movies, tmdb_available):
    """Configura la sidebar con le opzioni di ricerca"""
    st.sidebar.header("🎥 Movie Search")

    # Mostra status TMDB
    if tmdb_available:
        st.sidebar.success("✅ TMDB API connected - Posters enabled!")
    else:
        st.sidebar.warning("⚠️ TMDB API not available. Posters will not be displayed.")

    # Aggiungiamo un'opzione per ricerca testuale
    search_mode = st.sidebar.radio(
        "Search Mode:",
        ["Search by Movie Title", "Search by Description"],
        help="Choose how you want to search for recommendations"
    )

    if search_mode == "Search by Description":
        # Ricerca testuale usando il modello
        text_query = st.sidebar.text_area(
            "Describe the type of movie you want:",
            placeholder="e.g., 'action movie with superheroes', 'romantic comedy', 'sci-fi thriller'",
            height=100
        )
        
        if st.sidebar.button("🔍 Find Movies") and text_query.strip():
            st.session_state.active_movie = f"TEXT_SEARCH:{text_query}"
            st.sidebar.success("Search completed!")
    else:
        # Ricerca normale per titolo film
        all_movie_names = movies["names"].dropna().unique().tolist()
        sidebar_options = [""] + all_movie_names
        current_active = st.session_state.active_movie
        
        # Se è una ricerca testuale, resetta
        if current_active and current_active.startswith("TEXT_SEARCH:"):
            current_active = ""
            st.session_state.active_movie = ""
        
        if current_active and current_active not in all_movie_names:
            sidebar_options.append(current_active)
        try:
            sidebar_index = sidebar_options.index(current_active) if current_active else 0
        except ValueError:
            sidebar_index = 0

        sidebar_selection = st.sidebar.selectbox("Search a movie:", options=sidebar_options, index=sidebar_index)
        if sidebar_selection != st.session_state.active_movie:
            st.session_state.active_movie = sidebar_selection

def display_movie_card(movie_title, tmdb_available, headers, key):
    """Mostra una singola card del film"""
    st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
    
    # Try to get poster from TMDB
    search_results = search_movie_tmdb(movie_title, tmdb_available, headers)
    if search_results and search_results[0].get("poster_path"):
        st.image(IMAGE_URL + search_results[0]["poster_path"], use_container_width=True)
        # Add rating stars
        st.markdown(f"<div class='movie-rating'>⭐ {search_results[0].get('vote_average','N/A')}/10</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='height:300px; background: linear-gradient(135deg,#ddd,#f8f8f8); "
            "display:flex; align-items:center; justify-content:center; border-radius:8px; color:#666;'>"
            "🎬 No Image</div>",
            unsafe_allow_html=True,
        )
        # Add N/A rating for missing posters
        st.markdown("<div class='movie-rating'>⭐ N/A</div>", unsafe_allow_html=True)
    
    if st.button(movie_title, key=key):
        st.session_state.active_movie = movie_title
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

def display_recommendation_card(rec_title, tmdb_available, headers, key):
    """Mostra una card di raccomandazione con rating"""
    st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
    
    # Get poster and info from TMDB
    rec_results = search_movie_tmdb(rec_title, tmdb_available, headers)
    if rec_results:
        rec_movie = rec_results[0]
        poster_path = rec_movie.get("poster_path")
        if poster_path:
            st.image(IMAGE_URL + poster_path, use_container_width=True)
        else:
            st.markdown(
                "<div style='height:300px; background: linear-gradient(135deg,#ddd,#f8f8f8); "
                "display:flex; align-items:center; justify-content:center; border-radius:8px; color:#666;'>"
                "🎬 No Image</div>",
                unsafe_allow_html=True,
            )
        st.markdown(f"<div class='movie-rating'>⭐ {rec_movie.get('vote_average','N/A')}/10</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='height:300px; background: linear-gradient(135deg,#ddd,#f8f8f8); "
            "display:flex; align-items:center; justify-content:center; border-radius:8px; color:#666;'>"
            "🎬 No Image</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='movie-rating'>⭐ N/A</div>", unsafe_allow_html=True)
    
    if st.button(rec_title, key=key):
        st.session_state.active_movie = rec_title
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# MAIN APPLICATION
# -----------------------------
def main():
    """Funzione principale dell'applicazione"""
    # Initialize session state
    if "active_movie" not in st.session_state:
        st.session_state.active_movie = ""

    # Load data and model
    model = load_model()
    movies, embeddings = load_data()
    tmdb_available, headers = setup_tmdb_api()

    # Setup UI
    setup_css()
    setup_sidebar(movies, tmdb_available)

    active_movie_name = st.session_state.active_movie

    # --- MAIN SCREEN ---
    if not active_movie_name:
        st.markdown(
            "<div class='hero'><div><h1>🎬 Movie Recommendation System</h1>"
            "<p>Discover movies similar to your favorites using Sentence Transformers</p></div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.subheader("🔥 Popular Movies")

        # Show some popular movies from your dataset as default options
        popular_movies_sample = movies["names"].dropna().head(10).tolist()
        
        # Display in rows of 5
        for row_start in range(0, min(len(popular_movies_sample), 10), 5):
            cols = st.columns(5)
            for i, movie_title in enumerate(popular_movies_sample[row_start:row_start+5]):
                with cols[i]:
                    display_movie_card(movie_title, tmdb_available, headers, f"popular_{row_start + i}")

        st.markdown("---")
    else:
        # Check if it's a text search or movie search
        is_text_search = active_movie_name.startswith("TEXT_SEARCH:")
        
        if is_text_search:
            # Handle text search
            query_text = active_movie_name.replace("TEXT_SEARCH:", "")
            st.markdown(f"<div class='rec-header'><h2>🔍 AI Recommendations for: \"{query_text}\"</h2></div>", unsafe_allow_html=True)
            st.markdown("---")
            
            # Get recommendations using the model
            recs = recommend_with_model(model, movies, embeddings, query_text, top_k=15)
            
        else:
            # Handle movie search - Show sidebar poster & info
            search_results = search_movie_tmdb(active_movie_name, tmdb_available, headers)
            if search_results:
                selected_movie = search_results[0]
                poster_path = selected_movie.get("poster_path")
                if poster_path:
                    st.sidebar.image(IMAGE_URL + poster_path, caption=active_movie_name, use_container_width=True)
                st.sidebar.markdown(f"**Release Date:** {selected_movie.get('release_date','N/A')}")
                st.sidebar.markdown(f"**Rating:** {selected_movie.get('vote_average','N/A')}/10")
                if selected_movie.get("overview"):
                    st.sidebar.markdown(f"**Overview:** {selected_movie['overview'][:200]}...")

            st.markdown(f"<div class='rec-header'><h2>🎞️ Semantic AI Recommendations for {active_movie_name}</h2></div>", unsafe_allow_html=True)
            st.markdown("---")

            # Get recommendations from your Sentence Transformer model
            recs = recommend(movies, embeddings, active_movie_name, top_k=15)

        if st.sidebar.button("🔄 Clear Selection"):
            st.session_state.active_movie = ""
            st.rerun()

        if recs:
            # Show up to 15 recommendations in 3 rows of 5
            for row_start in range(0, min(len(recs), 15), 5):
                cols = st.columns(5)
                for i, rec_title in enumerate(recs[row_start:row_start+5]):
                    with cols[i]:
                        display_recommendation_card(rec_title, tmdb_available, headers, f"rec_{active_movie_name}_{row_start + i}")
        else:
            st.warning("No recommendations found.")
            if not is_text_search:
                st.info("This movie might not be in our training dataset. Try searching for a different movie using the sidebar.")

if __name__ == "__main__":
    main()

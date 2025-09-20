import streamlit as st
import pandas as pd
import requests
import joblib
import os


def load_model():
    """Load the trained KNN model and TF-IDF vectorizer."""
    try:
        # Load saved models and data
        model_path = os.path.join("pickle_model", "knn_model.pkl")
        vectorizer_path = os.path.join("pickle_model", "tfidf_vectorizer.pkl")
        
        knn_model = joblib.load(model_path)
        tfidf_vectorizer = joblib.load(vectorizer_path)
        
        # Load movie dataset (include overview column as it's used for recommendations)
        movies_df = pd.read_csv("imdb_movies.csv")
        movies_df = movies_df.dropna(subset=["names", "overview"])  # Need both names and overview
        
        return knn_model, tfidf_vectorizer, movies_df
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


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
        BASE_URL = "https://api.themoviedb.org/3"
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


def search_movie_tmdb(movie_title, tmdb_available, headers):
    """Search for a movie using TMDB API."""
    if not tmdb_available or not headers:
        return None
    
    BASE_URL = "https://api.themoviedb.org/3"
    url = f"{BASE_URL}/search/movie"
    
    params = {
        "query": movie_title,
        "include_adult": "false",
        "language": "en-US",
        "page": "1"
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            return data.get("results", [])
        else:
            st.error(f"TMDB API Error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"Error fetching movie data: {e}")
    
    return None


def recommend(movie_name, knn_model, tfidf_vectorizer, movies_df, n_recommendations=10):
    """Generate movie recommendations using TF-IDF + KNN."""
    try:
        # Check if movie exists in dataset
        if movie_name not in movies_df["names"].values:
            return []
        
        # Get movie index
        movie_index = movies_df[movies_df["names"] == movie_name].index[0]
        
        # Get the movie's TF-IDF vector using the overview column (as used in training)
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

    div[data-testid="stSelectbox"] label {
        color: #ffffff !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


def create_sidebar(movies_df, tmdb_available, headers, image_url):
    """Create and manage the sidebar interface."""
    st.sidebar.title("🎬 Movie Search")
    
    # Movie search
    search_query = st.sidebar.text_input("Search for a movie:")
    
    # Filter movies based on search
    if search_query:
        filtered_movies = movies_df[
            movies_df["names"].str.contains(search_query, case=False, na=False)
        ]["names"].tolist()
    else:
        filtered_movies = movies_df["names"].tolist()
    
    # Select movie
    if filtered_movies:
        sidebar_selection = st.sidebar.selectbox(
            "Select a movie:",
            options=[""] + filtered_movies[:50],  # Limit to first 50 for performance
            format_func=lambda x: "Choose a movie..." if x == "" else x
        )
    else:
        sidebar_selection = st.sidebar.selectbox(
            "Select a movie:",
            options=[""],
            format_func=lambda x: "No movies found..."
        )
    
    # Update session state
    if sidebar_selection and sidebar_selection != "":
        st.session_state.active_movie = sidebar_selection
    
    return st.session_state.get("active_movie", "")


def display_home_screen(movies_df, tmdb_available, headers, image_url):
    """Display the home screen with popular movies."""
    st.markdown(
        "<div class='hero'><div><h1>🎬 Movie Recommendation System</h1>"
        "<p>Discover movies similar to your favorites using TF-IDF + KNN! Search on the left to get started.</p></div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.subheader("🔥 Popular Movies")

    # Show some popular movies from your dataset as default options
    popular_movies_sample = movies_df["names"].dropna().head(10).tolist()
    
    # Display in rows of 5
    for row_start in range(0, min(len(popular_movies_sample), 10), 5):
        cols = st.columns(5)
        for i, movie_title in enumerate(popular_movies_sample[row_start:row_start+5]):
            with cols[i]:
                st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
                
                # Try to get poster from TMDB
                search_results = search_movie_tmdb(movie_title, tmdb_available, headers)
                if search_results and search_results[0].get("poster_path"):
                    st.image(image_url + search_results[0]["poster_path"], use_container_width=True)
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
                
                if st.button(movie_title, key=f"popular_{row_start + i}"):
                    st.session_state.active_movie = movie_title
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)


def display_movie_details(active_movie_name, tmdb_available, headers, image_url):
    """Display movie details in the sidebar."""
    search_results = search_movie_tmdb(active_movie_name, tmdb_available, headers)
    if search_results:
        selected_movie = search_results[0]
        poster_path = selected_movie.get("poster_path")
        if poster_path:
            st.sidebar.image(image_url + poster_path, caption=active_movie_name, use_container_width=True)
        st.sidebar.markdown(f"**Release Date:** {selected_movie.get('release_date','N/A')}")
        st.sidebar.markdown(f"**Rating:** {selected_movie.get('vote_average','N/A')}/10")
        if selected_movie.get("overview"):
            st.sidebar.markdown(f"**Overview:** {selected_movie['overview'][:200]}...")

    if st.sidebar.button("🔄 Clear Selection"):
        st.session_state.active_movie = ""
        st.rerun()


def display_recommendations(active_movie_name, knn_model, tfidf_vectorizer, movies_df, tmdb_available, headers, image_url):
    """Display movie recommendations."""
    st.markdown(f"<div class='rec-header'><h2>🎞️ AI Recommendations for {active_movie_name}</h2></div>", unsafe_allow_html=True)
    st.markdown("---")

    # Get recommendations from your ML model
    recs = recommend(active_movie_name, knn_model, tfidf_vectorizer, movies_df)
    if recs:
        # Show up to 15 recommendations in 3 rows of 5
        for row_start in range(0, min(len(recs), 15), 5):
            cols = st.columns(5)
            for i, rec_title in enumerate(recs[row_start:row_start+5]):
                with cols[i]:
                    st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
                    
                    # Get poster and info from TMDB
                    rec_results = search_movie_tmdb(rec_title, tmdb_available, headers)
                    if rec_results:
                        rec_movie = rec_results[0]
                        poster_path = rec_movie.get("poster_path")
                        if poster_path:
                            st.image(image_url + poster_path, use_container_width=True)
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
                    
                    if st.button(rec_title, key=f"rec_{active_movie_name}_{row_start + i}"):
                        st.session_state.active_movie = rec_title
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("No recommendations found for this movie.")
        st.info("This movie might not be in our training dataset. Try searching for a different movie using the sidebar.")


def main():
    """Main application function."""
    # Page configuration
    st.set_page_config(
        page_title="TF-IDF Movie Recommendations",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    apply_custom_css()
    
    # Load model and data
    knn_model, tfidf_vectorizer, movies_df = load_model()
    
    if knn_model is None or tfidf_vectorizer is None or movies_df is None:
        st.error("Failed to load model components. Please check the model files.")
        st.stop()
    
    # Setup TMDB API
    tmdb_available, headers = setup_tmdb_api()
    IMAGE_URL = "https://image.tmdb.org/t/p/w500"
    
    # Initialize session state
    if "active_movie" not in st.session_state:
        st.session_state.active_movie = ""
    
    # Create sidebar and get active movie
    active_movie_name = create_sidebar(movies_df, tmdb_available, headers, IMAGE_URL)
    
    # Main content area
    if not active_movie_name:
        display_home_screen(movies_df, tmdb_available, headers, IMAGE_URL)
    else:
        display_movie_details(active_movie_name, tmdb_available, headers, IMAGE_URL)
        display_recommendations(active_movie_name, knn_model, tfidf_vectorizer, movies_df, tmdb_available, headers, IMAGE_URL)


if __name__ == "__main__":
    main()
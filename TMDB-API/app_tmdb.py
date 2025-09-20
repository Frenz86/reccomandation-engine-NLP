import streamlit as st
import pandas as pd
import requests

# --- PAGE CONFIG MUST BE FIRST ---
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_data
def load_movies():
    """Load movie dataset."""
    return pd.read_csv("imdb_movies.csv")   # must contain a "names" column


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


def get_recommendations(movie_id, tmdb_available, headers):
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
            return []
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
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
        background-color: var(--brand);
        color: #ffffff;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 12px;
        margin: 8px auto 4px;
        display: inline-block;
    }

    .stButton>button { border-radius:8px; padding:6px 8px; }
    </style>
    """, unsafe_allow_html=True)


def main():
    """Main application function."""
    # Apply custom CSS
    apply_custom_css()
    
    # Load data
    movies_df = load_movies()
    IMAGE_URL = "https://image.tmdb.org/t/p/w500"
    
    # Initialize session state
    if "active_movie" not in st.session_state:
        st.session_state.active_movie = ""

    # Setup TMDB API
    tmdb_available, headers = setup_tmdb_api()

    # --- SIDEBAR SEARCH ---
    st.sidebar.header("🎥 Movie Search")

    # Show TMDB API status
    if tmdb_available:
        st.sidebar.success("✅ TMDB API connected!")
    else:
        st.sidebar.warning("⚠️ TMDB API not available. Limited functionality.")

    all_movie_names = movies_df["names"].dropna().unique().tolist()

    sidebar_options = [""] + all_movie_names
    current_active = st.session_state.active_movie
    if current_active and current_active not in all_movie_names:
        sidebar_options.append(current_active)
    try:
        sidebar_index = sidebar_options.index(current_active) if current_active else 0
    except ValueError:
        sidebar_index = 0

    sidebar_selection = st.sidebar.selectbox("Search a movie:", options=sidebar_options, index=sidebar_index)
    if sidebar_selection != st.session_state.active_movie:
        st.session_state.active_movie = sidebar_selection
    active_movie_name = st.session_state.active_movie

    # --- MAIN SCREEN ---
    if not active_movie_name:
        st.markdown(
            "<div class='hero'><div><h1>🎬 Movie Recommendation System</h1>"
            "<p>Discover movies similar to your favorites! Search on the left to get started.</p></div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.subheader("🔥 Popular Movies")

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
                if st.button(movie["title"], key=f"default_{i}"):
                    st.session_state.active_movie = movie["title"]
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.info("👉 Use the sidebar or click a movie to explore recommendations.")
    else:
        movies = search_movie_tmdb(active_movie_name, tmdb_available, headers)
        if movies:
            selected_movie = movies[0]
            movie_id = selected_movie["id"]
            poster_path = selected_movie.get("poster_path")

            if poster_path:
                st.sidebar.image(IMAGE_URL + poster_path, caption=active_movie_name, use_container_width=True)
            st.sidebar.markdown(f"**Release Date:** {selected_movie.get('release_date','N/A')}")
            st.sidebar.markdown(f"**Rating:** {selected_movie.get('vote_average','N/A')}/10")
            if selected_movie.get('overview'):
                st.sidebar.markdown(f"**Overview:** {selected_movie['overview'][:200]}...")

            if st.sidebar.button("🔄 Clear Selection"):
                st.session_state.active_movie = ""
                st.rerun()

            st.markdown(f"<div class='rec-header'><h2>🎞️ Recommendations for {active_movie_name}</h2></div>", unsafe_allow_html=True)
            st.markdown("---")

            recs = get_recommendations(movie_id, tmdb_available, headers)
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
                            if st.button(rec["title"], key=f"rec_{movie_id}_{row_start + i}"):
                                st.session_state.active_movie = rec["title"]
                                st.rerun()
                            st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("No recommendations found for this movie.")
                st.info("Try searching for a different movie using the sidebar.")
        else:
            st.error(f"Movie '{active_movie_name}' not found on TMDB 😢")
            st.info("Please try a different movie name from the sidebar.")
            if st.button("🔄 Try Another Movie"):
                st.session_state.active_movie = ""
                st.rerun()


if __name__ == "__main__":
    main()

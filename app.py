import streamlit as st
import pandas as pd
import requests
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

### Set Page config
st.set_page_config(page_title= f"Text Analytics w/ Movie Reviews",page_icon="🧑‍🚀",layout="wide")

### Set Background image
page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background-image: url("https://source.unsplash.com/a-close-up-of-a-white-wall-with-wavy-lines-75xPHEQBmvA");
background-size: cover;
}
</style>
"""
### Big title
st.markdown(f"<h1 style='text-align: center;'>What are people saying about the movies??</h1>", unsafe_allow_html=True)
    
### Introduction section
st.markdown(page_bg_img, unsafe_allow_html=True)

# Download necessary NLTK data
nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Function to calculate VADER sentiment polarity
def get_vader_sentiment(text):
    sentiment_dict = sid.polarity_scores(text)
    return sentiment_dict['compound']  # Using compound score for overall sentiment

# Function to calculate TextBlob sentiment polarity
def get_textblob_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Function to call the TMDb API and discover movies
def fetch_movies_from_api():
    url_discover = "https://api.themoviedb.org/3/discover/movie"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {st.secrets['tmdb']['bearer_token']}"
    }
    response = requests.get(url_discover, headers=headers)
    if response.status_code == 200:
        return response.json().get('results', [])
    else:
        st.error("Failed to fetch movies from the API")
        return []


def display_movie_poster(poster_path):
    base_url = 'https://image.tmdb.org/t/p/'
    size = 'w500'  # You can adjust the size as needed

    # Construct the full URL
    full_url = f"{base_url}{size}{poster_path}"

    # Display the image
    st.image(full_url)

# Function to call the TMDb API and search for movies
def search_movies_from_api(query):
    url_search = "https://api.themoviedb.org/3/search/movie"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {st.secrets['tmdb']['bearer_token']}"
    }
    params = {
        "query": query
    }
    response = requests.get(url_search, headers=headers, params=params)
    if response.status_code == 200:
        return response.json().get('results', [])
    else:
        st.error("Failed to search for movies from the API")
        return []

# Function to fetch movie details by movie ID
def fetch_movie_details(movie_id):
    url_movie = f"https://api.themoviedb.org/3/movie/{movie_id}"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {st.secrets['tmdb']['bearer_token']}"
    }
    response = requests.get(url_movie, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to fetch movie details from the API")
        return None


# Streamlit app layout
def main():
    st.title("Movie Reviews Sentiment Analysis")
    
     # Search for movies
    search_query = st.text_input("Search for a movie")

    if search_query:
        st.write(f"Searching for movies matching: {search_query}")
        movies = search_movies_from_api(search_query)

        if movies:
            # Convert the movies data to a DataFrame
            df_movies = pd.DataFrame(movies)

            # Display dropdown for movie selection
            movie_titles = df_movies['title'].tolist()
            selected_movie_title = st.selectbox("Select a movie", movie_titles)

            # Fetch details for the selected movie
            selected_movie = df_movies[df_movies['title'] == selected_movie_title].iloc[0]
            movie_details = fetch_movie_details(selected_movie['id'])

            if movie_details:
                # Display selected movie details
                st.write(f"**Title**: {movie_details['title']}")
                st.write(f"**Release Date**: {movie_details['release_date']}")
                st.write(f"**Overview**: {movie_details['overview']}")
                display_movie_poster(movie_details['poster_path'])
# Run the app
if __name__ == "__main__":
    main()

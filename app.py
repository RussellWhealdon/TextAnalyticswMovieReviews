import streamlit as st
import pandas as pd
import requests
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

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

# Function to clean the text
def clean_text(text):
    # Remove special characters and numbers, convert to lowercase
    text = ''.join(char.lower() if char.isalpha() or char.isspace() else ' ' for char in text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)  # Join tokens back into a single string

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
# Function to get reviews for a movie ID
def fetch_movie_reviews(movie_id):
    url_reviews = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {st.secrets['tmdb']['bearer_token']}"
    }
    response = requests.get(url_reviews, headers=headers)
    if response.status_code == 200:
        return response.json().get('results', [])
    else:
        st.error("Failed to fetch movie reviews from the API")
        return []


# Function to get the average score given to a movie by its ID
def get_average_score(movie_id):
    url_movie = f"https://api.themoviedb.org/3/movie/{movie_id}"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {st.secrets['tmdb']['bearer_token']}"
    }
    response = requests.get(url_movie, headers=headers)
    if response.status_code == 200:
        movie_details = response.json()
        return movie_details.get('vote_average', 0)  # Return the average score, default to 0 if not found
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
                st.write(pd.DataFrame(movie_details))
                display_movie_poster(movie_details['poster_path'])

                             # Fetch reviews for the selected movie
                reviews = fetch_movie_reviews(selected_movie['id'])
                if reviews:
                    df_reviews = pd.DataFrame(reviews)

                    # Extract review content
                    df_reviews['CleanedText'] = df_reviews['content'].apply(clean_text)

                    # Apply sentiment analysis
                    df_reviews['vader_sentiment'] = df_reviews['CleanedText'].apply(get_vader_sentiment)
                    df_reviews['textblob_sentiment'] = df_reviews['CleanedText'].apply(get_textblob_sentiment)

                    # Display sentiment scores
                    if st.checkbox("Show Sentiment Scores"):
                        st.write(df_reviews[['author', 'CleanedText', 'vader_sentiment', 'textblob_sentiment']])

                    # Average sentiment
                    average_sentiment = df_reviews['vader_sentiment'].mean()
                    st.write(f"Average Sentiment: {average_sentiment}")

                    # Review with the lowest sentiment
                    min_sentiment_index = df_reviews['vader_sentiment'].idxmin()
                    lowest_sentiment_review = df_reviews.loc[min_sentiment_index]

                    st.write("Review with the Lowest Sentiment")
                    st.write(lowest_sentiment_review)
                else:
                    st.write("No reviews found")
            else:
                st.write("No movie details to display")
        else:
            st.write("No movies found")
    else:
        st.write("Enter a movie title to search")


# Run the app
if __name__ == "__main__":
    main()

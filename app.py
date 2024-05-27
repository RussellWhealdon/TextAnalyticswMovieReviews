import streamlit as st
import pandas as pd
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

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

# Sample DataFrame (replace this with your actual data loading code)
def load_data():
    # For demonstration, we'll create a sample DataFrame
    data = {
        'movie_id': [1, 1, 2, 2, 3, 3],
        'title': ['Movie One', 'Movie One', 'Movie Two', 'Movie Two', 'Movie Three', 'Movie Three'],
        'CleanedText': [
            'This movie was great!', 'I loved the movie.', 'Not my cup of tea.', 'Could have been better.', 'Awful movie.', 'Terrible experience.'
        ]
    }
    df = pd.DataFrame(data)
    return df

# Streamlit app layout
def main():
    st.title("Movie Reviews Sentiment Analysis")

    # Load data
    reviews = load_data()

    # Show raw data
    if st.checkbox("Show Raw Data"):
        st.write(reviews)

    # Apply sentiment analysis
    reviews['vader_sentiment'] = reviews['CleanedText'].apply(get_vader_sentiment)
    reviews['textblob_sentiment'] = reviews['CleanedText'].apply(get_textblob_sentiment)

    # Display sentiment scores
    if st.checkbox("Show Sentiment Scores"):
        st.write(reviews[['movie_id', 'title', 'CleanedText', 'vader_sentiment', 'textblob_sentiment']])

    # Average sentiment by movie
    average_sentiment = reviews.groupby('movie_id')['vader_sentiment'].mean().reset_index()
    average_sentiment = average_sentiment.merge(reviews[['movie_id', 'title']].drop_duplicates(), on='movie_id')
    average_sentiment.rename(columns={'vader_sentiment': 'average_sentiment'}, inplace=True)

    # Display average sentiment scores
    if st.checkbox("Show Average Sentiment Scores"):
        st.write(average_sentiment.sort_values(by='average_sentiment', ascending=False))

    # Review with the lowest sentiment
    min_sentiment_index = reviews['vader_sentiment'].idxmin()
    lowest_sentiment_review = reviews.loc[min_sentiment_index]

    st.write("Review with the Lowest Sentiment")
    st.write(lowest_sentiment_review)

# Run the app
if __name__ == "__main__":
    main()

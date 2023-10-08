import requests
import sys
import re
import emoji
import pandas as pd
from emosent import get_emoji_sentiment_rank
from nltk.sentiment.vader import SentimentIntensityAnalyzer

LIKE_WEIGHT = 1.75
EMOJI_WEIGHT = 1.25

class Vader: 
    def read_data(self, url)->dict:
        """
        Reads data from the given URL and returns it as a dictionary.

        Args:
        url (str): The URL to read data from.

        Returns:
        dict: The data read from the URL as a dictionary.
        """
        try:
            response = requests.get(url)
            response.encoding = 'utf-8'
            data = response.json()     
            return data
        except Exception as e:
            print(e)
            sys.exit(1)

    def transform_data(self, data):
        """
        Transforms the given data into a pandas DataFrame.

        Args:
        data (dict): The data to transform.

        Returns:
        pandas.DataFrame: The transformed data as a pandas DataFrame.
        """
        text_display = []
        like_count = []
        for comment in data["items"]:
            snippet = comment["snippet"]["topLevelComment"]["snippet"]
            text = snippet["textDisplay"]
            text_display.append(text)
            like_count.append(snippet["likeCount"])
        df = pd.DataFrame({'comment': text_display, 'likeCount': like_count})
        return df          
    
    def seprate_emojis(self, df):
        """
        Separates emojis from comments in the given DataFrame and returns a new DataFrame.

        Args:
        df (pandas.DataFrame): The DataFrame to separate emojis from.

        Returns:
        pandas.DataFrame: A new DataFrame with emojis separated from comments.
        """
        emojis = []
        emoji_pattern = re.compile(emoji.get_emoji_regexp())

        # Create a new column to store comments without emojis
        df['comment_no_emojis'] = df['comment'].apply(lambda x: emoji_pattern.sub('', x))

        for comment in df['comment']:
            emojis_found = emoji_pattern.findall(comment)
            emojis.append(emojis_found)
        df['emojis'] = emojis 
        return df

    def clean_data(self, df):
        """
        Cleans the given DataFrame by removing unwanted columns and special characters.

        Args:
        df (pandas.DataFrame): The DataFrame to clean.

        Returns:
        pandas.DataFrame: The cleaned DataFrame.
        """
        df.drop(columns=['comment'], inplace=True)
        df['comment_no_emojis'] = df['comment_no_emojis'].str.replace(r'[^a-zA-Z]+', ' ') 
        return df 
    
    def _nltk_vader(self, df):
        """
        Performs sentiment analysis on the given DataFrame using the VADER algorithm from the NLTK library.

        Args:
        df (pandas.DataFrame): The DataFrame to perform sentiment analysis on.

        Returns:
        pandas.DataFrame: The DataFrame with sentiment scores added.
        """
        # VADER Sentiment Analysis
        sia = SentimentIntensityAnalyzer()
        scores = []
        vader_df = df.copy()

        for comment in vader_df['comment_no_emojis']:
            score = sia.polarity_scores(comment)
            scores.append(score)

        scores_df = pd.DataFrame(scores)
        vader_df = vader_df.join(scores_df)

        vader_df['compound'] = vader_df['compound'] * LIKE_WEIGHT * vader_df['emojis'].apply(lambda x: len(x) + 1) 
        return vader_df


    def _emojis_sentiment(self, df):
        """
        Calculates sentiment scores for emojis in the given DataFrame and returns a new DataFrame.

        Args:
        df (pandas.DataFrame): The DataFrame to calculate emoji sentiment scores for.

        Returns:
        pandas.DataFrame: A new DataFrame with emoji sentiment scores added.
        """
        emojis_sentiment = []
        for emojis in df['emojis']:
            sentiment_scores = [get_emoji_sentiment_rank(emoji) for emoji in emojis]
            valid_scores = [score['sentiment_score'] for score in sentiment_scores if score is not None]
            if valid_scores:
                emojis_sentiment.append(sum(valid_scores))
            else:
                emojis_sentiment.append(0)  # Handle the case where there are no valid sentiment scores
        df['emojis_sentiment'] = emojis_sentiment    
        return df

    def nltk_vader_sentiment_analysis(self, df): 
        """
        Performs sentiment analysis on the given DataFrame using the VADER algorithm from the NLTK library and emoji sentiment scores.

        Args:
        df (pandas.DataFrame): The DataFrame to perform sentiment analysis on.

        Returns:
        float: The average sentiment score for the DataFrame.
        """
        vader_df = self._nltk_vader(df)
        vader_emojis_score_df = self._emojis_sentiment(vader_df)
        vader_emojis_score_df['total_score'] = vader_df['compound'] + vader_emojis_score_df['emojis_sentiment'] * EMOJI_WEIGHT
        return vader_emojis_score_df['total_score'].mean()

    def runner(self, url):
        """
        Runs the sentiment analysis pipeline on the data from the given URL.

        Args:
        url (str): The URL to read data from.

        Returns:
        list, float: A list of cleaned comments and the average sentiment score for the DataFrame.
        """
        data = self.read_data(url)
        df = self.transform_data(data)
        df = self.seprate_emojis(df)
        df = self.clean_data(df)
        clened_comment = df['comment_no_emojis']
        value = self.nltk_vader_sentiment_analysis(df)
        return [clened_comment], value
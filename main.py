import os
import sys
import re
import emoji
import requests
import pandas as pd
from emosent import get_emoji_sentiment_rank
from dotenv import load_dotenv
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class SentimentAnalysis:
    def __init__(self):
        load_dotenv()
        self.YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
        self.MAX_RESULTS = os.getenv("MAX_RESULTS")
        self.LIKE_WEIGHT = float(os.getenv("LIKE_WEIGHT"))
        self.EMOJI_WEIGHT = float(os.getenv("EMOJI_WEIGHT"))
        self.video_id = 'eNZplHsZpvc'
        self.url = f'https://www.googleapis.com/youtube/v3/commentThreads?key={self.YOUTUBE_API_KEY}&textFormat=plainText&part=snippet&videoId={self.video_id}&maxResults={self.MAX_RESULTS}'

    def get_comments_data(self):
        try:
            response = requests.get(self.url)
            response.encoding = 'utf-8'
            self.data = response.json()
        except Exception as e:
            print(e)
            sys.exit(1)    

    def transform_data(self):
        text_display = []
        like_count = []
        for comment in self.data["items"]:
            snippet = comment["snippet"]["topLevelComment"]["snippet"]
            text = snippet["textDisplay"]
            #text = emoji.demojize(text)
            text_display.append(text)
            like_count.append(snippet["likeCount"])
        self.df = pd.DataFrame({'comment': text_display, 'likeCount': like_count})

    def seprate_emojis(self):
        emojis = []
        emoji_pattern = re.compile(emoji.get_emoji_regexp())

        # Create a new column to store comments without emojis
        self.df['comment_no_emojis'] = self.df['comment'].apply(lambda x: emoji_pattern.sub('', x))

        for comment in self.df['comment']:
            emojis_found = emoji_pattern.findall(comment)
            emojis.append(emojis_found)
        self.df['emojis'] = emojis  
          

    def clean_data(self):
        self.df.drop(columns=['comment'], inplace=True)
        self.df['comment_no_emojis'] = self.df['comment_no_emojis'].str.replace(r'[^a-zA-Z]+', ' ')

    def _nltk_vader(self):
        # VADER Sentiment Analysis
        sia = SentimentIntensityAnalyzer()
        scores = []
        vader_df = self.df.copy()

        for comment in vader_df['comment_no_emojis']:
            score = sia.polarity_scores(comment)
            scores.append(score)

        scores_df = pd.DataFrame(scores)
        vader_df = vader_df.join(scores_df)

        vader_df['compound'] = vader_df['compound'] * self.LIKE_WEIGHT * vader_df['emojis'].apply(lambda x: len(x) + 1) 
        return vader_df
    
    def _emojis_sentiment(self, df):
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


    def nltk_vader_sentiment_analysis(self):
        vader_df = self._nltk_vader() 
        vader_emojis_score_df = self._emojis_sentiment(vader_df)
        vader_emojis_score_df['total_score'] = vader_emojis_score_df['compound'] + vader_emojis_score_df['emojis_sentiment'] * self.EMOJI_WEIGHT 

        #vader_emojis_score_df['sentiment'] = vader_emojis_score_df['total_score'].apply(lambda x: 'positive' if x > 0 else 'negative')
        #vader_emojis_score_df['sentiment'].value_counts()
        return vader_emojis_score_df['total_score'].mean()


    def runner(self):
        self.get_comments_data()
        self.transform_data()
        self.seprate_emojis()
        self.clean_data()
        return self.nltk_vader_sentiment_analysis()

if __name__ == '__main__':
    sentiment_analysis = SentimentAnalysis()
    print(sentiment_analysis.runner())  

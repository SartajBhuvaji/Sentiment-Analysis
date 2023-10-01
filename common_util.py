import os
import sys
import re
import emoji
import requests
import pandas as pd
from emosent import get_emoji_sentiment_rank
from emosent import get_emoji_sentiment_rank
from dotenv import load_dotenv


class CommonUtil:
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
import os
import sys
import re
import emoji
import requests
import pandas as pd
from emosent import get_emoji_sentiment_rank
from emosent import get_emoji_sentiment_rank
from dotenv import load_dotenv

import os
import sys
import re
import emoji
import requests
import pandas as pd
from dotenv import load_dotenv
from emoji_sentiment import get_emoji_sentiment_rank

class CommonUtil:
    """
    A class that provides utility functions for sentiment analysis on YouTube comments.

    Attributes:
    YOUTUBE_API_KEY (str): The API key for the YouTube Data API v3.
    MAX_RESULTS (str): The maximum number of comments to retrieve from the YouTube API.
    LIKE_WEIGHT (float): The weight to assign to the number of likes a comment has.
    EMOJI_WEIGHT (float): The weight to assign to the sentiment score of emojis in a comment.
    video_id (str): The ID of the YouTube video to retrieve comments from.
    url (str): The URL to retrieve comments from using the YouTube Data API v3.
    data (dict): The JSON response from the YouTube Data API v3 containing comments data.
    df (pandas.DataFrame): A pandas DataFrame containing cleaned and transformed comments data.
    """

    def __init__(self):
        """
        Initializes a new instance of the CommonUtil class.
        """
        load_dotenv()
        self.YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
        self.MAX_RESULTS = os.getenv("MAX_RESULTS")
        self.LIKE_WEIGHT = 0.75
        self.EMOJI_WEIGHT = 0.25
        self.video_id = 'eNZplHsZpvc'
        self.url = f'https://www.googleapis.com/youtube/v3/commentThreads?key={self.YOUTUBE_API_KEY}&textFormat=plainText&part=snippet&videoId={self.video_id}&maxResults={self.MAX_RESULTS}'

    def get_comments_data(self):
        """
        Retrieves comments data from the YouTube Data API v3 and stores it in the data attribute.
        """
        try:
            response = requests.get(self.url)
            response.encoding = 'utf-8'
            self.data = response.json()
        except Exception as e:
            print(e)
            sys.exit(1)  

    def transform_data(self):
        """
        Transforms the comments data in the data attribute into a pandas DataFrame and stores it in the df attribute.
        """
        text_display = []
        like_count = []
        for comment in self.data["items"]:
            snippet = comment["snippet"]["topLevelComment"]["snippet"]
            text = snippet["textDisplay"]
            text_display.append(text)
            like_count.append(snippet["likeCount"])
        self.df = pd.DataFrame({'comment': text_display, 'likeCount': like_count})        

    def seprate_emojis(self):
        """
        Separates emojis from comments in the df attribute and stores them in a new emojis column.
        """
        emojis = []
        emoji_pattern = re.compile(emoji.get_emoji_regexp())

        # Create a new column to store comments without emojis
        self.df['comment_no_emojis'] = self.df['comment'].apply(lambda x: emoji_pattern.sub('', x))

        for comment in self.df['comment']:
            emojis_found = emoji_pattern.findall(comment)
            emojis.append(emojis_found)
        self.df['emojis'] = emojis      

    def clean_data(self):
        """
        Cleans the comments data in the df attribute by removing non-alphabetic characters from the comment_no_emojis column.
        """
        self.df.drop(columns=['comment'], inplace=True)
        self.df['comment_no_emojis'] = self.df['comment_no_emojis'].str.replace(r'[^a-zA-Z]+', ' ')   

    def _emojis_sentiment(self, df):
        """
        Calculates the sentiment score of emojis in the emojis column of the given DataFrame.

        Args:
        df (pandas.DataFrame): The DataFrame containing the emojis column.

        Returns:
        pandas.DataFrame: The input DataFrame with an additional emojis_sentiment column containing the sentiment scores of the emojis.
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
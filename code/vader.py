import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import common_util

class SentimentAnalysis:
    def __init__(self):
        self.common_util_obj = common_util.CommonUtil()
        self.common_util_obj.get_comments_data()
        self.common_util_obj.transform_data()
        self.common_util_obj.seprate_emojis()
        self.common_util_obj.clean_data()

        self.df = self.common_util_obj.df
        self.EMOJI_WEIGHT = float(self.common_util_obj.EMOJI_WEIGHT)

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

        vader_df['compound'] = vader_df['compound'] * self.common_util_obj.LIKE_WEIGHT * vader_df['emojis'].apply(lambda x: len(x) + 1) 
        return vader_df

    def nltk_vader_sentiment_analysis(self):
        vader_df = self._nltk_vader() 
        vader_emojis_score_df = self.common_util_obj._emojis_sentiment(vader_df)
        vader_emojis_score_df['total_score'] = vader_df['compound'] + vader_emojis_score_df['emojis_sentiment'] * self.EMOJI_WEIGHT

        #vader_emojis_score_df['sentiment'] = vader_emojis_score_df['total_score'].apply(lambda x: 'positive' if x > 0 else 'negative')
        #vader_emojis_score_df['sentiment'].value_counts()
        return vader_emojis_score_df['total_score'].mean()

    def runner(self):
        return self.nltk_vader_sentiment_analysis()

if __name__ == '__main__':
    sentiment_analysis = SentimentAnalysis()
    print(sentiment_analysis.runner())  
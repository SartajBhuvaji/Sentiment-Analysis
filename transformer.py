import common_util
import nltk
import pandas as pd
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax


class Transformer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        MODEL = f'cardiffnlp/twitter-roberta-base-sentiment'
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        
        self.common_util_obj = common_util.CommonUtil()
        self.common_util_obj.get_comments_data()
        self.common_util_obj.transform_data()
        self.common_util_obj.seprate_emojis()
        self.common_util_obj.clean_data()

        self.df = self.common_util_obj.df
        #print(self.df.columns) #Index(['likeCount', 'comment_no_emojis', 'emojis'], dtype='object')

    def run_model(self):
        scores = []
        for comment in self.df['comment_no_emojis']:
            score = self.sia.polarity_scores(comment)
            scores.append(score)

        scores_df = pd.DataFrame(scores)
        self.df = self.df.join(scores_df)

        for i in range(len(self.df)):
            encoded_text = self.tokenizer(self.df['comment_no_emojis'][i], return_tensors='pt')
            output = self.model(**encoded_text)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            self.df['compound'][i] = scores[2] - scores[0]

        self.df['compound'] = self.df['compound'] * self.common_util_obj.LIKE_WEIGHT * self.df['emojis'].apply(lambda x: len(x) + 1)
        return self.df

if __name__ == '__main__':
    transformer = Transformer()
    print(transformer.run_model())
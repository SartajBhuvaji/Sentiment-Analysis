#sentiment_mod.py
#from nltk.corpus import movie_reviews
import pickle
import os
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

# load everything
#print current dir
documents_f = open("./bag_of_words/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()


word_features5k_f = open("./bag_of_words/word_feature5k.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

#1
open_file = open("./bag_of_words/originalnaivebayes5k.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()

open_file = open("./bag_of_words/ComplementNB_classifier5k.pickle", "rb")
ComplementNB_classifier = pickle.load(open_file)
open_file.close()

#2
open_file = open("./bag_of_words/MNB_classifier5k.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()

#3
open_file = open("./bag_of_words/BernoulliNB_classifier5k.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()

#4
open_file = open("./bag_of_words/LogisticRegression_classifier5k.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()

#5
open_file = open("./bag_of_words/LinearSVC_classifier5k.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()

 # outputs voted class
voted_classifier = VoteClassifier(
                                  classifier,
                                  ComplementNB_classifier,    
                                  MNB_classifier,
                                  #BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  LinearSVC_classifier)

def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)

def runner(comment):
    sentiment_value, confidence = sentiment(comment)
    print(comment, sentiment_value, confidence)
    return sentiment_value 
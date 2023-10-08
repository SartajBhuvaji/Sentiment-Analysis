import nltk
import random
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB, ComplementNB
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

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


#read text_files     
short_pos = open("positive.txt","r").read() #training
short_neg = open("negative.txt","r").read()

documents = []
# adding posivitive/ negative for value to tuple

for r in short_pos.split('\n'):
    documents.append( (r, "pos") )

for r in short_neg.split('\n'):
    documents.append( (r, "neg") )

#save_pickle
save_documents = open ("documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()

#load_pickle
documents_f = open("documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()


all_words = []

short_pos_words = word_tokenize(short_pos)  # tokenize document by words
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

#arrange according to frequency
all_words = nltk.FreqDist(all_words) # FreqDist is tuple, most to lest

# words as feeature for learning p12
word_features = list(all_words.keys())[:5000]

#save_pickle
word_feature5k = open ("word_feature5k.pickle","wb")
pickle.dump(word_features, word_feature5k)
word_feature5k.close()

#load_pickle
word_features5k_f = open("word_feature5k.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()

def find_features(document):
   # words = set(document) # gives lower accuraacy
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]
random.shuffle(featuresets)

featuresets_f = open ("featuresets.pickle","wb")
pickle.dump(featuresets, featuresets_f)
featuresets_f.close()

#load_pickle
featuresets_f = open("featuresets.pickle", "rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()

# set that we'll train our classifier with
training_set = featuresets[:10000]

# set that we'll test against.
testing_set = featuresets[10000:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)
save_classifier = open ("originalnaivebayes5k.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()


classifier_f = open("originalnaivebayes5k.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:",(nltk.classify.accuracy(MNB_classifier, testing_set))*100)

MNB_classifier5k = open ("MNB_classifier5k.pickle","wb")
pickle.dump(MNB_classifier, MNB_classifier5k)
MNB_classifier5k.close()
#loading_pickle

open_file = open("MNB_classifier5k.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()


ComplementNB_classifier = SklearnClassifier(ComplementNB())
ComplementNB_classifier.train(training_set)
print("ComplementNB_classifier accuracy percent:",(nltk.classify.accuracy(ComplementNB_classifier, testing_set))*100)

ComplementNB_classifier5k = open ("ComplementNB_classifier5k.pickle","wb")
pickle.dump(ComplementNB_classifier, ComplementNB_classifier5k)
ComplementNB_classifier5k.close()

open_file = open("ComplementNB_classifier5k.pickle", "rb")
ComplementNB_classifier = pickle.load(open_file)
open_file.close()


'''
GaussianNB_classifier = SklearnClassifier(GaussianNB())
GaussianNB_classifier.train(training_set)
print("GaussianNB_classifier accuracy percent:",(nltk.classify.accuracy(GaussianNB_classifier, testing_set))*100)
#save_pickle
GaussianNB_classifier5k = open ("GaussianNB_classifier5k.pickle","wb")
pickle.dump(GaussianNB_classifier, GaussianNB_classifier5k)
GaussianNB_classifier5k.close()
#load_pickle
open_file = open("GaussianNB_classifier5k.pickle", "rb")
GaussianNB_classifier = pickle.load(open_file)
open_file.close()
'''

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:",(nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)
#save_pickle
BernoulliNB_classifier5k = open ("BernoulliNB_classifier5k.pickle","wb")
pickle.dump(BernoulliNB_classifier, BernoulliNB_classifier5k)
BernoulliNB_classifier5k.close()
#load_pickle
open_file = open("BernoulliNB_classifier5k.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)
#save_pickle
LogisticRegression_classifier5k = open ("LogisticRegression_classifier5k.pickle","wb")
pickle.dump(LogisticRegression_classifier, LogisticRegression_classifier5k)
LogisticRegression_classifier5k.close()
#load_pickle
open_file = open("LogisticRegression_classifier5k.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)
#save_pickle
LinearSVC_classifier5k = open ("LinearSVC_classifier5k.pickle","wb")
pickle.dump(LinearSVC_classifier, LinearSVC_classifier5k)
LinearSVC_classifier5k.close()
#load_pickle
open_file = open("LinearSVC_classifier5k.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()

voted_classifier = VoteClassifier(classifier,
                                  ComplementNB_classifier,    
                                  MNB_classifier,
                                 #BernoulliNB_classifier, #not good
                                  LogisticRegression_classifier,
                                  LinearSVC_classifier
                                  )

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)
print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:",voted_classifier.confidence(testing_set[0][0])*100)
print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence %:",voted_classifier.confidence(testing_set[1][0])*100)
print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence %:",voted_classifier.confidence(testing_set[2][0])*100)
print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence %:",voted_classifier.confidence(testing_set[3][0])*100)
print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence %:",voted_classifier.confidence(testing_set[4][0])*100)
print("Classification:", voted_classifier.classify(testing_set[5][0]), "Confidence %:",voted_classifier.confidence(testing_set[5][0])*100)

'''
Original Classifier accuracy percent: 72.13855421686746
MNB_classifier accuracy percent: 71.98795180722891
ComplementNB_classifier accuracy percent: 71.98795180722891
BernoulliNB_classifier accuracy percent: 72.13855421686746
LinearSVC_classifier accuracy percent: 71.08433734939759
voted_classifier accuracy percent: 73.3433734939759
Classification: pos Confidence %: 100.0
Classification: pos Confidence %: 100.0
Classification: neg Confidence %: 80.0
Classification: pos Confidence %: 100.0
Classification: neg Confidence %: 100.0
Classification: pos Confidence %: 100.0
'''
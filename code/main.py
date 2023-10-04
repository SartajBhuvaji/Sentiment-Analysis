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

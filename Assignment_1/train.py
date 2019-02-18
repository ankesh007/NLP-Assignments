import numpy as np
import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize,RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from nltk.sentiment.util import mark_negation
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, chi2,f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
import re
import sys
import utils
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from scipy.sparse import vstack

train_file=sys.argv[1]
dev_file=sys.argv[2]
model_file=sys.argv[3]


train_sent,train_y=utils.read_data(train_file)
dev_sent,dev_y=utils.read_data(dev_file)
print(len(train_sent),len(dev_sent))

train_x=utils.clean_data(train_sent)
dev_x=utils.clean_data(dev_sent)

vectorizer = TfidfVectorizer(max_features=1000000,tokenizer = utils.lemmatize_tokenize,analyzer='word',ngram_range=(1,2))
temp=train_x+dev_x
%time train_test_x_tfidf=vectorizer.fit_transform(temp)
train_x_tfidf=train_test_x_tfidf[0:len(train_x)]
dev_x_tfidf=train_test_x_tfidf[len(train_x):]

print(train_x_tfidf.shape)
print(dev_x_tfidf.shape)

y=train_y+dev_y
X=train_test_x_tfidf

model=linear_model.Ridge(alpha=2)
cv = ShuffleSplit(n_splits=6, test_size=0.2, random_state=0)
cv_results = cross_validate(model, X, y, cv=cv,return_train_score=True,return_estimator=True,n_jobs=2,scoring='neg_mean_squared_error')

print(cv_results['test_score'],"Test")
print(cv_results['train_score'],"Train")
print(cv_results['fit_time'],"Time")

pipeline={}
pipeline['TFIDF_Vectoriser']=vectorizer
pipeline['model']=cv_results

utils.dump_pickle(pipeline,model_file)
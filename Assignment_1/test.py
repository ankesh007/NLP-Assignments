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

model_file=sys.argv[1]
test_file=sys.argv[2]
output_file=sys.argv[3]
writer=open(output_file,"w")


test_sent,test_y=utils.read_data(test_file)
print(len(test_sent),len(dev_sent))
test_x=utils.clean_data(test_sent)

pipeline=utils.load_pickle(model_file)

vectorizer = pipeline['TFIDF_Vectoriser']
models = pipeline['model']
test_x_tfidf=vectorizer.transform(test_x)

print(test_x_tfidf.shape)

predictions=None

for model in models['estimator']:
	if predictions is None:
		predictions=model.predict(test_x)
	else:
		predictions+=models.predict(test_x)

predictions/=(1.0*len(models['estimator']))

for pred in predictions:
	print(pred,file=writer)

writer.close()
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import pickle
# from sklearn.naive_bayes import MultinomialNB
# from collections import Counter
# from sklearn.metrics import f1_score
# from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
import sys
import utils
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit

model_file='model.pkl' 
test_file="./data/A1_Data/dev.json" 
output_file="output.txt"

model_file=sys.argv[1]
test_file=sys.argv[2]
output_file=sys.argv[3]


test_sent,test_y=utils.read_data(test_file,label=False)
print(len(test_sent))
test_x,test_y=utils.clean_data(test_sent)

pipeline=utils.load_pickle(model_file)

vectorizer = pipeline['TFIDF_Vectoriser']
models = pipeline['model']
test_x_tfidf=vectorizer.transform(test_x)

print(test_x_tfidf.shape)

predictions=None

for model in models['estimator']:
	if predictions is None:
		predictions=model.predict(test_x_tfidf)
	else:
		predictions+=model.predict(test_x_tfidf)

predictions/=(1.0*len(models['estimator']))
predictions[predictions[:]<1]=1
predictions[predictions[:]>5]=5

print(mean_squared_error(test_y,predictions))
writer=open(output_file,"w")

for pred in predictions:
	print(pred,file=writer)

writer.close()
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
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
import sys
import utils
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit

train_file="./data/A1_Data/train.json"
dev_file="./data/A1_Data/dev.json" 
model_file='model.pkl' 

train_file=sys.argv[1]
dev_file=sys.argv[2]
model_file=sys.argv[3]


train_sent,train_y=utils.read_data(train_file)
dev_sent,dev_y=utils.read_data(dev_file)
print(len(train_sent),len(dev_sent))

print("Cleaning data")
train_x=utils.clean_data(train_sent)
dev_x=utils.clean_data(dev_sent)

vectorizer = TfidfVectorizer(max_features=1000000,tokenizer = utils.lemmatize_tokenize,analyzer='word',ngram_range=(1,2))
temp=train_x+dev_x
print("Transforming data to tdidf")
train_test_x_tfidf=vectorizer.fit_transform(temp)
train_x_tfidf=train_test_x_tfidf[0:len(train_x)]
dev_x_tfidf=train_test_x_tfidf[len(train_x):]

print(train_x_tfidf.shape)
print(dev_x_tfidf.shape)

y=train_y+dev_y
X=train_test_x_tfidf

print("Training Logistic model")
model=LogisticRegression(random_state=0, solver='sag',multi_class='multinomial',n_jobs=-1,verbose=1,max_iter=30)
cv = ShuffleSplit(n_splits=4, test_size=0.25, random_state=0)
cv_results = cross_validate(model, X, y, cv=cv,return_train_score=True,return_estimator=True,n_jobs=1,scoring='neg_mean_squared_error')

print(cv_results['test_score'],"Test")
print(cv_results['train_score'],"Train")
print(cv_results['fit_time'],"Time")

linear_regression_models=[]

for model in cv_results['estimator']:
	pred_x=model.predict_proba(train_x_tfidf)
	pred_dev=model.predict_proba(dev_x_tfidf)
	model=linear_model.Ridge(alpha=2)
	model.fit(pred_x,train_y)
	output=utils.purge(model.predict(pred_dev))
	print(mean_squared_error(output,dev_y))
	linear_regression_models.append(model)

pipeline={}
pipeline['TFIDF_Vectoriser']=vectorizer
pipeline['model_logistic']=cv_results
pipeline['model_regress']=linear_regression_models

utils.dump_pickle(pipeline,model_file)
import pickle
from nltk.stem import WordNetLemmatizer
import json
import re
from nltk.tokenize import word_tokenize
from nltk.sentiment.util import mark_negation
import nltk
import gensim
log=10**5

def load_pickle(filename):
	with open(filename,"rb") as f:
		dump=pickle.load(f)
	return dump

def dump_pickle(dump,filename):
	with open(filename,"wb") as f:
		pickle.dump(dump,f)

def read_data(filename):
    x=[]
    
    file_reader=open(filename,"r")
    for line in file_reader:
        # line=line.encode('utf-8').strip()
        temp=gensim.utils.simple_preprocess(line,min_len=1)
        x.append(temp)
    
    return x

def clean_data(data):
    cleaned_data=[]
    count=0
    for line in data:
        count+=1
        if(count%log==0):
            print(count)
        purge=line
        purge=re.sub("((:\))|(:-\)))","good",purge)
        purge=re.sub("((:D)|(:-\)\)|(:-D)))","very good",purge)
        purge=re.sub("((:\())","bad",purge)
        purge=re.sub("((:p))","tricky",purge)
        purge=re.sub("((,)|(\n))"," ",purge)
        purge = mark_negation(nltk.word_tokenize(purge), double_neg_flip=True, shallow=True)
        new_purge=[]
        for x in purge:
            if x not in new_purge:
                new_purge.append(x)
        purge=new_purge
        purge= " ".join(purge)
        cleaned_data.append(purge)
    return cleaned_data

lemmatizer=WordNetLemmatizer()

def lemmatize(lines):
    new_lines=[]
    lemmatizer=WordNetLemmatizer()
    counter=0
    for line in lines:
        counter+=1
        if(counter%10000==0):
            print(counter)
        new_line=[]
        for word in line:
            new_line.append(lemmatizer.lemmatize(word))
        new_lines.append(new_line)
    return new_lines

def lemmatize_1d(y):
    new_y=[]
    for i in y:
        new_y.append(lemmatizer.lemmatize(i))
    return new_y
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize
import nltk
import gensim

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
        temp=wordpunct_tokenize(line.strip())
        x.append(temp)
    return x

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
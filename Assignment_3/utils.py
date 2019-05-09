import nltk
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import pickle

def greater(word,lim):
    count=0
    for i in word:
        if(i.isdigit()):
            count+=1
    return count>=lim

def read_data(data_path,label=True):
    x=[]
    y=[]
    with open(data_path,'r') as f:
        tokens=[]
        labels=[]
        count=0
        for token in f:
            count+=1
            if(count%10000==0):
                print(count)
            token=token.strip()
            if(token==''):
                x.append(list(tokens))
                y.append(list(labels))
                tokens=[]
                labels=[]
                
            else:
                token=token.split(" ")
                tokens.append(token[0])
                if label:
                    labels.append(token[1])
    return x,y

def get_pos(x):
    new_x=[]
    for sent in x:
        new_x.append(nltk.pos_tag(sent))
    return new_x

def hassec(word):
    return 'sec' in word.lower()

def hashttp(word):
    return 'http' in word.lower()

def hasgen(word,st):
    return st in word.lower()

def hasPfeat(word):
    word=word.lower()
    if 'lac' in word or 'cr ' in word or 'rs' in word or 'lakh' in word or 'crore' in word:
        return True
    return False

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[-4:]': word[-4:],        
        'word[:2]': word[:2],        
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
        'digit>=8': greater(word,8),
        'digit>=3': greater(word,3),
        'hassec': hassec(word),
        'hashttp': hashttp(word),
        'hasPfeat': hasPfeat(word),
        'hasbhk': hasgen(word,'bhk'),
        'hassq': hasgen(word,'sq'),
        'has/': hasgen(word,'/'),
        'has@': hasgen(word,'@'),
        'per': hasgen(word,'per'),
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
#             '-1word[-3:]': word1[-3:],
#             '-1word[-2:]': word1[-2:],
#             '-1word[:2]': word1[:2],        
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            '-1digit>=8': greater(word1,8),
            '-1digit>=3': greater(word1,3),            
            '-1hassec': hassec(word1),
            '-1hashttp': hashttp(word1),
            '-1hasPfeat': hasPfeat(word1),
            '-1hasbhk': hasgen(word1,'bhk'),
            '-1hassq': hasgen(word1,'sq'),        
            '-1has/': hasgen(word1,'/'),
            '-1has@': hasgen(word1,'@'),
            '-1per': hasgen(word1,'per') or ((i>1)and hasgen(sent[i-2][0],'per')),
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
#             '+1word[-3:]': word1[-3:],
#             '+1word[-2:]': word1[-2:],
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            '+1digit>=8': greater(word1,8),
            '+1digit>=3': greater(word1,3),
            '+1hassec': hassec(word1),
            '+1hashttp': hashttp(word1),
            '+1hasPfeat': hasPfeat(word1),
            '+1hasbhk': hasgen(word1,'bhk'),
            '+1hassq': hasgen(word1,'sq'),        
            '+1has/': hasgen(word1,'/'),
            '+1has@': hasgen(word1,'@'),
            '+1per': hasgen(word1,'per') or ((i<len(sent)-2)and hasgen(sent[i+2][0],'per')),
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

def evaluate(crf,test_feat,y_test):
    y_pred = crf.predict(test_feat)
    labels = list(crf.classes_)

    print(metrics.flat_f1_score(y_test, y_pred,average='weighted', labels=labels))
    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )
    print(metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3
    ))

def load_pickle(filename):
	with open(filename,"rb") as f:
		dump=pickle.load(f)
	return dump

def dump_pickle(dump,filename):
	with open(filename,"wb") as f:
		pickle.dump(dump,f)

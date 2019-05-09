import gensim
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import os
import utils
from gensim.models import Word2Vec
import sys
from nltk.tag import pos_tag

data_dir=sys.argv[1]
model_path=sys.argv[2]
pretrained_vectors=sys.argv[3]
context_window=int(sys.argv[4])
print(context_window)

x=[]
for dir,subdir,files in os.walk(data_dir):
    for file in files:
        file_path=os.path.join(dir,file)
        proc=utils.read_data(file_path)
        x+=proc
        print(file_path)
print(len(x))


aux_data={}
def add(w1,w2):
    key=w1+"_"+w2
    key2=w2+"_"+w1
    if key not in aux_data:
        aux_data[key]=0
    if key2 not in aux_data:
        aux_data[key2]=0
    aux_data[key]+=1
    aux_data[key2]+=1

def is_prop_noun(tag):
    if (tag[1]=='NNP' or tag[1]=='NNPS'):
        return True
    return False

counter=0
for line in x:
    if(counter%5000==0):
        print(counter)
    le=len(line)
    pos_tag=nltk.pos_tag(line)
    if(len(line)!=len(pos_tag)):
        print(len(line),len(pos_tag))
        print(line)
        print(pos_tag)
        exit(0)
    counter+=1

    for i in range(le):
        line[i]=line[i].lower()
    for i in range(le):
        w1=line[i]
        for j in range(1,context_window+1,1):
            if i+j<le:
                w2=line[i+j]
                add(w1,w2)
                if is_prop_noun(pos_tag[i]):
                    add('-pro-',w2)
                if is_prop_noun(pos_tag[j]):
                    add(w1,'-pro-')

utils.dump_pickle(dump=aux_data,filename="aux_ds"+str(context_window)+".pkl")
print("Created Aux DS")

def my_train(x,size,epochs,model,model_path,pret=False):
    model_2 = Word2Vec(size=300, min_count=1,workers=1,sg=1)
    model_2.build_vocab(x)
    total_examples = model_2.corpus_count
    print(total_examples)
    print(len(list(model_2.wv.vocab.keys())))
    print(list(model_2.wv.vocab.keys())[0:10])
    total_examples = model_2.corpus_count
    if pret:
        model_2.intersect_word2vec_format(pretrained_vectors, binary=True, lockf=1.0)
    model_2.train(x, total_examples=total_examples, epochs=epochs,report_delay=20)
    utils.dump_pickle(model_2,model_path)

model=None
model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_vectors, binary=True)
print("Loaded Google pretrained_vectors")
my_train(x,300,10,model,model_path,True)

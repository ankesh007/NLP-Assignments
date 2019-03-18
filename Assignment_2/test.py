import gensim
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import os
import utils
from random import shuffle
import time
import sys


eval_data_file=sys.argv[1]
eval_words_file=sys.argv[2]
model_path=sys.argv[3]
context_window=int(sys.argv[4])
print(context_window)

reader=open(eval_words_file,'r')
eval_words_list=[]
for line in reader:
    tok_line=line.strip().split(" ")
    x=len(tok_line)
    for i in range(x):
        tok_line[i]=tok_line[i].split(":")[-1]
    eval_words_list.append(tok_line)
print(len(eval_words_list))
reader.close()

reader=open(eval_data_file,'r')
context_list=[]
y=[]

for line in reader:
    tok_line=line.strip().split("<<target>>")
    context=gensim.utils.simple_preprocess(tok_line[0],min_len=1)
    x=len(context)
    
    y.append(tok_line[1].split("::::")[-1])
    context2=gensim.utils.simple_preprocess(tok_line[1])

    if len(context2)>0:
        context2.pop()
    
    context_list.append(context)
    context_list.append(context2)
print(len(context_list),len(y))
print(context_list[0],y[0])
print(context_list[1],y[0])

def write_file(arr,file):
    with open(file,'w') as f:
        for line in arr:
            st=" ".join([str(x) for x in line])
            print(st,file=f)
    f.close()

def func(func,flag=True,hits=2000,context_win=2,upto=2000):

    mrr=0
    mr=0
    ranking=[]

    if flag==True:
        upto=len(eval_words_list)

    for i in range(upto):
        if i%200==0:
            print(mrr/(i+1),mr*1.0/(i+1),i)
        context1=context_list[2*i]
        context2=context_list[2*i+1]
        x1=len(context1)
        x2=len(context2)
        if x1>context_win:
            context1=context1[x1-context_win:x1]
        if x2>context_win:
            context2=context2[0:context_win]
        
        context=context1+context2
        word=y[i]
        val=func(context,topn=hits)
        local_rank=[]
        mapper={}
        mapper_count={}
        for w in eval_words_list[i]:
            mapper[w]=0
            if w not in mapper_count:
                mapper_count[w]=0
            mapper_count[w]+=1

        count=1
        if val is not None:
            for j in range(len(val)):
                if val[j][0] in mapper:
                    mapper[val[j][0]]=count
                    count+=mapper_count[val[j][0]]

        temp=[]
        for w in eval_words_list[i]:
            if mapper[w]==0:
                temp.append(w)
        shuffle(temp)
        
        for w in temp:
            if mapper[w]==0:
                mapper[w]=count
                count+=mapper_count[w]

        for w in eval_words_list[i]:
            local_rank.append(mapper[w])
            mapper[w]+=1
        ranking.append(local_rank)

        if word not in mapper:
            continue
        else:        
            mr+=(mapper[word]-1)
            mrr+=1.0/(mapper[word]-1)
    print(mrr*1.0/len(eval_words_list),mr*1.0/len(eval_words_list))
    return ranking

def run_eval(path,flag=False,hits=2000,context_win=1):
    model_path=path
    model=utils.load_pickle(model_path)
    
    x=time.time()
    t=func(model.predict_output_word,flag=flag,hits=hits,context_win=context_win)
    print(time.time()-x)
    return t

def getC(w1,w2,ds):
    key=w1+"_"+w2
    if key in ds:
        return ds[key]
    return 0
    
def counter(ds,word,context,begin):
    x=len(context)
    count=0.0
    wt=1.0

    if begin==1:
        for i in range(min(x,context_window)):
            count+=getC(word,context[i],ds)*wt
            wt/=2
    else:
        for i in range(x-1,max(0,x-context_window-1),-1):
            count+=getC(word,context[i],ds)*wt
            wt/=2
    return count

def rerank(rank_list,context_list,eval_words_list,rr=30):
    le=len(rank_list)
    aux_ds=utils.load_pickle(filename="aux_ds"+str(context_window)+".pkl")
    for i in range(le):
        imp_words=[]
        lee=len(rank_list[i])
        for j in range(lee):
            if rank_list[i][j]<rr:
                imp_words.append((eval_words_list[i][j],rank_list[i][j],j))
        new_lis=[]
        for w in imp_words:
            word=w[0]
            rank=w[1]
            pos=w[2]
            count=0
            count+=counter(aux_ds,word,context_list[2*i],0)
            count+=counter(aux_ds,word,context_list[2*i+1],1)
            new_lis.append((count,-rank,pos))
        new_lis=sorted(new_lis,reverse=True)
        c=0
        for entry in new_lis:
            c+=1
            pos=entry[2]
            rank_list[i][pos]=c

ranking=run_eval(model_path,flag=True,hits=2000,context_win=2)
print(len(ranking))
rerank(ranking,context_list=context_list,eval_words_list=eval_words_list,rr=1000)
write_file(ranking,str(context_window)+'output.txt')
import nltk
import sklearn_crfsuite
import sys
from utils import *

x_path=sys.argv[1]
output_path=sys.argv[2]

crf=load_pickle('model.pt')
flag=False

x_test,y_test=read_data(x_path,label=flag)
test_pos=get_pos(x_test)
test_feat=[sent2features(x) for x in test_pos]

y=crf.predict(test_feat)

if flag:
    evaluate(crf,test_feat,y_test)

file_writer=open(output_path,'w')

for i in range(len(x_test)):
    for j in range(len(x_test[i])):
        token=x_test[i][j]
        y_p=y[i][j]
        print(token+" "+y_p,file=file_writer)
    print(file=file_writer)

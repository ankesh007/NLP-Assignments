import nltk
import sklearn_crfsuite
import sys
from utils import *

path=sys.argv[1]

x_train,y_train=read_data(path)
train_pos=get_pos(x_train)
train_feat=[sent2features(x) for x in train_pos]

crf = sklearn_crfsuite.CRF(
    algorithm='pa',
#     c1=0.1,
#     c2=0.1,
    max_iterations=25,
    all_possible_transitions=True,
)

crf.fit(train_feat, y_train)

evaluate(crf,train_feat,y_train)
dump_pickle(crf,'model.pt')
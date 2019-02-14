import pickle

def load_pickle(filename):
	with open(filename,"rb") as f:
		dump=pickle.load(f)
	return dump

def dump_pickle(dump,filename):
	with open(filename,"wb") as f:
		pickle.dump(dump,f)

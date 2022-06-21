import numpy as np

def nml_sentence(seq_len, snts):
	ret = np.zeros((len(snts),seq_len),dtype=np.float32)
	for ind, snt in enumerate(snts):
		ret[ind][:len(snt)] = np.array(snt[:seq_len], dtype=np.float32)
	return ret

def nml_label(seq_len, labels):
	'''
	labels: 
		list(sentences)
	sentences:
		list(vector labels)
	
	Length of senences is not compatible,
	this method nomalize the length of sentence
	'''
	ret = np.zeros((len(labels), seq_len, len(labels[0][0])), dtype=np.float32)
	for ind, label in enumerate(labels):
		ret[ind][:len(label)] = np.array(label[:seq_len], dtype=np.float32)
	return ret
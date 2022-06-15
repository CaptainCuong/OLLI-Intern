import numpy as np

def nml_sentence(seq_len, snts):
	ret = np.zeros((len(snts),seq_len),dtype=int)
	for ind, snt in enumerate(snts):
		ret[ind][:len(snt)] = np.array(snt[:seq_len], dtype=int)
	print(ret.dtype)
	return ret
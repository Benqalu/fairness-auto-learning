from math import log
import numpy as np

def ranking(pmat,max_iter=100000):

	if type(pmat)==list:
		pmat=np.array(pmat)
	n=pmat.shape[0]
	wins=np.sum(pmat,axis=0)
	params=np.ones(n,dtype=float)
	for _ in range(max_iter):
		tiled=np.tile(params,(n,1))
		combined=1.0/(tiled+tiled.T)
		np.fill_diagonal(combined,0)
		nxt=wins/np.sum(combined,axis=0)
		nxt=nxt/np.mean(nxt)
		tmp=np.linalg.norm(nxt-params,ord=np.inf)
		if tmp<1e-5:
			return nxt
		params=nxt
	raise RuntimeError('did not converge')

if __name__=='__main__':
	a=np.array([[0.0,0.1,0.2],[0.9,0.0,0.6],[0.8,0.4,0.0]])
	print(ranking(pmat=a))
import os
import matplotlib
from matplotlib import pyplot as plt
font = {'size':18}

matplotlib.rc('font', **font)

def draw(fname):

	f=open(fname)
	apc=eval(f.readline())
	f.close()

	f=open('knn_'+fname)
	knn=eval(f.readline())
	f.close()

	x=[i for i in range(1,11)]
	legends=[]
	for alpha in [0.3,0.5,0.7]:
		plt.clf()
		z='Alpha=%.1f'%alpha

		plt.plot(x,apc[z],linestyle='-',color='black')
		legends.append('Similarity-free')
		plt.plot(x,knn[z],linestyle='--',color='black')
		legends.append('Similarity-based')

		plt.xlabel('$k$')
		plt.xticks([1,2,3,4,5,6,7,8,9,10])
		plt.ylabel('Performance Error')
		plt.legend(legends)
		plt.tight_layout()
		plt.savefig('./figures/'+fname.split('.')[0].replace('dictformat_','')+'_alpha%02d.pdf'%(int(alpha*10)))


fnames=os.listdir('.')
for fname in fnames:
	if 'knn' in fname:
		continue
	if '.txt' not in fname:
		continue
	draw(fname)
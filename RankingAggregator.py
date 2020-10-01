from math import log
import numpy as np
import pickle as pk

class RankingAggregator(object):

	def __init__(self, datafile):
		self._all_accuracy_metric=['Acc','BAcc']
		self._all_fairness_metric=['AOD', 'DI', 'EOD', 'SPD', 'TI', 'UF']
		self._matrix_truth=[]
		self._matrix_prediction=[]
		self._data=None
		self._algorithm_tags=None

		self._load_data(datafile=datafile)
		self._load_algorithm_tags()

	def _load_data(self,datafile):
		data=pk.load(open(datafile,'rb'))
		for i in range(0,len(data)):
			for metric in data[i]['gt']:
				if metric=='DI' or metric in self._all_accuracy_metric:
					best_value=1.0
				else:
					best_value=0.0
				data[i]['gt'][metric]=np.array(data[i]['gt'][metric])
				data[i]['gt'][metric]=-abs(data[i]['gt'][metric]-best_value)
				data[i]['pred'][metric]=np.array(data[i]['pred'][metric])
		self._data=data

	def get_true_values(self,metric):
		ret=[]
		for i in range(0,len(self._data)):
			ret.append(self._data[i]['gt'][metric])
		return ret

	def _load_algorithm_tags(self):
		# tags=[]
		# for i in range(5):
		# 	for j in range(3):
		# 		for k in range(4):
		# 			tags.append((i,j,k))
		# 
		self._algorithm_tags=self._data[0]['methods']
		
	def get_algorithm_tags(self):
		return self._algorithm_tags

	def _get_matrix_truth(self,data,metric):
		score=data['gt'][metric]
		n=len(score)
		ret=[]
		for i in range(0,n):
			ret.append([])
			for j in range(0,n):
				if i==j:
					ret[-1].append(0.5)
					continue
				if score[i]>score[j]:
					ret[-1].append(1.0)
				elif score[i]<score[j]:
					ret[-1].append(0.0)
				else:
					ret[-1].append(0.5)
		return np.array(ret)

	def _get_matrix_prediction(self,data,metric):
		return data['pred'][metric]

	def _pairwised_ranking(self,pmat,max_iter=100000):
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

	def predicted_rankings(self,metric):
		ret=[]
		for i in range(0,len(self._data)):
			matrix=self._get_matrix_prediction(data=self._data[i],metric=metric)
			ranking_score=self._pairwised_ranking(pmat=matrix).tolist()
			ranking=[x for _,x in sorted(zip(ranking_score,self._algorithm_tags))]
			ret.append(ranking)
		return ret

	def true_rankings(self,metric):
		ret=[]
		for i in range(0,len(self._data)):
			ranking_score=self._data[i]['gt'][metric].tolist()
			ranking=[x for _,x in sorted(zip(ranking_score,self._algorithm_tags),reverse=True)]
			ret.append(ranking)
		return ret

	def predicted_rankings_mix(self,accuracy_metric,fairness_metric,alpha,beta):
		ret=[]
		for i in range(0,len(self._data)):
			matrix_accuracy=self._get_matrix_prediction(data=self._data[i],metric=accuracy_metric)
			matrix_fairness=self._get_matrix_prediction(data=self._data[i],metric=fairness_metric)
			matrix_performance=alpha*matrix_accuracy+beta*matrix_fairness
			ranking_score=self._pairwised_ranking(pmat=matrix_performance).tolist()
			ranking=[x for _,x in sorted(zip(ranking_score,self._algorithm_tags))]
			ret.append(ranking)
		return ret

	def true_rankings_mix(self,accuracy_metric,fairness_metric,alpha,beta):
		ret=[]
		for i in range(0,len(self._data)):
			matrix_accuracy=self._get_matrix_truth(data=self._data[i],metric=accuracy_metric)
			matrix_fairness=self._get_matrix_truth(data=self._data[i],metric=fairness_metric)
			matrix_performance=alpha*matrix_accuracy+beta*matrix_fairness
			ranking_score=self._pairwised_ranking(pmat=matrix_performance).tolist()
			ranking=[x for _,x in sorted(zip(ranking_score,self._algorithm_tags))]
			ret.append(ranking)
		return ret


if __name__=='__main__':
	agg=RankingAggregator(datafile='./data/dictformat_compas_race_to_adult_race.pkl')
	rankings_pred=agg.predicted_rankings_mix(
		accuracy_metric='Acc',
		fairness_metric='DI',
		alpha=1.0,
		beta=1.0
	)
	rankings_true=agg.true_rankings_mix(
		accuracy_metric='Acc',
		fairness_metric='DI',
		alpha=1.0,
		beta=1.0
	)
	for item in rankings_true:
		print(item)
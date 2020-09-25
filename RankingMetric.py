from scipy.stats import kendalltau

class RankingMetric(object):
	def __init__(self, true, pred):
		self._true=true
		self._pred=pred

	def tau_distance(self):
		ranking={}
		index_true=list(range(len(self._true)))
		for i in range(len(self._true)):
			ranking[self._true[i]]=index_true[i]
		index_pred=[]
		for item in self._pred:
			index_pred.append(ranking[item])
		return (1-kendalltau(index_pred,index_true)[0])/2

	def k_coverage(self,k):
		true_top=set(self._true[:k])
		pred_top=set(self._pred[:k])
		return len(true_top&pred_top)/k
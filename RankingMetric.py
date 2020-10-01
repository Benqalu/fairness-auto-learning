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

	def k_minimum_difference(self,k,true_values,methods=None):
		value_dict={}
		if methods is None:
			c=0
			for i in range(5):
				for j in range(3):
					for l in range(4):
						value_dict[(i,j,l)]=true_values[c]
						c+=1
		else:
			for i in range(0,len(methods)):
				value_dict[methods[i]]=true_values[i]
				
		best_value=max(true_values)

		min_difference=float('inf')
		for i in range(0,k):
			diff=abs(value_dict[self._pred[i]]-best_value)
			if diff<min_difference:
				min_difference=diff

		return min_difference
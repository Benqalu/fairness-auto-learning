import os
import numpy as np
from RankingMetric import RankingMetric
from RankingAggregator import RankingAggregator

def parse(fname):
	agg=RankingAggregator(datafile=fname)
	alphas=[0.3,0.5,0.7]
	betas=[1.0-alphas[i] for i in range(0,len(alphas))]
	
	for alpha, beta in zip(alphas,betas):
		tau_distances=[]
		for accuracy_metric in agg._all_accuracy_metric:
			for fairness_metric in agg._all_fairness_metric:
				rankings_pred=agg.predicted_rankings(
					accuracy_metric=accuracy_metric,
					fairness_metric=fairness_metric,
					alpha=alpha,
					beta=beta
				)
				rankings_true=agg.true_rankings(
					accuracy_metric=accuracy_metric,
					fairness_metric=fairness_metric,
					alpha=alpha,
					beta=beta
				)
				for i in range(len(rankings_pred)):
					pred=rankings_pred[i]
					true=rankings_true[i]
					metric=RankingMetric(pred=pred,true=true)
					tau_distances.append(metric.tau_distance())
		f=open('./results/ranking_aggregation_partial.txt','a')
		f.write(
			'%s'%(fname.split('/')[-1].split('.')[0])+'\t'+'(%.2f, %s)'%(alpha,'All')+'\t'+'(%.2f, %s)'%(beta,'All')+'\t'+'%.4f'%(np.mean(tau_distances))+'\n'
		)
		f.close()
		print((float('%.2f'%alpha),'All'),'\t',(float('%.2f'%beta),'All'),'\t',np.mean(tau_distances))


if __name__=='__main__':
	fnames=os.listdir('./data')
	for fname in fnames:
		if '.pkl' not in fname:
			continue
		print(fname)
		parse(fname='./data/'+fname)
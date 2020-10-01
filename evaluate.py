import os,sys
import numpy as np
from RankingMetric import RankingMetric
from RankingAggregator import RankingAggregator

def k_minimum_difference(fname):
	report={}
	agg=RankingAggregator(datafile=fname)
	for metric in agg._all_accuracy_metric+agg._all_fairness_metric:
		rankings_pred=agg.predicted_rankings(metric=metric)
		values=agg.get_true_values(metric=metric)
		diffs=np.zeros(10)
		for i in range(0,len(values)):
			ranking_metric=RankingMetric(pred=rankings_pred[i],true=None)
			res=[]
			for k in range(1,11):
				tmp=ranking_metric.k_minimum_difference(k=k,true_values=values[i],methods=agg.get_algorithm_tags())
				res.append(tmp)
			diffs+=np.array(res)
		diffs/=len(values)
		print(metric,end='\t')
		for item in diffs:
			print(item,end='\t')
		print()
		report[metric]=diffs.tolist()

	for alpha in [i*0.1 for i in range(1,10)]:

		alpha=float('%.1f'%alpha)

		res_accs=np.zeros(10)
		for accs in agg._all_accuracy_metric:
			res_accs+=report[accs]
		res_accs/=len(agg._all_accuracy_metric)

		res_fais=np.zeros(10)
		for fais in agg._all_fairness_metric:
			res_fais+=report[fais]
		res_fais/=len(agg._all_fairness_metric)

		res=alpha*res_accs+(1-alpha)*res_fais

		print('Alpha=%.1f'%alpha,end='\t')
		for item in res:
			print(item,end='\t')
		print()

		report['Alpha=%.1f'%alpha]=res.tolist()

	print()

	name=fname.split('/')[-1].split('.')[0]
	f=open('./results/kmd/'+name+'.txt','w')
	f.write(str(report))
	f.close()

def tau_distance(fname):

	report={}

	agg=RankingAggregator(datafile=fname)
	alphas=[0.3,0.5,0.7]
	for accuracy_metric in agg._all_accuracy_metric:
		for fairness_metric in agg._all_fairness_metric:
			report[(accuracy_metric,fairness_metric)]={}
			for alpha in alphas:
				tau_distances=[]
				rankings_pred=agg.predicted_rankings_mix(
					accuracy_metric=accuracy_metric,
					fairness_metric=fairness_metric,
					alpha=alpha,
					beta=1-alpha,
				)
				rankings_true=agg.true_rankings_mix(
					accuracy_metric=accuracy_metric,
					fairness_metric=fairness_metric,
					alpha=alpha,
					beta=1-alpha,
				)
				for i in range(len(rankings_pred)):
					pred=rankings_pred[i]
					true=rankings_true[i]
					metric=RankingMetric(pred=pred,true=true)
					tau_dis=metric.tau_distance()
					tau_distances.append(tau_dis)

				report[(accuracy_metric,fairness_metric)][alpha]=tau_distances

				print((float('%.2f'%alpha),accuracy_metric),'\t',(float('%.2f'%(1-alpha)),fairness_metric),'\t',np.mean(tau_distances))

		name=fname.split('/')[-1].split('.')[0]
		f=open('./results/mix_ranking/%s.txt'%name,'w')
		f.write(str(report))
		f.close()

if __name__=='__main__':
	fnames=os.listdir('./data')
	for fname in fnames:
		if '.pkl' not in fname:
			continue
		print(fname)
		# tau_distance(fname='./data/'+fname)
		k_minimum_difference(fname='./data/'+fname)
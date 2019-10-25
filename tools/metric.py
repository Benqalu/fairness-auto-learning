def get_metric_reports(true_dataset,classfied_dataset,privileged_groups,unprivileged_groups):

	mirror_dataset=classfied_dataset.copy(deepcopy=True)
	mirror_dataset.labels=copy.deepcopy(true_dataset.labels)

	metric=ClassificationMetric(
		dataset=mirror_dataset,
		classified_dataset=classfied_dataset,
		unprivileged_groups=unprivileged_groups,
		privileged_groups=privileged_groups)
	#Measuring unfairness end
	
	report=OrderedDict()
	report['TPR']=metric.true_positive_rate()
	report['TNR']=metric.true_negative_rate()
	report['FPR']=metric.false_positive_rate()
	report['FNR']=metric.false_negative_rate()
	report['Balanced_Acc']=0.5*(report['TPR']+report['TNR'])
	report['Acc']=metric.accuracy()
	report["Statistical parity difference"]=metric.statistical_parity_difference()
	report["Disparate impact"]=metric.disparate_impact()
	report["Equal opportunity difference"]=metric.equal_opportunity_difference()
	report["Average odds difference"]=metric.average_odds_difference()
	report["Theil index"]=metric.theil_index()
	report["United Fairness"]=metric.generalized_entropy_index()

	return report
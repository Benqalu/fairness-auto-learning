from pyitlib import discrete_random_variable as drv

def get_entropy(dataset,sensitive_attr,top_n=5):

	sensitive_index=dataset.feature_names.index(sensitive_attr)

	res=[]
	res.append(drv.entropy(dataset.features[:,sensitive_index]))
	corr=[]
	corr_condi=[]

	for i in range(0,dataset.features.shape[1]):
		if i==sensitive_index:
			continue
		corr.append(drv.entropy(dataset.features[:,i]))
		corr_condi.append(drv.entropy_conditional(dataset.features[:,sensitive_index],dataset.features[:,i]))

	corr.sort(reverse=True)
	corr_condi.sort(reverse=True)

	res+=corr[:5]
	res+=corr[-5:]

	res+=corr_condi[:5]
	res+=corr_condi[-5:]

	res+=[drv.entropy(dataset.labels[:,0])]

	res+=[drv.entropy_conditional(dataset.features[:,sensitive_index],dataset.labels[:,0])]

	for i in range(0,len(res)):
		res[i]=float(res[i])

	return res

from pyitlib import discrete_random_variable as drv

def get_entropy(dataset,sensitive_attr,top_n=5):

	sensitive_index=dataset.feature_names.index(sensitive_attr)

	res=[]

	#Independent entropy
	res.append(drv.entropy(dataset.features[:,sensitive_index]))
	res.append(drv.entropy(dataset.labels[:,0]))
	entropy_feats=[]
	for i in range(0,dataset.features.shape[1]):
		if i==sensitive_index:
			continue
		entropy_feats.append(drv.entropy(dataset.features[:,i]))
	entropy_feats.sort(reverse=True)
	res+=entropy_feats[:5]
	res+=entropy_feats[-5:]
	#Independent entropy

	#Cross entropy
	res.append(drv.entropy_conditional(dataset.features[:,sensitive_index],dataset.labels[:,0]))
	res.append(drv.entropy_conditional(dataset.labels[:,0],dataset.features[:,sensitive_index]))

	cross_entropy_A=[]
	cross_entropy_B=[]
	for i in range(0,dataset.features.shape[1]):
		if i==sensitive_index:
			continue
		cross_entropy_A.append(drv.entropy_conditional(dataset.features[:,sensitive_index],dataset.features[:,i]))
		cross_entropy_B.append(drv.entropy_conditional(dataset.features[:,i],dataset.features[:,sensitive_index]))
	cross_entropy_A.sort(reverse=True)
	cross_entropy_B.sort(reverse=True)
	res+=cross_entropy_A[:5]
	res+=cross_entropy_A[-5:]
	res+=cross_entropy_B[:5]
	res+=cross_entropy_B[-5:]
	
	cross_entropy_A=[]
	cross_entropy_B=[]
	for i in range(0,dataset.features.shape[1]):
		if i==sensitive_index:
			continue
		cross_entropy_A.append(drv.entropy_conditional(dataset.labels[:,0],dataset.features[:,i]))
		cross_entropy_B.append(drv.entropy_conditional(dataset.features[:,i],dataset.labels[:,0]))
	cross_entropy_A.sort(reverse=True)
	cross_entropy_B.sort(reverse=True)
	res+=cross_entropy_A[:5]
	res+=cross_entropy_A[-5:]
	res+=cross_entropy_B[:5]
	res+=cross_entropy_B[-5:]
	#Cross entropy

	for i in range(0,len(res)):
		res[i]=float(res[i])
	
	return res

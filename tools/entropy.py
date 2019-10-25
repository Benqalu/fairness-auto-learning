from scipy.stats import entropy

def get_entropy(dataset,sensitive_index):
	print(entropy(dataset.features[:,sensitive_index]))e
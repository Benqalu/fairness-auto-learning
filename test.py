import sys,os,warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.simplefilter(action='ignore', category=Warning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german

from classfiers.adversarial_debiasing import AdversarialDebiasing
from classfiers.plain_model import PlainModel

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score

from IPython.display import Markdown, display
import matplotlib.pyplot as plt

import tensorflow as tf

def test_German_Adversarial():

	dataset_orig = load_preproc_data_german()

	privileged_groups = [{'sex': 1}]
	unprivileged_groups = [{'sex': 0}]

	dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

	print("#### Training Dataset shape")
	print(dataset_orig_train.features.shape)
	print("#### Favorable and unfavorable labels")
	print(dataset_orig_train.favorable_label, dataset_orig_train.unfavorable_label)
	print("#### Protected attribute names")
	print(dataset_orig_train.protected_attribute_names)
	print("#### Privileged and unprivileged protected attribute values")
	print(dataset_orig_train.privileged_protected_attributes, dataset_orig_train.unprivileged_protected_attributes)
	print("#### Dataset feature names")
	print(dataset_orig_train.feature_names)

	metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
											unprivileged_groups=unprivileged_groups,
											privileged_groups=privileged_groups)
	print("#### Original training dataset")
	print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())
	metric_orig_test = BinaryLabelDatasetMetric(dataset_orig_test, 
												unprivileged_groups=unprivileged_groups,
												privileged_groups=privileged_groups)
	print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_test.mean_difference())

	min_max_scaler = MaxAbsScaler()
	dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
	dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)
	metric_scaled_train = BinaryLabelDatasetMetric(dataset_orig_train, 
								unprivileged_groups=unprivileged_groups,
								privileged_groups=privileged_groups)
	print("#### Scaled dataset - Verify that the scaling does not affect the group label statistics")
	print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_scaled_train.mean_difference())
	metric_scaled_test = BinaryLabelDatasetMetric(dataset_orig_test, 
								unprivileged_groups=unprivileged_groups,
								privileged_groups=privileged_groups)
	print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_scaled_test.mean_difference())

	sess = tf.Session()
	plain_model = AdversarialDebiasing(
							privileged_groups = privileged_groups,
							unprivileged_groups = unprivileged_groups,
							scope_name='plain_classifier',
							num_epochs=500,
							debias=False,
							sess=sess)
	plain_model.fit(dataset_orig_train)

	dataset_nodebiasing_train = plain_model.predict(dataset_orig_train)
	dataset_nodebiasing_test = plain_model.predict(dataset_orig_test)

	print("#### Plain model - without debiasing - dataset metrics")
	metric_dataset_nodebiasing_train = BinaryLabelDatasetMetric(dataset_nodebiasing_train, 
												unprivileged_groups=unprivileged_groups,
												privileged_groups=privileged_groups)

	print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_train.mean_difference())

	metric_dataset_nodebiasing_test = BinaryLabelDatasetMetric(dataset_nodebiasing_test, 
												unprivileged_groups=unprivileged_groups,
												privileged_groups=privileged_groups)

	print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_test.mean_difference())

	print("#### Plain model - without debiasing - classification metrics")
	classified_metric_nodebiasing_test = ClassificationMetric(dataset_orig_test, 
													dataset_nodebiasing_test,
													unprivileged_groups=unprivileged_groups,
													privileged_groups=privileged_groups)
	print("Test set: Classification accuracy = %f" % classified_metric_nodebiasing_test.accuracy())
	TPR = classified_metric_nodebiasing_test.true_positive_rate()
	TNR = classified_metric_nodebiasing_test.true_negative_rate()
	bal_acc_nodebiasing_test = 0.5*(TPR+TNR)
	print("Test set: Balanced classification accuracy = %f" % bal_acc_nodebiasing_test)
	print("Test set: Disparate impact = %f" % classified_metric_nodebiasing_test.disparate_impact())
	print("Test set: Equal opportunity difference = %f" % classified_metric_nodebiasing_test.equal_opportunity_difference())
	print("Test set: Average odds difference = %f" % classified_metric_nodebiasing_test.average_odds_difference())
	print("Test set: Theil_index = %f" % classified_metric_nodebiasing_test.theil_index())

	sess.close()
	tf.reset_default_graph()
	sess = tf.Session()

	debiased_model = AdversarialDebiasing(
						privileged_groups = privileged_groups,
						unprivileged_groups = unprivileged_groups,
						scope_name='debiased_classifier',
						num_epochs=500,
						debias=True,
						sess=sess)
	
	debiased_model.fit(dataset_orig_train)

	dataset_debiasing_train = debiased_model.predict(dataset_orig_train)
	dataset_debiasing_test = debiased_model.predict(dataset_orig_test)

	# Metrics for the dataset from plain model (without debiasing)
	print("#### Plain model - without debiasing - dataset metrics")
	print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_train.mean_difference())
	print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_test.mean_difference())

	# Metrics for the dataset from model with debiasing
	print("#### Model - with debiasing - dataset metrics")
	metric_dataset_debiasing_train = BinaryLabelDatasetMetric(dataset_debiasing_train, 
												unprivileged_groups=unprivileged_groups,
												privileged_groups=privileged_groups)

	print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_debiasing_train.mean_difference())

	metric_dataset_debiasing_test = BinaryLabelDatasetMetric(dataset_debiasing_test, 
												unprivileged_groups=unprivileged_groups,
												privileged_groups=privileged_groups)

	print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_debiasing_test.mean_difference())


	print("#### Plain model - without debiasing - classification metrics")
	print("Test set: Classification accuracy = %f" % classified_metric_nodebiasing_test.accuracy())
	TPR = classified_metric_nodebiasing_test.true_positive_rate()
	TNR = classified_metric_nodebiasing_test.true_negative_rate()
	bal_acc_nodebiasing_test = 0.5*(TPR+TNR)
	print("Test set: Balanced classification accuracy = %f" % bal_acc_nodebiasing_test)
	print("Test set: Disparate impact = %f" % classified_metric_nodebiasing_test.disparate_impact())
	print("Test set: Equal opportunity difference = %f" % classified_metric_nodebiasing_test.equal_opportunity_difference())
	print("Test set: Average odds difference = %f" % classified_metric_nodebiasing_test.average_odds_difference())
	print("Test set: Theil_index = %f" % classified_metric_nodebiasing_test.theil_index())



	print("#### Model - with debiasing - classification metrics")
	classified_metric_debiasing_test = ClassificationMetric(dataset_orig_test, 
													dataset_debiasing_test,
													unprivileged_groups=unprivileged_groups,
													privileged_groups=privileged_groups)
	print("Test set: Classification accuracy = %f" % classified_metric_debiasing_test.accuracy())
	TPR = classified_metric_debiasing_test.true_positive_rate()
	TNR = classified_metric_debiasing_test.true_negative_rate()
	bal_acc_debiasing_test = 0.5*(TPR+TNR)
	print("Test set: Balanced classification accuracy = %f" % bal_acc_debiasing_test)
	print("Test set: Disparate impact = %f" % classified_metric_debiasing_test.disparate_impact())
	print("Test set: Equal opportunity difference = %f" % classified_metric_debiasing_test.equal_opportunity_difference())
	print("Test set: Average odds difference = %f" % classified_metric_debiasing_test.average_odds_difference())
	print("Test set: Theil_index = %f" % classified_metric_debiasing_test.theil_index())

def test_German_Plain():

	dataset_orig = load_preproc_data_german()

	privileged_groups = [{'sex': 1}]
	unprivileged_groups = [{'sex': 0}]

	dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

	print("#### Training Dataset shape")
	print(dataset_orig_train.features.shape)
	print("#### Favorable and unfavorable labels")
	print(dataset_orig_train.favorable_label, dataset_orig_train.unfavorable_label)
	print("#### Protected attribute names")
	print(dataset_orig_train.protected_attribute_names)
	print("#### Privileged and unprivileged protected attribute values")
	print(dataset_orig_train.privileged_protected_attributes, dataset_orig_train.unprivileged_protected_attributes)
	print("#### Dataset feature names")
	print(dataset_orig_train.feature_names)

	metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
											unprivileged_groups=unprivileged_groups,
											privileged_groups=privileged_groups)
	print("#### Original training dataset")
	print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())
	metric_orig_test = BinaryLabelDatasetMetric(dataset_orig_test, 
												unprivileged_groups=unprivileged_groups,
												privileged_groups=privileged_groups)
	print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_test.mean_difference())

	min_max_scaler = MaxAbsScaler()
	dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
	dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)
	metric_scaled_train = BinaryLabelDatasetMetric(dataset_orig_train, 
								unprivileged_groups=unprivileged_groups,
								privileged_groups=privileged_groups)
	print("#### Scaled dataset - Verify that the scaling does not affect the group label statistics")
	print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_scaled_train.mean_difference())
	metric_scaled_test = BinaryLabelDatasetMetric(dataset_orig_test, 
								unprivileged_groups=unprivileged_groups,
								privileged_groups=privileged_groups)
	print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_scaled_test.mean_difference())

	sess = tf.Session()
	plain_model = PlainModel(
				privileged_groups = privileged_groups,
				unprivileged_groups = unprivileged_groups,
				scope_name='plain_classifier',
				num_epochs=500,
				sess=sess)
	plain_model.fit(dataset_orig_train)

	dataset_nodebiasing_train = plain_model.predict(dataset_orig_train)
	dataset_nodebiasing_test = plain_model.predict(dataset_orig_test)

	print("Plain model, no unfairness")
	metric_dataset_nodebiasing_train = BinaryLabelDatasetMetric(dataset_nodebiasing_train, 
												unprivileged_groups=unprivileged_groups,
												privileged_groups=privileged_groups)

	print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_train.mean_difference())

	metric_dataset_nodebiasing_test = BinaryLabelDatasetMetric(dataset_nodebiasing_test, 
												unprivileged_groups=unprivileged_groups,
												privileged_groups=privileged_groups)

	print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_test.mean_difference())

	print("#### Plain model - without debiasing - classification metrics")
	classified_metric_nodebiasing_test = ClassificationMetric(dataset_orig_test, 
													dataset_nodebiasing_test,
													unprivileged_groups=unprivileged_groups,
													privileged_groups=privileged_groups)
	print("Test set: Classification accuracy = %f" % classified_metric_nodebiasing_test.accuracy())
	TPR = classified_metric_nodebiasing_test.true_positive_rate()
	TNR = classified_metric_nodebiasing_test.true_negative_rate()
	bal_acc_nodebiasing_test = 0.5*(TPR+TNR)
	print("Test set: Balanced classification accuracy = %f" % bal_acc_nodebiasing_test)
	print("Test set: Disparate impact = %f" % classified_metric_nodebiasing_test.disparate_impact())
	print("Test set: Equal opportunity difference = %f" % classified_metric_nodebiasing_test.equal_opportunity_difference())
	print("Test set: Average odds difference = %f" % classified_metric_nodebiasing_test.average_odds_difference())
	print("Test set: Theil_index = %f" % classified_metric_nodebiasing_test.theil_index())

	sess.close()
	tf.reset_default_graph()

if __name__=='__main__':
	# test_German_Adversarial()
	test_German_Plain()

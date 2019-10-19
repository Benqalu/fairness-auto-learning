import sys
import numpy as np
import pandas as pd
from collections import OrderedDict

sys.path.append("../")
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
                import load_preproc_data_adult, load_preproc_data_compas

from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression

from IPython.display import Markdown, display
import matplotlib.pyplot as plt

## import dataset
dataset_used = "german" # "adult", "german", "compas"
protected_attribute_used = 1 # 1, 2

if dataset_used == "adult":
    dataset_orig = AdultDataset()
#     dataset_orig = load_preproc_data_adult()
    if protected_attribute_used == 1:
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
    else:
        privileged_groups = [{'race': 1}]
        unprivileged_groups = [{'race': 0}]
    
elif dataset_used == "german":
    dataset_orig = GermanDataset()
    if protected_attribute_used == 1:
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
    else:
        privileged_groups = [{'age': 1}]
        unprivileged_groups = [{'age': 0}]
    
elif dataset_used == "compas":
#     dataset_orig = CompasDataset()
    dataset_orig = load_preproc_data_compas()
    if protected_attribute_used == 1:
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
    else:
        privileged_groups = [{'race': 1}]
        unprivileged_groups = [{'race': 0}]    

# cost constraint of fnr will optimize generalized false negative rates, that of
# fpr will optimize generalized false positive rates, and weighted will optimize
# a weighted combination of both
cost_constraint = "fnr" # "fnr", "fpr", "weighted"
#random seed for calibrated equal odds prediction
randseed = 12345679

dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.6], shuffle=True)
dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)

# print out some labels, names, etc.
print("#### Dataset shape")
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
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())

metric_orig_valid = BinaryLabelDatasetMetric(dataset_orig_valid, 
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
print("#### Original validation dataset")
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_valid.mean_difference())

metric_orig_test = BinaryLabelDatasetMetric(dataset_orig_test, 
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
print("#### Original test dataset")
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_test.mean_difference())


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve

# Placeholder for predicted and transformed datasets
dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)

dataset_new_valid_pred = dataset_orig_valid.copy(deepcopy=True)
dataset_new_test_pred = dataset_orig_test.copy(deepcopy=True)

#
# from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler
# from plain_model import PlainModel
# import tensorflow as tf
# scale_orig = MinMaxScaler()
# dataset_orig_train.features=scale_orig.fit_transform(dataset_orig_train.features)
# dataset_orig_valid.features=scale_orig.fit_transform(dataset_orig_valid.features)
# dataset_orig_test.features=scale_orig.fit_transform(dataset_orig_test.features)
# sess=tf.Session()
# in_PM=PlainModel(
#     privileged_groups=privileged_groups,
#     unprivileged_groups=unprivileged_groups,
#     scope_name='plain_classifier',
#     num_epochs=1000,
#     sess=sess)
# in_PM.fit(dataset_orig_train)
# dataset_orig_train_pred=in_PM.predict(dataset_orig_train)
# dataset_orig_valid_pred=in_PM.predict(dataset_orig_valid)
# dataset_orig_test_pred=in_PM.predict(dataset_orig_test)
# sess.close()
# tf.reset_default_graph()
#

#
from logistic_plain_model import LogisticPlainModel
model=LogisticPlainModel()
model.fit(dataset_orig_train)
dataset_orig_train_pred=model.predict(dataset_orig_train)
dataset_orig_valid_pred=model.predict(dataset_orig_valid)
dataset_orig_test_pred=model.predict(dataset_orig_test)
#

#
#Logistic regression classifier and predictions for training data
# scale_orig = StandardScaler()
# X_train = scale_orig.fit_transform(dataset_orig_train.features)
# y_train = dataset_orig_train.labels.ravel()
# lmod = LogisticRegression()
# lmod.fit(X_train, y_train)

# fav_idx = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]
# y_train_pred_prob = lmod.predict_proba(X_train)[:,fav_idx]

# # Prediction probs for validation and testing data
# X_valid = scale_orig.transform(dataset_orig_valid.features)
# y_valid_pred_prob = lmod.predict_proba(X_valid)[:,fav_idx]

# X_test = scale_orig.transform(dataset_orig_test.features)
# y_test_pred_prob = lmod.predict_proba(X_test)[:,fav_idx]

# class_thresh = 0.5
# dataset_orig_train_pred.scores = y_train_pred_prob.reshape(-1,1)
# dataset_orig_valid_pred.scores = y_valid_pred_prob.reshape(-1,1)
# dataset_orig_test_pred.scores = y_test_pred_prob.reshape(-1,1)

# y_train_pred = np.zeros_like(dataset_orig_train_pred.labels)
# y_train_pred[y_train_pred_prob >= class_thresh] = dataset_orig_train_pred.favorable_label
# y_train_pred[~(y_train_pred_prob >= class_thresh)] = dataset_orig_train_pred.unfavorable_label
# dataset_orig_train_pred.labels = y_train_pred

# y_valid_pred = np.zeros_like(dataset_orig_valid_pred.labels)
# y_valid_pred[y_valid_pred_prob >= class_thresh] = dataset_orig_valid_pred.favorable_label
# y_valid_pred[~(y_valid_pred_prob >= class_thresh)] = dataset_orig_valid_pred.unfavorable_label
# dataset_orig_valid_pred.labels = y_valid_pred
    
# y_test_pred = np.zeros_like(dataset_orig_test_pred.labels)
# y_test_pred[y_test_pred_prob >= class_thresh] = dataset_orig_test_pred.favorable_label
# y_test_pred[~(y_test_pred_prob >= class_thresh)] = dataset_orig_test_pred.unfavorable_label
# dataset_orig_test_pred.labels = y_test_pred
#



cm_pred_train = ClassificationMetric(dataset_orig_train, dataset_orig_train_pred,
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
print("#### Original-Predicted training dataset")
result=OrderedDict()
result['TPR']=cm_pred_train.true_positive_rate()
result['TNR']=cm_pred_train.true_negative_rate()
result['FPR']=cm_pred_train.false_positive_rate()
result['FNR']=cm_pred_train.false_negative_rate()
result['Balanced_Acc']=0.5*(result['TPR']+result['TNR'])
result['Acc']=cm_pred_train.accuracy()
result["Statistical parity difference"]=cm_pred_train.statistical_parity_difference()
result["Disparate impact"]=cm_pred_train.disparate_impact()
result["Equal opportunity difference"]=cm_pred_train.equal_opportunity_difference()
result["Average odds difference"]=cm_pred_train.average_odds_difference()
result["Theil index"]=cm_pred_train.theil_index()
result["United Fairness"]=cm_pred_train.generalized_entropy_index()
print(result)

cm_pred_valid = ClassificationMetric(dataset_orig_valid, dataset_orig_valid_pred,
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
print("#### Original-Predicted validation dataset")
result=OrderedDict()
result['TPR']=cm_pred_valid.true_positive_rate()
result['TNR']=cm_pred_valid.true_negative_rate()
result['FPR']=cm_pred_valid.false_positive_rate()
result['FNR']=cm_pred_valid.false_negative_rate()
result['Balanced_Acc']=0.5*(result['TPR']+result['TNR'])
result['Acc']=cm_pred_valid.accuracy()
result["Statistical parity difference"]=cm_pred_valid.statistical_parity_difference()
result["Disparate impact"]=cm_pred_valid.disparate_impact()
result["Equal opportunity difference"]=cm_pred_valid.equal_opportunity_difference()
result["Average odds difference"]=cm_pred_valid.average_odds_difference()
result["Theil index"]=cm_pred_valid.theil_index()
result["United Fairness"]=cm_pred_valid.generalized_entropy_index()
print(result)

cm_pred_test = ClassificationMetric(dataset_orig_test, dataset_orig_test_pred,
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
result=OrderedDict()
result['TPR']=cm_pred_test.true_positive_rate()
result['TNR']=cm_pred_test.true_negative_rate()
result['FPR']=cm_pred_test.false_positive_rate()
result['FNR']=cm_pred_test.false_negative_rate()
result['Balanced_Acc']=0.5*(result['TPR']+result['TNR'])
result['Acc']=cm_pred_test.accuracy()
result["Statistical parity difference"]=cm_pred_test.statistical_parity_difference()
result["Disparate impact"]=cm_pred_test.disparate_impact()
result["Equal opportunity difference"]=cm_pred_test.equal_opportunity_difference()
result["Average odds difference"]=cm_pred_test.average_odds_difference()
result["Theil index"]=cm_pred_test.theil_index()
result["United Fairness"]=cm_pred_test.generalized_entropy_index()
print(result)

# Odds equalizing post-processing algorithm
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from tqdm import tqdm

# Learn parameters to equalize odds and apply to create a new dataset
cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                     unprivileged_groups = unprivileged_groups,
                                     cost_constraint=cost_constraint,
                                     seed=randseed)
cpp = cpp.fit(dataset_orig_valid, dataset_orig_valid_pred)

dataset_transf_valid_pred = cpp.predict(dataset_orig_valid_pred)
dataset_transf_test_pred = cpp.predict(dataset_orig_test_pred)

cm_transf_valid = ClassificationMetric(dataset_orig_valid, dataset_transf_valid_pred,
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
print("#### Original-Transformed validation dataset")
result=OrderedDict()
result['TPR']=cm_transf_valid.true_positive_rate()
result['TNR']=cm_transf_valid.true_negative_rate()
result['FPR']=cm_transf_valid.false_positive_rate()
result['FNR']=cm_transf_valid.false_negative_rate()
result['Balanced_Acc']=0.5*(result['TPR']+result['TNR'])
result['Acc']=cm_transf_valid.accuracy()
result["Statistical parity difference"]=cm_transf_valid.statistical_parity_difference()
result["Disparate impact"]=cm_transf_valid.disparate_impact()
result["Equal opportunity difference"]=cm_transf_valid.equal_opportunity_difference()
result["Average odds difference"]=cm_transf_valid.average_odds_difference()
result["Theil index"]=cm_transf_valid.theil_index()
result["United Fairness"]=cm_transf_valid.generalized_entropy_index()
print(result)

cm_transf_test = ClassificationMetric(dataset_orig_test, dataset_transf_test_pred,
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
print("#### Original-Transformed testing dataset")
result=OrderedDict()
result['TPR']=cm_transf_test.true_positive_rate()
result['TNR']=cm_transf_test.true_negative_rate()
result['FPR']=cm_transf_test.false_positive_rate()
result['FNR']=cm_transf_test.false_negative_rate()
result['Balanced_Acc']=0.5*(result['TPR']+result['TNR'])
result['Acc']=cm_transf_test.accuracy()
result["Statistical parity difference"]=cm_transf_test.statistical_parity_difference()
result["Disparate impact"]=cm_transf_test.disparate_impact()
result["Equal opportunity difference"]=cm_transf_test.equal_opportunity_difference()
result["Average odds difference"]=cm_transf_test.average_odds_difference()
result["Theil index"]=cm_transf_test.theil_index()
result["United Fairness"]=cm_transf_test.generalized_entropy_index()
print(result)
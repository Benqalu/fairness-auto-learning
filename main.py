#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 13:38:14 2019

@author: huihuiliu
"""

# Load all necessary packages
import sys,os,copy
import numpy as np
import tensorflow as tf
from collections import OrderedDict

from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler

#Metrics & Data
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions import get_distortion_adult, get_distortion_german, get_distortion_compas

#Pre-processing
from aif360.algorithms.preprocessing.disparate_impact_remover import DisparateImpactRemover
from aif360.algorithms.preprocessing.lfr import LFR
from aif360.algorithms.preprocessing.optim_preproc  import OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from aif360.algorithms.preprocessing.reweighing import Reweighing

#In-processing
from logistic_plain_model import LogisticPlainModel
from adversarial_debiasing import AdversarialDebiasing
from aif360.algorithms.inprocessing.art_classifier import ARTClassifier
from sklearn.linear_model import LogisticRegression
from art.classifiers import SklearnClassifier
from aif360.algorithms.inprocessing import PrejudiceRemover

#Post-processing
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing.eq_odds_postprocessing import EqOddsPostprocessing
from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification


dataset=["adult", "german", "compas"]  
attribute=["sex", "race", "age"]
pre_process_algorithms=["Nothing","DisparateImpactRemover","LFR","OptimPreproc","Reweighing"] 
in_process_algorithms=["Plain","AdversarialDebiasing","ARTClassifier","PrejudiceRemover"]
post_process_algorithms=["Nothing","CalibratedEqOddsPostprocessing","EqOddsPostprocessing","RejectOptionClassification"]    

def LoadData(dataset_name,protected_attribute_name):

    optim_options=None

    if dataset_name=="adult":
        dataset_original=AdultDataset()
        optim_options={
            "distortion_fun": get_distortion_adult,
            "epsilon": 0.05,
            "clist": [0.99, 1.99, 2.99],
            "dlist": [.1, 0.05, 0]
        }
    elif dataset_name == "german":
        dataset_original=GermanDataset()
        if protected_attribute_name=='sex':
            optim_options = {
                "distortion_fun": get_distortion_german,
                "epsilon": 0.05,
                "clist": [0.99, 1.99, 2.99],
                "dlist": [.1, 0.05, 0]
            }
        if protected_attribute_name=='race':
            optim_options = {
                "distortion_fun": get_distortion_german,
                "epsilon": 0.1,
                "clist": [0.99, 1.99, 2.99],
                "dlist": [.1, 0.05, 0]
            }
    elif dataset_name == "compas":
        dataset_original=CompasDataset()
        optim_options = {
            "distortion_fun": get_distortion_compas,
            "epsilon": 0.05,
            "clist": [0.99, 1.99, 2.99],
            "dlist": [.1, 0.05, 0]
        }

    protected_attribute_set={
        'sex':[[{'sex': 1}],[{'sex': 0}]],
        'age':[[{'age': 1}],[{'age': 0}]],
        'race':[[{'race': 1}],[{'race': 0}]]
    }

    if optim_options==None:
        print('No such dataset & group option.')
        exit()

    return dataset_original,protected_attribute_set[protected_attribute_name][0],protected_attribute_set[protected_attribute_name][1],optim_options

    
def calculate(pre_process,in_process,post_process,dataset_original,privileged_groups,unprivileged_groups,optim_options,in_process_epochs):

    dataset_original_train, dataset_original_test = dataset_original.split([0.3], shuffle=True)

    min_max_scaler=MinMaxScaler()
    dataset_original_train.features=min_max_scaler.fit_transform(dataset_original_train.features)
    dataset_original_test.features=min_max_scaler.transform(dataset_original_test.features)

    #Pre-processing begin
    dataset_after_pre_train=copy.deepcopy(dataset_original_train)
    dataset_after_pre_test=copy.deepcopy(dataset_original_test)
    if pre_process==0:
        pass
    if pre_process==1:
        pre_DIR=DisparateImpactRemover(repair_level=1.0)
        dataset_after_pre_train=pre_DIR.fit_transform(dataset_after_pre_train)
        dataset_after_pre_test=pre_DIR.fit_transform(dataset_after_pre_test)
    if pre_process==2:
        pre_LFR=LFR(unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)
        pre_LFR.fit(dataset_after_pre_train)
        dataset_after_pre_train=pre_LFR.transform(dataset_after_pre_train)
        dataset_after_pre_test=pre_LFR.transform(dataset_after_pre_test)
    if pre_process==3:
        pre_OP=OptimPreproc(OptTools,optim_options,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)
        pre_OP.fit(dataset_original_train)
        dataset_after_pre_train=pre_OP.transform(dataset_original_train)
        dataset_after_pre_test=pre_OP.transform(dataset_original_test)
    if pre_process==4:
        pre_RW=Reweighing(unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)
        pre_RW.fit(dataset_original_train)
        dataset_after_pre_train=pre_RW.transform(dataset_original_train)
        dataset_after_pre_test=pre_RW.transform(dataset_original_test)
    #Pre-processing end
    

    #In-processing begin
    dataset_after_in_train=copy.deepcopy(dataset_after_pre_train)
    dataset_after_in_test=copy.deepcopy(dataset_after_pre_test)
    if in_process==0:
        sess=tf.Session()
        in_LPM=LogisticPlainModel(max_iter=in_process_epochs)
        in_LPM.fit(dataset_after_in_train)
        dataset_after_in_train=in_LPM.predict(dataset_after_in_train)
        dataset_after_in_test=in_LPM.predict(dataset_after_in_test)
    if in_process==1:
        sess = tf.Session()
        in_AD=AdversarialDebiasing(
            privileged_groups=privileged_groups,
            unprivileged_groups=unprivileged_groups,
            scope_name='debiased_classifier',
            num_epochs=in_process_epochs,
            debias=True,
            sess=sess)
        in_AD.fit(dataset_after_in_train)
        dataset_after_in_train=in_AD.predict(dataset_after_in_train)
        dataset_after_in_test=in_AD.predict(dataset_after_in_test)
        sess.close()
    if in_process==2:
        in_ART=ARTClassifier(SklearnClassifier(model=LogisticRegression(max_iter=in_process_epochs)))
        in_ART.fit(dataset_after_in_train,nb_epochs=in_process_epochs)
        dataset_after_in_train=in_AD.predict(dataset_after_in_train)
        dataset_after_in_test=in_AD.predict(dataset_after_in_test)
    if in_process==3:
        sens_attr=privileged_groups[0].keys()[0]
        in_PM=APrejudiceRemover(sensitive_attr=sens_attr,eta=25.0)
        in_PM.fit(dataset_after_in_train)
        dataset_after_in_train=in_PM.predict(dataset_after_in_train)
        dataset_after_in_test=in_PM.predict(dataset_after_in_test)
    #In-process end

    #Post-process begin
    dataset_after_post_train=copy.deepcopy(dataset_after_in_train)
    dataset_after_post_test=copy.deepcopy(dataset_after_in_test)
    if post_process==0:
        pass
    if post_process==1:
        post_CEO=CalibratedEqOddsPostprocessing(
            privileged_groups=privileged_groups,
            unprivileged_groups=unprivileged_groups)
        post_CEO.fit(dataset_true=dataset_after_pre_train,dataset_pred=dataset_after_in_train)
        dataset_after_post_train=post_CEO.predict(dataset_after_post_train)
        dataset_after_post_test=post_CEO.predict(dataset_after_post_test)
    if post_process==2:
        post_EO=EqOddsPostprocessing(unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)
        post_EO.fit(dataset_true=dataset_after_pre_train,dataset_pred=dataset_after_in_train)
        dataset_after_post_train=post_EO.predict(dataset_after_post_train)
        dataset_after_post_test=post_EO.predict(dataset_after_post_test)
    if post_process==3:
        metric_ub=0.05
        metric_lb=-0.05
        post_ROC=RejectOptionClassification(
            unprivileged_groups=unprivileged_groups, 
            privileged_groups=privileged_groups)
            # low_class_thresh=0.01, high_class_thresh=0.99,
            # num_class_thresh=100, num_ROC_margin=50,
            # metric_name="Statistical parity difference",
            # metric_ub=metric_ub, metric_lb=metric_lb)
        post_ROC.fit(dataset_true=dataset_after_pre_train,dataset_pred=dataset_after_in_train)
        dataset_after_post_train=post_ROC.predict(dataset_after_post_train)
        dataset_after_post_test=post_ROC.predict(dataset_after_post_test)
    #Post-processing end

    #Measuring unfairness begin
    dataset_after_pre_test_original_label=copy.deepcopy(dataset_after_pre_test)
    dataset_after_pre_test_original_label.labels=copy.deepcopy(dataset_original_test.labels)
    metric=ClassificationMetric(
        dataset=dataset_after_pre_test_original_label,
        classified_dataset=dataset_after_post_test,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)
    #Measuring unfairness end
    
    result=OrderedDict()
    result['TPR']=metric.true_positive_rate()
    result['TNR']=metric.true_negative_rate()
    result['FPR']=metric.false_positive_rate()
    result['FNR']=metric.false_negative_rate()
    result['Balanced_Acc']=0.5*(result['TPR']+result['TNR'])
    result['Acc']=metric.accuracy()
    result["Statistical parity difference"]=metric.statistical_parity_difference()
    result["Disparate impact"]=metric.disparate_impact()
    result["Equal opportunity difference"]=metric.equal_opportunity_difference()
    result["Average odds difference"]=metric.average_odds_difference()
    result["Theil index"]=metric.theil_index()
    result["United Fairness"]=metric.generalized_entropy_index()

    return result

if __name__=='__main__':

    dataset_name="german"
    protected_attribute_name="sex"
    dataset_orig,privileged_groups,unprivileged_groups,optim_options = LoadData(dataset_name,protected_attribute_name)

    pre_process=0
    in_process=1
    post_process=1
    
    # for pre_process in range(5):
    #     for in_process in range(4):
    #         for post_process in range(4):
                  
    
    algorithms=[pre_process_algorithms[pre_process],in_process_algorithms[in_process],post_process_algorithms[post_process]]

    report=calculate(
        pre_process=pre_process,
        in_process=in_process,
        post_process=post_process,
        dataset_original=copy.deepcopy(dataset_orig),
        privileged_groups=copy.deepcopy(privileged_groups),
        unprivileged_groups=copy.deepcopy(unprivileged_groups),
        optim_options=copy.deepcopy(optim_options),
        in_process_epochs=1000
    )

    print(report)
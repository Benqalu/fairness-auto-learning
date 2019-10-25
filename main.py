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
from time import time
from tools.calculate_entropy import get_entropy
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler
from random import uniform

#Metrics & Data
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
        import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions\
            import get_distortion_adult, get_distortion_german, get_distortion_compas

#Pre-processing
from aif360.algorithms.preprocessing.disparate_impact_remover import DisparateImpactRemover
from aif360.algorithms.preprocessing.lfr import LFR
from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
from tools.opt_tools import OptTools
from aif360.algorithms.preprocessing.reweighing import Reweighing

#In-processing
from classifiers.plain_model import PlainModel
from classifiers.adversarial_debiasing import AdversarialDebiasing
from classifiers.art_classifier import ARTClassifier
from sklearn.linear_model import LogisticRegression
from art.classifiers import SklearnClassifier
from classifiers.prejudice_remover import PrejudiceRemover

#Post-processing
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing.eq_odds_postprocessing import EqOddsPostprocessing
from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification


dataset=["adult", "german", "compas"]  
attribute=["sex", "race", "age"]
pre_process_algorithms=["Nothing","DisparateImpactRemover","LFR","OptimPreproc","Reweighing"] 
in_process_algorithms=["PlainModel","AdversarialDebiasing","ARTClassifier","PrejudiceRemover"]
post_process_algorithms=["Nothing","CalibratedEqOddsPostprocessing","EqOddsPostprocessing","RejectOptionClassification"]	

def LoadData(dataset_name,protected_attribute_name,raw=True):

	optim_options=None

	if dataset_name == "adult":
		if raw:
			dataset_original = AdultDataset()
		if protected_attribute_name == "sex":
			privileged_groups = [{'sex': 1}]
			unprivileged_groups = [{'sex': 0}]
			if not raw:
				dataset_original = load_preproc_data_adult(['sex'])
			optim_options = {
				"distortion_fun": get_distortion_adult,
				"epsilon": 0.05,
				"clist": [0.99, 1.99, 2.99],
				"dlist": [.1, 0.05, 0]
			}
		elif protected_attribute_name == "race":
			privileged_groups = [{'race': 1}]
			unprivileged_groups = [{'race': 0}]
			if not raw:
				dataset_original = load_preproc_data_adult(['race'])
			optim_options = {
			"distortion_fun": get_distortion_adult,
			"epsilon": 0.05,
			"clist": [0.99, 1.99, 2.99],
			"dlist": [.1, 0.05, 0]
		}
	elif dataset_name == "german":
		if raw:
			dataset_original = GermanDataset()
		if protected_attribute_name == "sex":
			privileged_groups = [{'sex': 1}]
			unprivileged_groups = [{'sex': 0}]
			if not raw:
				dataset_original = load_preproc_data_german(['sex'])
			optim_options = {
				"distortion_fun": get_distortion_german,
				"epsilon": 0.05,
				"clist": [0.99, 1.99, 2.99],
				"dlist": [.1, 0.05, 0]
			}
		elif protected_attribute_name == "age":
			privileged_groups = [{'age': 1}]
			unprivileged_groups = [{'age': 0}]
			if not raw:
				dataset_original = load_preproc_data_german(['age'])
			optim_options = {
				"distortion_fun": get_distortion_german,
				"epsilon": 0.05,
				"clist": [0.99, 1.99, 2.99],
				"dlist": [.1, 0.05, 0]
			}
		dataset_original.labels = 2 - dataset_original.labels
		dataset_original.unfavorable_label = 0.
	elif dataset_name == "compas":
		if raw:
			dataset_original = CompasDataset()
		if protected_attribute_name == "sex":
			privileged_groups = [{'sex': 0}]
			unprivileged_groups = [{'sex': 1}]
			if not raw:
				dataset_original = load_preproc_data_compas(['sex'])
			optim_options = {
				"distortion_fun": get_distortion_compas,
				"epsilon": 0.05,
				"clist": [0.99, 1.99, 2.99],
				"dlist": [.1, 0.05, 0]
			}
		elif protected_attribute_name == "race":
			privileged_groups = [{'race': 1}]
			unprivileged_groups = [{'race': 0}]
			if not raw:
				dataset_original = load_preproc_data_compas(['race'])
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
		print('No such dataset & group option:', dataset_name, protected_attribute_name)
		exit()

	return dataset_original,protected_attribute_set[protected_attribute_name][0],protected_attribute_set[protected_attribute_name][1],optim_options


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
		dataset_after_pre_train=pre_OP.transform(dataset_original_train,transform_Y=True)
		dataset_after_pre_test=pre_OP.transform(dataset_original_test,transform_Y=True)
	if pre_process==4:
		pre_RW=Reweighing(unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)
		pre_RW.fit(dataset_original_train)
		dataset_after_pre_train=pre_RW.transform(dataset_original_train)
		dataset_after_pre_test=pre_RW.transform(dataset_original_test)
	#Pre-processing end

	report=get_metric_reports(
		true_dataset=dataset_original_test,
		classfied_dataset=dataset_after_pre_test,
		privileged_groups=privileged_groups,
		unprivileged_groups=unprivileged_groups
	)
	# print('After Pre-process:')
	# print(report)

	#In-processing begin
	dataset_after_in_train=copy.deepcopy(dataset_after_pre_train)
	dataset_after_in_test=copy.deepcopy(dataset_after_pre_test)
	if in_process==0:
		sess = tf.Session()
		in_PM=PlainModel(
			privileged_groups=privileged_groups,
			unprivileged_groups=unprivileged_groups,
			scope_name='plain_classifier',
			num_epochs=in_process_epochs,
			sess=sess)
		in_PM.fit(dataset_after_in_train)
		dataset_after_in_train=in_PM.predict(dataset_after_in_train)
		dataset_after_in_test=in_PM.predict(dataset_after_in_test)
		sess.close()
		tf.reset_default_graph()
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
		tf.reset_default_graph()
	if in_process==2:
		in_ART=ARTClassifier(SklearnClassifier(model=LogisticRegression(max_iter=in_process_epochs)))
		in_ART.fit(dataset_after_in_train)
		dataset_after_in_train=in_ART.predict(dataset_after_in_train)
		dataset_after_in_test=in_ART.predict(dataset_after_in_test)
	if in_process==3:
		sens_attr=list(privileged_groups[0].keys())[0]
		in_PM=PrejudiceRemover(sensitive_attr=sens_attr,eta=25.0)
		in_PM.fit(dataset_after_in_train)
		dataset_after_in_train=in_PM.predict(dataset_after_in_train)
		dataset_after_in_test=in_PM.predict(dataset_after_in_test)
	#In-process end

	report=get_metric_reports(
		true_dataset=dataset_original_test,
		classfied_dataset=dataset_after_in_test,
		privileged_groups=privileged_groups,
		unprivileged_groups=unprivileged_groups
	)
	# print('After In-process:')
	# print(report)

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
			privileged_groups=privileged_groups,
			low_class_thresh=0.01, high_class_thresh=0.99,
			num_class_thresh=100, num_ROC_margin=50,
			metric_name="Statistical parity difference",
			metric_ub=metric_ub, metric_lb=metric_lb)
		post_ROC.fit(dataset_true=dataset_after_pre_train,dataset_pred=dataset_after_in_train)
		dataset_after_post_train=post_ROC.predict(dataset_after_post_train)
		dataset_after_post_test=post_ROC.predict(dataset_after_post_test)
	#Post-processing end

	#Measuring unfairness begin
	report=get_metric_reports(
		true_dataset=dataset_original_test,
		classfied_dataset=dataset_after_post_test,
		privileged_groups=privileged_groups,
		unprivileged_groups=unprivileged_groups
	)

	# print('After Post-process:')
	# print(report)

	return report

def try_combination(dataname,attr,pre=0,inp=0,post=0,creating_dataset=None):

	dataset_name=dataname
	protected_attribute_name=attr
	dataset_orig,privileged_groups,unprivileged_groups,optim_options = LoadData(dataset_name,protected_attribute_name,raw=False)

	vector=[]

	if creating_dataset!=None:
		size=creating_dataset['sample_size']
		portion=1.0*size/len(dataset_orig.labels)
		dataset_orig,_=dataset_orig.split([portion], shuffle=True)
		entropy=get_entropy(
			dataset_orig,
			sensitive_attr=attr,
			top_n=creating_dataset['top_number']
		)
		vector+=entropy

	pre_process=pre
	in_process=inp
	post_process=post

	error_count=0

	while True:
		try:
			algorithms=[
				pre_process_algorithms[pre_process],
				in_process_algorithms[in_process],
				post_process_algorithms[post_process]
			]
			print('\n','-'*10,algorithms,'-'*10)

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
			if not report['Balanced_Acc']<=0.5:
				break
		except:
			print('Had an error, retrying...')
			error_count+=1
			if error_count>20:
				print('Fuck, I give up!')
				break

	if creating_dataset==None:
		return report
	else:
		vector.append(report['TPR'])
		vector.append(report['TNR'])
		vector.append(report['FPR'])
		vector.append(report['FNR'])
		vector.append(report['Balanced_Acc'])
		vector.append(report['Acc'])
		vector.append(report["Statistical parity difference"])
		vector.append(report["Disparate impact"])
		vector.append(report["Equal opportunity difference"])
		vector.append(report["Average odds difference"])
		vector.append(report["Theil index"])
		vector.append(report["United Fairness"])
		return vector

if __name__=='__main__':

	pre_process=1
	in_process=0
	post_process=0

	try:
		os.remove('result_%s_%s.txt'%(dataset_name,protected_attribute_name))
	except:
		pass

	os.system('clear')

	dataname='adult'
	# attr='sex'

	sensitive_attribute={
		'adult':['sex','race'],
		'german':['sex','age'],
		'compas':['sex','race']
	}
	
	# while True:

	approach=[[0,0,0,0,0],[0,0,0,0],[0,0,0,0]]

	pre_process=int(uniform(0,5))
	in_process=int(uniform(0,4))
	post_process=int(uniform(0,4))
	attr=sensitive_attribute[dataname][int(uniform(0,2))]

	approach[0][pre_process]=1
	approach[1][in_process]=1
	approach[2][post_process]=1

	vector=try_combination(
		dataname=dataname,
		attr=attr,
		pre=pre_process,
		inp=in_process,
		post=post_process,
		creating_dataset={
			'sample_size':1000,
			'top_number':5,
			'with_label':True
		}
	)

	f=open('./results/vectors_%s_%s_.txt'%(dataname,attr),'a')
	f.write(str(approach)+'\t'+str(vector)+'\n')
	f.close()

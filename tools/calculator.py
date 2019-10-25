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
	print('After Pre-process:')
	print(report)

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
	print('After In-process:')
	print(report)

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

	print('After Post-process:')
	print(report)

	return report

def try_combination(datasetpre=0,inp=0,post=0):

	dataset_name="german"
	protected_attribute_name="sex"
	dataset_orig,privileged_groups,unprivileged_groups,optim_options = LoadData(dataset_name,protected_attribute_name,raw=False)


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
			if not report['Balanced_Acc']==0.5:
				break
		except:
			print('Had an error, retrying...')
	return report
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve



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

class LogisticPlainModel(object):
	def __init__(self,max_iter=1000):
		self.scale_orig=StandardScaler()
		self.model=LogisticRegression()
		self.fav_idx=None

	def fit(self,dataset_):
		dataset=dataset_.copy(deepcopy=True)
		X_train = self.scale_orig.fit_transform(dataset.features)
		y_train = dataset.labels.ravel()
		self.model.fit(X_train,y_train)
		self.fav_idx = np.where(self.model.classes_ == dataset.favorable_label)[0][0]

	def predict(self,dataset_):

		dataset=dataset_.copy(deepcopy=True)

		X_test=self.scale_orig.transform(dataset.features)
		y_pred_proba=self.model.predict_proba(X_test)[:,self.fav_idx]

		class_thresh=0.5
		dataset.scores=y_pred_proba.reshape(-1,1)
		y_pred = np.zeros_like(dataset.labels)
		y_pred[y_pred_proba >= class_thresh] = dataset.favorable_label
		y_pred[~(y_pred_proba >= class_thresh)] = dataset.unfavorable_label
		dataset.labels = y_pred
		
		return dataset
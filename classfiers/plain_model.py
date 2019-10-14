import numpy as np
import tensorflow as tf

from aif360.algorithms import Transformer


class PlainModel(Transformer):

	def __init__(self,
				 unprivileged_groups,
				 privileged_groups,
				 scope_name,
				 sess,
				 seed=None,
				 adversary_loss_weight=0.1,
				 num_epochs=50,
				 batch_size=128
				 ):
		super(PlainModel, self).__init__(
			unprivileged_groups=unprivileged_groups,
			privileged_groups=privileged_groups)

		self.scope_name = scope_name
		self.seed = seed

		self.unprivileged_groups = unprivileged_groups
		self.privileged_groups = privileged_groups
		if len(self.unprivileged_groups) > 1 or len(self.privileged_groups) > 1:
			raise ValueError("Only one unprivileged_group or privileged_group supported.")
		self.protected_attribute_name = list(self.unprivileged_groups[0].keys())[0]

		self.sess = sess
		self.adversary_loss_weight = adversary_loss_weight
		self.num_epochs = num_epochs
		self.batch_size = batch_size

		self.features_dim = None
		self.features_ph = None
		self.protected_attributes_ph = None
		self.true_labels_ph = None
		self.pred_labels = None

	def _classifier_model(self, features, features_dim, keep_prob):

		with tf.variable_scope("classifier_model"):

			W = tf.get_variable('W', [features_dim, 1],
								  initializer=tf.contrib.layers.xavier_initializer())
			b = tf.Variable(tf.zeros(shape=[1]), name='b')

			pred_logit = tf.matmul(features, W) + b
			pred_label = tf.sigmoid(pred_logit)

		return pred_label, pred_logit

	# def _classifier_model(self, features, features_dim, keep_prob):

	# 	with tf.variable_scope("classifier_model"):
	# 		Wa = tf.get_variable('Wa', [features_dim, self.classifier_num_hidden_units],
	# 							  initializer=tf.contrib.layers.xavier_initializer())
	# 		ba = tf.Variable(tf.zeros(shape=[self.classifier_num_hidden_units]), name='ba')

	# 		h=[]
	# 		b=[]
	# 		W=[]
	# 		for i in range(0,self.classifier_num_hidden_layers):

	# 			if i==0:
	# 				h.append(tf.nn.relu(tf.matmul(features, Wa) + ba))
	# 			else:
	# 				h.append(tf.nn.relu(tf.matmul(h[-1], W[-1]) + b[-1]))
	# 			h[-1]=tf.nn.dropout(h[-1], keep_prob=keep_prob)

	# 			if i<self.classifier_num_hidden_layers-1:

	# 				b.append(tf.Variable(tf.zeros(shape=[self.classifier_num_hidden_units]), name='b%d'%(i+1)))

	# 				W.append(tf.get_variable('W%d'%(i+1), [self.classifier_num_hidden_units, self.classifier_num_hidden_units],
	# 								 initializer=tf.contrib.layers.xavier_initializer()))

	# 		Wz = tf.get_variable('Wz', [self.classifier_num_hidden_units, 1],
	# 							 initializer=tf.contrib.layers.xavier_initializer())
	# 		bz = tf.Variable(tf.zeros(shape=[1]), name='bz')

	# 		pred_logit = tf.matmul(h[-1], Wz) + bz
	# 		pred_label = tf.sigmoid(pred_logit)

	# 	return pred_label, pred_logit

	def _adversary_model(self, pred_logits, true_labels):
		"""Compute the adversary predictions for the protected attribute.
		"""
		with tf.variable_scope("adversary_model"):
			c = tf.get_variable('c', initializer=tf.constant(1.0))
			s = tf.sigmoid((1 + tf.abs(c)) * pred_logits)

			W2 = tf.get_variable('W2', [3, 1],
								 initializer=tf.contrib.layers.xavier_initializer())
			b2 = tf.Variable(tf.zeros(shape=[1]), name='b2')

			pred_protected_attribute_logit = tf.matmul(tf.concat([s, s * true_labels, s * (1.0 - true_labels)], axis=1), W2) + b2
			pred_protected_attribute_label = tf.sigmoid(pred_protected_attribute_logit)

		return pred_protected_attribute_label, pred_protected_attribute_logit

	def fit(self, dataset):

		if self.seed is not None:
			np.random.seed(self.seed)

		# Map the dataset labels to 0 and 1.
		temp_labels = dataset.labels.copy()

		temp_labels[(dataset.labels == dataset.favorable_label).ravel(),0] = 1.0
		temp_labels[(dataset.labels == dataset.unfavorable_label).ravel(),0] = 0.0

		with tf.variable_scope(self.scope_name):
			num_train_samples, self.features_dim = np.shape(dataset.features)

			# Setup placeholders
			self.features_ph = tf.placeholder(tf.float32, shape=[None, self.features_dim])
			self.protected_attributes_ph = tf.placeholder(tf.float32, shape=[None,1])
			self.true_labels_ph = tf.placeholder(tf.float32, shape=[None,1])
			self.keep_prob = tf.placeholder(tf.float32)

			# Obtain classifier predictions and classifier loss
			self.pred_labels, pred_logits = self._classifier_model(self.features_ph, self.features_dim, self.keep_prob)
			pred_labels_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.true_labels_ph, logits=pred_logits))

			# Setup optimizers with learning rates
			global_step = tf.Variable(0, trainable=False)
			starter_learning_rate = 0.001
			learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
													   1000, 0.96, staircase=True)
			classifier_opt = tf.train.AdamOptimizer(learning_rate)

			classifier_vars = [var for var in tf.trainable_variables() if 'classifier_model' in var.name]

			normalize = lambda x: x / (tf.norm(x) + np.finfo(np.float32).tiny)

			classifier_grads = []
			for (grad,var) in classifier_opt.compute_gradients(pred_labels_loss, var_list=classifier_vars):
				classifier_grads.append((grad, var))
			classifier_minimizer = classifier_opt.apply_gradients(classifier_grads, global_step=global_step)

			self.sess.run(tf.global_variables_initializer())
			self.sess.run(tf.local_variables_initializer())

			# Begin training
			for epoch in range(self.num_epochs):
				shuffled_ids = np.random.choice(num_train_samples, num_train_samples)
				for i in range(num_train_samples//self.batch_size):
					batch_ids = shuffled_ids[self.batch_size*i: self.batch_size*(i+1)]
					batch_features = dataset.features[batch_ids]
					batch_labels = np.reshape(temp_labels[batch_ids], [-1,1])
					batch_protected_attributes = np.reshape(dataset.protected_attributes[batch_ids][:,
												 dataset.protected_attribute_names.index(self.protected_attribute_name)], [-1,1])

					batch_feed_dict = {self.features_ph: batch_features,
									   self.true_labels_ph: batch_labels,
									   self.protected_attributes_ph: batch_protected_attributes,
									   self.keep_prob: 0.8}

					_, pred_labels_loss_value = self.sess.run(
						[classifier_minimizer,
						 pred_labels_loss], feed_dict=batch_feed_dict)
				print("epoch %d; batch classifier loss: %f" % (epoch, pred_labels_loss_value))
		return self

	def predict(self, dataset):
		"""Obtain the predictions for the provided dataset using the fair
		classifier learned.

		Args:
			dataset (BinaryLabelDataset): Dataset containing labels that needs
				to be transformed.
		Returns:
			dataset (BinaryLabelDataset): Transformed dataset.
		"""
		if self.seed is not None:
			np.random.seed(self.seed)

		num_test_samples, _ = np.shape(dataset.features)

		samples_covered = 0
		pred_labels = []
		while samples_covered < num_test_samples:
			start = samples_covered
			end = samples_covered + self.batch_size
			if end > num_test_samples:
				end = num_test_samples
			batch_ids = np.arange(start, end)
			batch_features = dataset.features[batch_ids]
			batch_labels = np.reshape(dataset.labels[batch_ids], [-1,1])
			batch_protected_attributes = np.reshape(dataset.protected_attributes[batch_ids][:,
										 dataset.protected_attribute_names.index(self.protected_attribute_name)], [-1,1])

			batch_feed_dict = {self.features_ph: batch_features,
							   self.true_labels_ph: batch_labels,
							   self.protected_attributes_ph: batch_protected_attributes,
							   self.keep_prob: 1.0}

			pred_labels += self.sess.run(self.pred_labels, feed_dict=batch_feed_dict)[:,0].tolist()
			samples_covered += len(batch_features)

		# Mutated, fairer dataset with new labels
		dataset_new = dataset.copy(deepcopy = True)
		dataset_new.labels = (np.array(pred_labels)>0.5).astype(np.float64).reshape(-1,1)

		# Map the dataset labels to back to their original values.
		temp_labels = dataset_new.labels.copy()

		temp_labels[(dataset_new.labels == 1.0).ravel(), 0] = dataset.favorable_label
		temp_labels[(dataset_new.labels == 0.0).ravel(), 0] = dataset.unfavorable_label

		dataset_new.labels = temp_labels.copy()

		return dataset_new

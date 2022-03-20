import tensorflow as tf
import numpy as np
import os

class Model():
	def __init__(self, args):
		self.args = args
		sess_config = tf.ConfigProto()
		sess_config.gpu_options.allow_growth=True
		self.sess = tf.Session(config=sess_config)

		self.build_graph()

		self.saver = tf.train.Saver(max_to_keep=5)
		self.best_saver = tf.train.Saver()
		self.sess.run(tf.global_variables_initializer())
		self.sess.run(tf.local_variables_initializer())

	def build_graph(self):
		self.data = tf.placeholder(tf.int32, [None, None], name='data')
		self.label = tf.placeholder(tf.int32, [None, 1], name='label')
		self.seq_len = tf.placeholder(tf.int32, [None], name='seq_len')
		self.global_step = tf.Variable(0, trainable=False)
		self.embed()
		if self.args.algo == 'rnn':
			self.core_rnn()
		elif self.args.algo == 'cnn':
			self.core_cnn()
		elif self.args.algo == 'lstm':
			self.core_lstm()
		self.build_loss()
		self.build_trainop()

	def embed(self):
		with tf.variable_scope('word_embedding'):
			self.word_embeddings = tf.get_variable(
				'word_embeddings',
				shape=(self.args.vocab_size, self.args.emb_size),
				initializer=tf.truncated_normal_initializer(stddev=0.1),
				trainable=True
			)
			self.data_emb = tf.nn.embedding_lookup(self.word_embeddings, self.data)		# [N, D, ed]

	def core_rnn(self):
		N = tf.shape(self.data)[0]
		# Bi-LSTM
		cell_fw = tf.contrib.rnn.BasicLSTMCell(self.args.hidden_size)
		cell_bw = tf.contrib.rnn.BasicLSTMCell(self.args.hidden_size)

		if self.args.dropout_keep_prob < 1:
			cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.args.dropout_keep_prob)
			cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.args.dropout_keep_prob)

		with tf.variable_scope('bi-lstm'):
			(outputs_fw, outputs_bw), (states_fw, states_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, 
												self.data_emb, sequence_length=self.seq_len, dtype=tf.float32)
			# outputs = tf.reshape(tf.reduce_mean(tf.concat(outputs, 2), axis=1), [N, self.args.hidden_size*2])	# [N, 2d]
			outputs = tf.reshape(tf.concat([states_fw.h, states_bw.h], 1), [N, self.args.hidden_size*2])	# [N, 2d]
		print("___________________________________________%s" % (states_fw.h))
		print("___________________________________________%s" % (states_bw.h))
		# fully-connection
		with tf.variable_scope('fc1'):
			shape = 2*self.args.hidden_size
			fc1w = tf.Variable(tf.truncated_normal([shape, 2], dtype=tf.float32,
													stddev=1e-1), name='weights')
			fc1b = tf.Variable(tf.constant(0.0, shape=[2], dtype=tf.float32),
								trainable=True, name='biases')
			self.fc1 = tf.nn.bias_add(tf.matmul(outputs, fc1w), fc1b)

		self.logits = tf.nn.softmax(self.fc1)

	# def core_lstm(self):
	# 	N = tf.shape(self.data)[0]
	# 	# lstm
	# 	# multi-lstm
	# 	# NUM_LAYER = 2
	# 	# cell = tf.contrib.rnn.MultiRNNCell(
	# 	# 	[tf.contrib.rnn.BasicLSTMCell(self.args.hidden_size)
	# 	# 	 for _ in range(NUM_LAYER)])
	# 	# single-lstm
	#
	# 	cell = tf.contrib.rnn.BasicLSTMCell(self.args.hidden_size)
	#
	# 	with tf.variable_scope("dynamic_rnn"):
	# 		outputs, last_state = tf.nn.dynamic_rnn(
	# 			cell,
	# 			self.data_emb,
	# 			sequence_length=self.seq_len,
	# 			dtype=tf.float32)
	# 	outputs = tf.reshape(last_state.h, [N, self.args.hidden_size])
	# 	# fully-connection
	# 	with tf.variable_scope('fc1'):
	# 		shape = self.args.hidden_size
	# 		fc1w = tf.Variable(tf.truncated_normal([shape, 2], dtype=tf.float32,
	# 											   stddev=1e-1), name='weights')
	# 		fc1b = tf.Variable(tf.constant(0.0, shape=[2], dtype=tf.float32),
	# 						   trainable=True, name='biases')
	# 		self.fc1 = tf.nn.bias_add(tf.matmul(outputs, fc1w), fc1b)
	#
	# 	self.logits = tf.nn.softmax(self.fc1)

	def core_lstm(self):
		N = tf.shape(self.data)[0]
		# convolution layer
		conv1 = tf.layers.conv1d(
			inputs=self.data_emb,
			filters=2,
			kernel_size=2,
			strides=1,
			padding='same',
			activation=tf.nn.relu
		)

		# multi-lstm
		# cell = tf.contrib.rnn.BasicLSTMCell(self.args.hidden_size)
		NUM_LAYER = 2
		cell = tf.contrib.rnn.MultiRNNCell(
			[tf.contrib.rnn.BasicLSTMCell(self.args.hidden_size)
			 for _ in range(NUM_LAYER)])

		with tf.variable_scope("dynamic_rnn"):
			outputs, last_state = tf.nn.dynamic_rnn(
				cell,
				conv1,
				self.seq_len,
				dtype=tf.float32)
		newoutputs = last_state[-1][1]
		print(last_state)
		print('----------------------------------%s' % (newoutputs.get_shape()))
		outputs = tf.reshape(newoutputs, [N, self.args.hidden_size])
		# fully-connection
		with tf.variable_scope('fc1'):
			shape = self.args.hidden_size
			fc1w = tf.Variable(tf.truncated_normal([shape, 2], dtype=tf.float32,
												   stddev=1e-1), name='weights')
			fc1b = tf.Variable(tf.constant(0.0, shape=[2], dtype=tf.float32),
							   trainable=True, name='biases')
			self.fc1 = tf.nn.bias_add(tf.matmul(outputs, fc1w), fc1b)

		self.logits = tf.nn.softmax(self.fc1)

	def core_cnn(self):
		ed = self.args.emb_size
		N = tf.shape(self.data_emb)[0]
		D = tf.shape(self.data_emb)[1]
		filter_size = 5
		self.num_filters = self.args.num_filters

		with tf.variable_scope('conv1_%s'%filter_size):
			kernel1 = tf.Variable(tf.truncated_normal([filter_size, ed, 1, self.num_filters],
								dtype=tf.float32, stddev=1e-1), name='weights_%s'%filter_size)
			conv1 = tf.nn.conv2d(
				tf.expand_dims(self.data_emb, -1),
				kernel1,
				[1, 1, 1, 1],
				padding='VALID'
			)   # [N, D-fs+1, nf]

			biases1 = tf.Variable(tf.constant(0.0, shape=[self.num_filters], dtype=tf.float32),
								trainable=True, name='biases_%s'%filter_size)
			h1 = tf.reshape(tf.nn.relu(tf.nn.bias_add(conv1, biases1)), [N, D-filter_size+1, self.num_filters]) 
			# pooled = tf.nn.max_pool(h, ksize=[1, D-filter_size+1, 1, 1], strides=[1, 1, 1, 1], 
			# 						padding='VALID',name="pool")			# [N, 1, 1, nf]


		with tf.variable_scope('conv2_%s'%filter_size):
			kernel2 = tf.Variable(tf.truncated_normal([filter_size, self.num_filters, 1, self.num_filters], 
								dtype=tf.float32, stddev=1e-1), name='weights_%s'%filter_size)
			conv2 = tf.nn.conv2d(tf.expand_dims(h1, -1), kernel2, [1, 1, 1, 1], padding='VALID')   # [N, D-2fs+2, nf]
			biases2 = tf.Variable(tf.constant(0.0, shape=[self.num_filters], dtype=tf.float32),
								trainable=True, name='biases_%s'%filter_size)
			h = tf.nn.relu(tf.nn.bias_add(conv2, biases2)) 

		self.pool_flat = tf.reshape(tf.reduce_max(h, axis=1), [N, self.num_filters])		# [N, nf]

		with tf.variable_scope('dropout'):
			self.h_drop = tf.nn.dropout(self.pool_flat, keep_prob=self.args.dropout_keep_prob)

		# fully-connection
		with tf.variable_scope('fc1'):
			shape = self.num_filters
			fc1w = tf.Variable(tf.truncated_normal([shape, 2], dtype=tf.float32,
													stddev=1e-1), name='weights')
			fc1b = tf.Variable(tf.constant(0.0, shape=[2], dtype=tf.float32),
								trainable=True, name='biases')
			self.fc1 = tf.nn.bias_add(tf.matmul(self.h_drop, fc1w), fc1b)

		self.logits = tf.nn.softmax(self.fc1)


	def build_loss(self):
		N = tf.shape(self.data)[0]
		label = tf.one_hot(self.label, 2, dtype=tf.float32)
		self.acc_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=self.logits))
		# regularization
		all_vars = tf.trainable_variables()
		# lstm_vars = [var for var in all_vars if 'bi-lstm' in var.name and 'kernel' in var.name]
		self.l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in all_vars])

		self.loss = self.acc_loss + self.args.reg * self.l2_loss

		# self.precision, _ = tf.metrics.precision(tf.squeeze(self.label), 
		# 										   tf.argmax(self.logits, axis=-1, output_type=tf.int32))
		# self.recall, _ = tf.metrics.recall(tf.squeeze(self.label), 
		# 									 tf.argmax(self.logits, axis=-1, output_type=tf.int32))


	def build_trainop(self):
		# learning_rate = tf.train.exponential_decay(self.args.learning_rate, self.global_step, 
		# 										   100000, 0.96, staircase=True)
		# optimizer = tf.train.AdamOptimizer(learning_rate)
		# self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

		optimizer = tf.train.AdamOptimizer(self.args.learning_rate)
		self.train_op = optimizer.minimize(self.loss)

	def save(self, epoch):
		model_dir = os.path.join(self.args.model_dir, str(self.args.id))
		if not os.path.exists(model_dir):
			os.mkdir(model_dir)
		self.saver.save(self.sess, os.path.join(model_dir, 'model_'+str(self.args.algo)+'_'+str(epoch)))

	def save_best(self):
		model_dir = os.path.join(self.args.model_dir, str(self.args.id))
		if not os.path.exists(model_dir):
			os.mkdir(model_dir)
		self.best_saver.save(self.sess, os.path.join(model_dir, 'model_'+str(self.args.algo)))

	def load(self, epoch):
		model_dir = os.path.join(self.args.model_dir, str(self.args.id))
		self.saver.restore(self.sess, os.path.join(model_dir, 'model_'+str(self.args.algo)+'_'+str(epoch)))

	def load_model(self):
		model_dir = os.path.join(self.args.model_dir, str(self.args.id))
		self.saver.restore(self.sess, os.path.join(model_dir, 'model_' + str(self.args.algo) + '_' + str(99)))

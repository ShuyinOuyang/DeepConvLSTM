# coding=utf-8
import numpy as np
import tensorflow as tf
from data_iter import Dataset
from model import Model
import argparse
import time
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

vocab_size = 20000

class Dataset():
	def __init__(self):
		self.max_len = 0
		self.batch_size = 32
		self.data_path = 'dataset/Data/exp2/Java/0/30'
		self.vocab = json.load(open(os.path.join(self.data_path, 'vocab.json')))
		self.vocab_size = len(self.vocab)
		global vocab_size
		vocab_size = self.vocab_size
		print('Vocab Size: ', self.vocab_size)
		self.predict_data = self.read_predict('predict')
		print('Load predict data: ', len(self.predict_data['data']))

	def pad(self, datas):
		seq_len = []
		max_len = 0
		for data in datas:
			seq_len.extend([len(data)])
			if len(data) > max_len:
				max_len = len(data)
		if max_len < self.max_len:
			max_len = self.max_len
		new_d = []
		for data in datas:
			nd = []
			nd.extend(data)
			rest = max_len-len(data)
			nd.extend([0]*rest)
			new_d.append(nd)
		return new_d, seq_len

	def read_predict(self, name):
		# test
		name = 'test'
		with open(os.path.join(self.data_path, name+'.data')) as f:
			lines = f.readlines()
		data = {'data':[]}
		for i, line in enumerate(lines):
			seq = []
			for word in line.strip().split():
				if word not in self.vocab:
					seq.extend([1])
				else:
					seq.extend([self.vocab[word]])
			data['data'].append(seq)
		return data

	def gen_mini_batches(self, data_name=None, shuffle=True):
		if data_name == 'predict':
			data = self.predict_data

		batch_size = self.batch_size
		data_size = len(data['data'])
		indices = np.arange(data_size)
		if shuffle:
			np.random.shuffle(indices)
		for batch_start in np.arange(0, data_size, batch_size):
			batch_indices = indices[batch_start: batch_start + batch_size]
			new_d, seq_len = self.pad([data['data'][i] for i in batch_indices])
			batch_data = {'data': new_d,
						  'length': seq_len}
			yield batch_data

class Model():
	def __init__(self):
		self.algo = 'lstm'
		global vocab_size
		self.vocab_size = vocab_size
		self.emb_size = 150
		self.batch_size = 32
		self.data_path = 'dataset/Data/exp2/Java/0/30'
		self.epochs = 100
		self.hidden_size = 256
		self.dropout_keep_prob = 0.5
		self.id = 'newlstm_Java_0_30'
		self.learning_rate = 0.001
		self.max_len = 0
		self.model_dir = 'models'
		self.model_id = 0
		self.num_filters = 150
		self.optim = 'adam'
		self.reg = 0.001
		self.test = False
		self.train = True

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
		self.seq_len = tf.placeholder(tf.int32, [None], name='seq_len')
		self.global_step = tf.Variable(0, trainable=False)
		self.embed()
		if self.algo == 'rnn':
			self.core_rnn()
		elif self.algo == 'cnn':
			self.core_cnn()
		elif self.algo == 'lstm':
			self.core_lstm()


	def embed(self):
		with tf.variable_scope('word_embedding'):
			self.word_embeddings = tf.get_variable(
				'word_embeddings',
				shape=(self.vocab_size, self.emb_size),
				initializer=tf.truncated_normal_initializer(stddev=0.1),
				trainable=True
			)
			self.data_emb = tf.nn.embedding_lookup(self.word_embeddings, self.data)  # [N, D, ed]

	def core_rnn(self):
		N = tf.shape(self.data)[0]
		# Bi-LSTM
		cell_fw = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
		cell_bw = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)

		if self.dropout_keep_prob < 1:
			cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.dropout_keep_prob)
			cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.dropout_keep_prob)

		with tf.variable_scope('bi-lstm'):
			(outputs_fw, outputs_bw), (states_fw, states_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
																							   self.data_emb,
																							   sequence_length=self.seq_len,
																							   dtype=tf.float32)
			# outputs = tf.reshape(tf.reduce_mean(tf.concat(outputs, 2), axis=1), [N, self.hidden_size*2])	# [N, 2d]
			outputs = tf.reshape(tf.concat([states_fw.h, states_bw.h], 1), [N, self.hidden_size * 2])  # [N, 2d]
		# fully-connection
		with tf.variable_scope('fc1'):
			shape = 2 * self.hidden_size
			fc1w = tf.Variable(tf.truncated_normal([shape, 2], dtype=tf.float32,
												   stddev=1e-1), name='weights')
			fc1b = tf.Variable(tf.constant(0.0, shape=[2], dtype=tf.float32),
							   trainable=True, name='biases')
			self.fc1 = tf.nn.bias_add(tf.matmul(outputs, fc1w), fc1b)

		self.logits = tf.nn.softmax(self.fc1)
		self.predict_res = tf.nn.softmax(self.logits)

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
		# cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
		NUM_LAYER = 2
		cell = tf.contrib.rnn.MultiRNNCell(
			[tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
			 for _ in range(NUM_LAYER)])

		with tf.variable_scope("dynamic_rnn"):
			outputs, last_state = tf.nn.dynamic_rnn(
				cell,
				conv1,
				self.seq_len,
				dtype=tf.float32)
		newoutputs = last_state[-1][1]
		outputs = tf.reshape(newoutputs, [N, self.hidden_size])
		# fully-connection
		with tf.variable_scope('fc1'):
			shape = self.hidden_size
			fc1w = tf.Variable(tf.truncated_normal([shape, 2], dtype=tf.float32,
												   stddev=1e-1), name='weights')
			fc1b = tf.Variable(tf.constant(0.0, shape=[2], dtype=tf.float32),
							   trainable=True, name='biases')
			self.fc1 = tf.nn.bias_add(tf.matmul(outputs, fc1w), fc1b)

		self.logits = tf.nn.softmax(self.fc1)
		self.predict_res = tf.nn.softmax(self.logits)

	def core_cnn(self):
		ed = self.emb_size
		N = tf.shape(self.data_emb)[0]
		D = tf.shape(self.data_emb)[1]
		filter_size = 5
		self.num_filters = self.num_filters

		with tf.variable_scope('conv1_%s' % filter_size):
			kernel1 = tf.Variable(tf.truncated_normal([filter_size, ed, 1, self.num_filters],
													  dtype=tf.float32, stddev=1e-1), name='weights_%s' % filter_size)
			conv1 = tf.nn.conv2d(
				tf.expand_dims(self.data_emb, -1),
				kernel1,
				[1, 1, 1, 1],
				padding='VALID'
			)  # [N, D-fs+1, nf]

			biases1 = tf.Variable(tf.constant(0.0, shape=[self.num_filters], dtype=tf.float32),
								  trainable=True, name='biases_%s' % filter_size)
			h1 = tf.reshape(tf.nn.relu(tf.nn.bias_add(conv1, biases1)), [N, D - filter_size + 1, self.num_filters])
		# pooled = tf.nn.max_pool(h, ksize=[1, D-filter_size+1, 1, 1], strides=[1, 1, 1, 1],
		# 						padding='VALID',name="pool")			# [N, 1, 1, nf]

		with tf.variable_scope('conv2_%s' % filter_size):
			kernel2 = tf.Variable(tf.truncated_normal([filter_size, self.num_filters, 1, self.num_filters],
													  dtype=tf.float32, stddev=1e-1), name='weights_%s' % filter_size)
			conv2 = tf.nn.conv2d(tf.expand_dims(h1, -1), kernel2, [1, 1, 1, 1], padding='VALID')  # [N, D-2fs+2, nf]
			biases2 = tf.Variable(tf.constant(0.0, shape=[self.num_filters], dtype=tf.float32),
								  trainable=True, name='biases_%s' % filter_size)
			h = tf.nn.relu(tf.nn.bias_add(conv2, biases2))

		self.pool_flat = tf.reshape(tf.reduce_max(h, axis=1), [N, self.num_filters])  # [N, nf]

		with tf.variable_scope('dropout'):
			self.h_drop = tf.nn.dropout(self.pool_flat, keep_prob=self.dropout_keep_prob)

		# fully-connection
		with tf.variable_scope('fc1'):
			shape = self.num_filters
			fc1w = tf.Variable(tf.truncated_normal([shape, 2], dtype=tf.float32,
												   stddev=1e-1), name='weights')
			fc1b = tf.Variable(tf.constant(0.0, shape=[2], dtype=tf.float32),
							   trainable=True, name='biases')
			self.fc1 = tf.nn.bias_add(tf.matmul(self.h_drop, fc1w), fc1b)

		self.logits = tf.nn.softmax(self.fc1)
		self.predict_res = tf.nn.softmax(self.logits)

	def save(self, epoch):
		model_dir = os.path.join(self.model_dir, str(self.id))
		if not os.path.exists(model_dir):
			os.mkdir(model_dir)
		self.saver.save(self.sess, os.path.join(model_dir, 'model_'+str(self.algo)+'_'+str(epoch)))

	def save_best(self):
		model_dir = os.path.join(self.model_dir, str(self.id))
		if not os.path.exists(model_dir):
			os.mkdir(model_dir)
		self.best_saver.save(self.sess, os.path.join(model_dir, 'model_'+str(self.algo)))

	def load_model(self):
		model_dir = os.path.join(self.model_dir, str(self.id))
		self.saver.restore(self.sess, os.path.join(model_dir, 'model_' + str(self.algo) + '_' + str(99)))

def predict():
	# load predict data
	print('Reading data ...')
	data = Dataset()
	model = Model()
	print('Loading model ...')
	model.load_model()
	print('Start predicting ...')
	res = []
	for bid, batch in enumerate(data.gen_mini_batches(data_name='predict', shuffle=False)):
		feed_dict = {model.data: batch['data'],
					 model.seq_len: batch['length']}

		predict_res = model.sess.run(model.predict_res, feed_dict)
		batch_res = []
		for predict in predict_res:
			if predict[0] > 0.5:
				batch_res.append(0)
			else:
				batch_res.append(1)
		res += batch_res
	print(res)
	print(sum(res))


if __name__ == '__main__':
	predict()
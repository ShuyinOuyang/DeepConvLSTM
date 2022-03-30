import numpy as np
import tensorflow as tf
from data_iter import Dataset
from model import Model
import argparse
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def parse_args():
	parser = argparse.ArgumentParser('classifier')
	parser.add_argument('--train', action='store_true',
						help='train the model')
	parser.add_argument('--test', action='store_true',
						help='test the model')
	parser.add_argument('--predict', action='store_true',
						help='predict')
	# parser.add_argument('--id', type=int, default=0,
	# 					help='program id')
	parser.add_argument('--id', default='0',
						help='program id')
	parser.add_argument('--model_id', type=int, default=0,
						help='load model id')

	train_settings = parser.add_argument_group('train settings')
	train_settings.add_argument('--optim', default='adam',
								help='optimizer type')
	train_settings.add_argument('--algo',
								choices=['rnn', 'cnn', 'lstm'],
								default='rnn',
                                help='choose the algorithm to use')
	train_settings.add_argument('--learning_rate', type=float, default=1e-3,
								help='learning rate')
	train_settings.add_argument('--dropout_keep_prob', type=float, default=0.5,
								help='dropout keep rate')
	train_settings.add_argument('--batch_size', type=int, default=32,
								help='train batch size')
	train_settings.add_argument('--epochs', type=int, default=100,
								help='train epochs')

	model_settings = parser.add_argument_group('model settings')
	model_settings.add_argument('--vocab_size', type=int, default=20000,
								help='vocab size')
	model_settings.add_argument('--emb_size', type=int, default=150,
								help='embedding size')
	model_settings.add_argument('--max_len', type=int, default=0,
								help='max length of data')
	model_settings.add_argument('--hidden_size', type=int, default=256,
								help='size of LSTM hidden units')
	model_settings.add_argument('--num_filters', type=int, default=150,
								help='num of filters in CNN')
	model_settings.add_argument('--reg', type=float, default=1e-3,
								help='l2 reg')

	# commons.collections
	path_settings = parser.add_argument_group('path settings')
	path_settings.add_argument('--data_path',
							   default='dataset/Data/exp4/commons.collections',
							   help='train data')
	path_settings.add_argument('--model_dir', default='models',
							   help='the dir to store models')

	return parser.parse_args()

def metric(logits, labels):
	all_labels = np.array(labels)
	all_logits = np.array(logits)
	TP = np.count_nonzero(all_labels*all_logits)
	TN = np.count_nonzero((all_labels-1)*(all_logits-1))
	FP = np.count_nonzero((all_labels-1)*all_logits)
	FN = np.count_nonzero(all_labels*(all_logits-1))
	print('TP:', TP,'TN:', TN, 'FP:', FP, 'FN:', FN)
	if TP == 0:
		return 1e-5, 1e-5
	else:
		return TP*1.0/(TP+FP), TP*1.0/(TP+FN)

def train(args):
	print(args)
	print('Reading data ...')
	data = Dataset(args)
	model = Model(args)
	print('Start training ...')
	best_f1 = 0
	for epoch in range(args.epochs):
		print('epoch', epoch, 'finished, start test ...')
		step = 0
		losses = 0
		all_logits = []
		all_labels = []
		for bid, batch in enumerate(data.gen_mini_batches(data_name='test')):
			feed_dict = {model.data: batch['data'],
						 model.label: batch['label'],
						 model.seq_len: batch['length']}
			loss, logits = model.sess.run([model.loss, model.logits], feed_dict)

			all_logits.extend(np.argmax(logits, 1))
			all_labels.extend(np.reshape(batch['label'], [-1]))

			step += 1
			losses += loss
		precision, recall = metric(all_logits, all_labels)
		print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), \
			  'epoch', epoch, 'test data loss:', losses/step, 'precision:', precision, 'recall:', recall, \
			  'F-measure:', 2/(1/precision+1/recall))

		if 2/(1/precision+1/recall) > best_f1:
			best_f1 = 2/(1/precision+1/recall)
			print('Saving best model ... ')
			model.save_best()

		step = 0
		train_loss = 0
		all_logits = []
		all_labels = []
		for bid, batch in enumerate(data.gen_mini_batches(data_name='train')):
			feed_dict = {model.data: batch['data'],
						 model.label: batch['label'],
						 model.seq_len: batch['length']}
			_, loss, logits = model.sess.run([model.train_op, model.loss, model.logits], feed_dict)
			step += 1
			train_loss += loss
			all_logits.extend(np.argmax(logits, 1))
			all_labels.extend(np.squeeze(batch['label']))

			if bid % 100 == 0:
				precision, recall = metric(all_logits, all_labels)
				print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), \
					  'epoch', epoch, 'batch', bid, \
					  ': loss', train_loss/step, 'precision', precision, 'recall', recall)
				step = 0
				train_loss = 0
				all_logits = []
				all_labels = []

		print('Saving model ...')
		model.save(epoch)

def test(args):
	print('Reading data ...')
	data = Dataset(args)
	model = Model(args)
	print('Loading model ...')
	model.load_model()
	print('Start testing ...')
	data_names = ['test']
	res = []
	for data_name in data_names:
		print('Dataset:', data_name)
		step = 0
		losses = 0
		all_logits = []
		all_labels = []

		for bid, batch in enumerate(data.gen_mini_batches(data_name=data_name, shuffle=False)):
			feed_dict = {model.data: batch['data'],
						 model.label: batch['label'],
						 model.seq_len: batch['length']}

			loss, logits = model.sess.run([model.loss, model.logits], feed_dict)
			tmp_session = tf.Session()
			predict_res = tmp_session.run(tf.nn.softmax(logits))
			batch_res = []
			for predict in predict_res:
				if predict[0] > 0.5:
					batch_res.append(0)
				else:
					batch_res.append(1)
			res += batch_res

			all_logits.extend(np.argmax(logits, 1))
			all_labels.extend(np.squeeze(batch['label']))

			step += 1
			losses += loss
		precision, recall = metric(all_logits, all_labels)
		print('loss:', losses / step, 'precision:', precision, 'recall:', recall, 'F-measure:',
			  2 / (1 / precision + 1 / recall))



def predict(args):
	print('Reading data ...')
	data = Dataset(args)
	model = Model(args)
	print('Loading model ...')
	model.load(args.model_id)
	print('Start testing ...')
	# take other test data as example to predict
	data_names = ['test']
	res = []
	for data_name in data_names:
		print('Dataset:', data_name)

		for bid, batch in enumerate(data.gen_mini_batches(data_name=data_name, shuffle=False)):
			feed_dict = {model.data: batch['data'],
						 model.label: batch['label'],
						 model.seq_len: batch['length']}
			_, logits = model.sess.run([model.loss, model.logits], feed_dict)
			tmp_session = tf.Session()
			predict_res = tmp_session.run(tf.nn.softmax(logits))
			batch_res = []
			for predict in predict_res:
				if predict[0] > 0.5:
					batch_res.append(0)
				else:
					batch_res.append(1)
			res += batch_res
	count = 0
	with open(os.path.join(args.data_path, 'test.data')) as f:
		for line in f.readlines():
			print('sentence: %s, predict result: %s' % (line, res[count]))
			count += 1

def run():
	args = parse_args()
	if args.train:
		train(args)
	if args.test:
		test(args)
	if args.predict:
		predict(args)

if __name__ == '__main__':
	run()

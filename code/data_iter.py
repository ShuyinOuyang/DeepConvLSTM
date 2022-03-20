import os
import numpy as np
import json

class Dataset():
	def __init__(self, args):
		self.args = args
		self.data_path = args.data_path
		self.vocab = json.load(open(os.path.join(self.data_path, 'vocab.json')))
		self.args.vocab_size = len(self.vocab)
		print('Vocab Size: ', self.args.vocab_size)
		self.train_data, pos_cnt = self.read('train')
		print('Load train data: ', len(self.train_data['data']), ' with ', pos_cnt, ' positive')
		self.test_data, pos_cnt = self.read('test')
		print('Load test data: ', len(self.test_data['data']), ' with ', pos_cnt, ' positive')


	def read(self, name):
		with open(os.path.join(self.data_path, name+'.data')) as f:
			lines = f.readlines()
		with open(os.path.join(self.data_path, name+'.label')) as f:
			labels = f.readlines()
		data = {'data':[], 'label':[]}
		pos_cnt = 0
		for i, line in enumerate(lines):
			seq = []
			for word in line.strip().split():
				if word not in self.vocab:
					seq.extend([1])
				else:
					seq.extend([self.vocab[word]])
			data['data'].append(seq)
			data['label'].append([int(labels[i])])
			if int(labels[i]) == 1:
				pos_cnt += 1
		return data, pos_cnt

	def pad(self, datas):
		seq_len = []
		max_len = 0
		for data in datas:
			seq_len.extend([len(data)])
			if len(data) > max_len:
				max_len = len(data)
		if max_len < self.args.max_len:
			max_len = self.args.max_len
		new_d = []
		for data in datas:
			nd = []
			nd.extend(data)
			rest = max_len-len(data)
			nd.extend([0]*rest)
			new_d.append(nd)
		return new_d, seq_len

	def gen_mini_batches(self, data_name=None, shuffle=True):
		if data_name == 'train':
			data = self.train_data
		elif data_name == 'test':
			data = self.test_data

		batch_size = self.args.batch_size
		data_size = len(data['data'])
		indices = np.arange(data_size)
		if shuffle:
			np.random.shuffle(indices)
		for batch_start in np.arange(0, data_size, batch_size):
			batch_indices = indices[batch_start: batch_start + batch_size]
			new_d, seq_len = self.pad([data['data'][i] for i in batch_indices])
			batch_data = {'data': new_d,
						  'label': [data['label'][i] for i in batch_indices],
						  'length': seq_len}
			yield batch_data



import numpy as np
import os
import pandas as pd
import json
import nltk
import random
from collections import OrderedDict
import sys
import traceback
# def create():
# 	data = pd.read_csv('./dataset2.csv', encoding='utf-8')
# 	Data = []
#
# 	for i in range(len(data[0])):
# 		one = {}
# 		one['fold'] = data[0].iloc(i)
# 		one['project'] = data[1].iloc(i)
# 		one['content'] = data[3].iloc(i)
# 		one['label'] = data[5].iloc(i)
# 		# only one special
# 		if data[5].iloc(i) != 0 and data[5].iloc(i) != 1:
# 			one['label'] = data[4].iloc(i)
# 		Data.append(one)
# 	return Data

def create():
	Data = []
	count = 0
	with open('./dataset2.csv', 'r', encoding='utf-8') as f:
		for line in f:
			try:
				content = line.strip().split(',')
				one = {}
				one['fold'] = content[0]
				one['project'] = content[1]
				one['content'] = content[3]
				if content[5] != '0' and content[5] != '1':
					one['label'] = content[4]
				one['label'] = content[5]

				Data.append(one)
			except:
				count += 1
				continue
	return Data

def build_dic(path):
	vocab = OrderedDict()
	files = [os.path.join(path, 'train.data'), os.path.join(path, 'test.data')]
	for file in files:
		with open(file, 'r') as f:
			lines = f.readlines()
		for line in lines:
			for word in line.strip().split():
				try:
					word = word.encode('utf-8').decode('utf-8')
				except UnicodeDecodeError:
					pass
				if word not in vocab:
					vocab[word] = 0
				vocab[word] += 1
	words = list(vocab.keys())
	freqs = list(vocab.values())

	sorted_idx = np.array(freqs).argsort()
	sorted_words = [words[ii] for ii in sorted_idx[::-1]]
	worddict = OrderedDict()

	worddict['<pad>'] = 0
	worddict['<unk>'] = 1
	for ii, ww in enumerate(sorted_words):
		worddict[ww] = ii + 2
	w_dict = json.dumps(worddict, indent=2)
	with open(os.path.join(path, 'vocab.json'), 'w') as f:
		f.write(w_dict)

def exp4(datas):
	projects = ['Java', 'JFace', 'commons.collections']
	for project in projects:
		Train = []
		Test = []
		pos = cnt = 0.0
		for data in datas:
			if data['project'] != project:
				Train.append({'content': nltk.word_tokenize(data['content']),
							  'label': data['label']})
				if data['label'] == '1':
					pos += 1
				cnt += 1
			else:
				Test.append({'content': nltk.word_tokenize(data['content']),
							 'label': data['label']})
		print(pos, cnt)
		times = int(np.around(cnt/pos))-1
		path = os.path.join('Data/exp4', project)
		if not os.path.exists(path):
			os.makedirs(path)
		train_cnt = 0
		train_pos = 0
		test_cnt = 0
		test_pos = 0
		df = open(os.path.join(path, 'train.data'), 'w')
		lf = open(os.path.join(path, 'train.label'), 'w')
		for data in Train:
			if data['label'] == '1':
				for k in range(times):
					for ii, token in enumerate(data['content']):
						df.write(token+' ')
					df.write('\n')
					lf.write(str(data['label']))
					lf.write('\n')
					train_pos += 1
					train_cnt += 1
			else:
				for ii, token in enumerate(data['content']):
					df.write(token+' ')
				df.write('\n')
				lf.write(str(data['label']))
				lf.write('\n')
				train_cnt += 1
		df.close()
		lf.close()
		df = open(os.path.join(path, 'test.data'), 'w')
		lf = open(os.path.join(path, 'test.label'), 'w')
		for data in Test:
			for ii, token in enumerate(data['content']):
				df.write(token+' ')
			df.write('\n')
			lf.write(str(data['label']))
			lf.write('\n')
			test_cnt += 1
			if data['label'] == '1':
				test_pos += 1
		df.close()
		lf.close()
		print(project, '   train:', train_cnt, 'train_pos:', train_pos, 'test:', test_cnt, 'test_pos:', test_pos)
		build_dic(path)

def exp2(datas):
	projects = ['Java', 'JFace', 'commons.collections']
	for project in projects:
		for fold in range(10):
			Train = []
			Test = []
			pos = cnt = 0.0
			for data in datas:
				if data['project'] == project:
					if data['fold'] != str(fold):
						Train.append({'content': nltk.word_tokenize(data['content']),
									  'label': data['label']})
						if data['label'] == '1':
							pos += 1
						cnt += 1
					else:
						Test.append({'content': nltk.word_tokenize(data['content']),
									 'label': data['label']})
			print(pos, cnt)
			for cent in [6, 30, 50, 80]:
			# for cent in [20, 40, 60, 70]:
				if cent == 6:
					times = 1
				else:
					times = int(np.around(cent/(100.0-cent)*(cnt/pos-1)))
				path = os.path.join('Data/exp2', project, str(fold), str(cent))
				if not os.path.exists(path):
					os.makedirs(path)
				train_cnt = 0
				train_pos = 0
				test_cnt = 0
				test_pos = 0
				df = open(os.path.join(path, 'train.data'), 'w')
				lf = open(os.path.join(path, 'train.label'), 'w')
				for data in Train:
					if data['label'] == '1':
						for k in range(times):
							for ii, token in enumerate(data['content']):
								df.write(token+' ')
							df.write('\n')
							lf.write(str(data['label']))
							lf.write('\n')
							train_pos += 1
							train_cnt += 1
					else:
						for ii, token in enumerate(data['content']):
							df.write(token+' ')
						df.write('\n')
						lf.write(str(data['label']))
						lf.write('\n')
						train_cnt += 1
				df.close()
				lf.close()
				df = open(os.path.join(path, 'test.data'), 'w')
				lf = open(os.path.join(path, 'test.label'), 'w')
				for data in Test:
					for ii, token in enumerate(data['content']):
						df.write(token+' ')
					df.write('\n')
					lf.write(str(data['label']))
					lf.write('\n')
					test_cnt += 1
					if data['label'] == '1':
						test_pos += 1
				df.close()
				lf.close()			
				print(project, fold, cent, '   train:', train_cnt, 'train_pos:', train_pos, \
					  'test:', test_cnt, 'test_pos:', test_pos)
				build_dic(path)

def main():
	datas = create()
	# print(len(datas))
	exp2(datas)


if __name__ == '__main__':
    main()
import numpy as np
from gensim.models import Word2Vec
from keras.layers import Input, LSTM, Dense, Conv1D, Concatenate, Flatten
from keras.models import Model, load_model
from keras.optimizers import adam
from keras.preprocessing.sequence import pad_sequences
import argparse

def transform_w2v(s, model):
	vec = []
	for w in s:
		if w in model.wv.vocab:
			vec.append(model.wv[w])
		else:
			vec.append(model.wv['OOV'])
	return np.array(vec)

def read_data(path, model, task):
	question = []
	ans = []
	R_num = []
	C_num = []
	# test data format ORGQ+++$$$+++RELQ+++$$$+++RELC1+++$$$+++....+++$$$+++RELC10
	# according to different task get different sentence
	if task == 'taskA':
		with open(path, 'r', encoding='utf8') as fp:
			for line in fp:
				line = line.replace('\n','').split('+++$$$+++')
				for i,c in enumerate(line[4:len(line)-1]):
					R_num.append(line[2])
					question.append(transform_w2v(line[3].replace(',','').split(), model))
					C_num.append(line[2]+'_C'+str(i+1))
					ans.append(transform_w2v(c.replace(',','').split(), model))
	elif task == 'taskB':
		with open(path, 'r', encoding='utf8') as fp:
			for line in fp:
				line = line.replace('\n','').split('+++$$$+++')
				R_num.append(line[0])
				question.append(transform_w2v(line[3].replace(',','').split(), model))
				C_num.append(line[2])
				ans.append(transform_w2v(line[4].replace(',','').split(), model))
	else:
		for line in fp:
			line = line.replace('\n','').split('+++$$$+++')
			for i,c in enumerate(line[4:len(line)-1]):
				R_num.append(line[0])
				question.append(transform_w2v(line[3].replace(',','').split(), model))
				C_num.append(line[2]+'_C'+str(i+1))
				ans.append(transform_w2v(c.replace(',','').split(), model))

	return np.array(question), np.array(ans), np.array(R_num), np.array(C_num)

def main():
	# parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--task', type=str)
	parser.add_argument('--w2v', type=str)
	parser.add_argument('--model_name', type=str)
	parser.add_argument('--output_name', type=str)
	args = parser.parse_args()
	# load data and word2vec model
	data_path = 'test_data/testing_data.txt'
	w2v = Word2Vec.load(args.w2v)
	question, answer, R_num, C_num = read_data(data_path, w2v, args.task)
	print(len(question))
	print(len(answer))
	print(len(R_num))
	print(len(C_num))
	# pad sentence to the same length
	question_pad = pad_sequences(question, maxlen=50, dtype='float32', padding='post', value=w2v.wv['PAD'])
	answer_pad = pad_sequences(answer, maxlen=50, dtype='float32', padding='post', value=w2v.wv['PAD'])
	# load NN model
	model = load_model(args.model_name)
	# predict score
	ans = model.predict([question_pad,answer_pad])
	with open(args.output_name, 'w', encoding='utf8') as fp:
		for i in range(len(ans)):
			# set label to true if score > 0.5, else false
			if ans[i] > 0.5:
				fp.write(str(R_num[i]) + '\t' + str(C_num[i]) + '\t' + '0' + '\t' + str(ans[i][0]) + '\t' + 'true\n')
			else:
				fp.write(str(R_num[i]) + '\t' + str(C_num[i]) + '\t' + '0' + '\t' + str(ans[i][0]) + '\t' + 'false\n')
	print(ans)

	
	

if __name__ == '__main__':
	main()
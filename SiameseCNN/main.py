import numpy as np
from gensim.models import Word2Vec
from keras.layers import Input, LSTM, Dense, Conv1D, Concatenate, Flatten, Bidirectional, LSTM, MaxPooling1D, BatchNormalization
from keras.models import Model, load_model
from keras.optimizers import adam, Adam
from keras.preprocessing.sequence import pad_sequences
import argparse

# transform word to vector
def transform_w2v(s, model):
	vec = []
	for w in s:
		# if word in vocabuary the get the vector, else use OOV to represent
		if w in model.wv.vocab:
			vec.append(model.wv[w])
		else:
			vec.append(model.wv['OOV'])
	return np.array(vec)

def read_data(path, model, task):
	question = []
	ans = []
	label = []
	with open(path, 'r', encoding='utf8') as fp:
		for line in fp:
			line = line.replace('\t','EOS').replace('\n','').split('EOS')
			# line[0] is question, line[1] is comment, line[2] is label
			question.append(transform_w2v(line[0].split(),model))
			ans.append(transform_w2v(line[1].split(), model))
			l = line[2].split()
			# split label into three interval
			if task == 'taskA':
				if l[0] == '__label__Good':
					label.append(int(1))
				elif l[0] == '__label__Bad':
					label.append(int(0))
				else:
					label.append(float(0.5))
			elif task == 'taskB':
				if l[0] == '__label__Irrelevant':
					label.append(int(0))
				elif l[0] == '__label__Relevant':
					label.append(float(0.5))
				else:
					label.append(int(1))

	return np.array(question), np.array(ans), np.array(label)

def data_generator(data, data2, targets, w2v):
	batch_size = 64
	shuffle_cnt = 1
	idx = np.arange(len(data))
	while True:
		np.random.shuffle(idx)
		# random data that size is batch_size
		batches = [idx[range(batch_size*i, min(len(data), batch_size*(i+1)))] for i in range(len(data)//batch_size+1)] 
		for i in batches:
			xx, xx2, yy = [], [], []
			# get these data
			for j in i:
				xx.append(data[j])
				xx2.append(data2[j])
				yy.append(targets[j])
			# pad question and comment to the same length
			xx_pad = pad_sequences(xx, maxlen=50, dtype='float32', padding='post', value=w2v.wv['PAD'])
			xx2_pad = pad_sequences(xx2, maxlen=50, dtype='float32', padding='post', value=w2v.wv['PAD'])
			xx, xx2, yy = np.asarray(xx), np.asarray(xx2), np.asarray(yy)
			# yield data to train
			yield [xx_pad, xx2_pad], yy

def build_model(length=50,dim=100):
	# define model architecture
	Conv1 = Conv1D(16,5,strides=1)
	Conv2 = Conv1D(32,5,strides=1)
	Conv3 = Conv1D(64,5,strides=1)
	nor = BatchNormalization()
	dense1 = Dense(64)
	dense2 = Dense(32)
	# question CNN model
	q_input = Input(shape=(length,dim))
	q_1 = Conv1(q_input)
	q_2 = Conv2(q_1)
	q_3 = Conv3(q_2)
	q_4 = nor(q_3)
	# comment CNN model
	a_input = Input(shape=(length,dim))
	a_1 = Conv1(a_input)
	a_2 = Conv2(a_1)
	a_3 = Conv3(a_2)
	a_4 = nor(a_3)
	# concat and DNN model
	concat = Concatenate()([q_4,a_4])
	o = Flatten()(concat)
	o = dense1(o)
	o = dense2(o)
	# output score
	out = Dense(1, activation='sigmoid')(o)
	model = Model([q_input,a_input],out)
	model.summary()
	return model

def main():
	# parse argument
	parser = argparse.ArgumentParser()
	parser.add_argument('--task', type=str)
	parser.add_argument('--data_path', default='None', type=str)
	parser.add_argument('--model_name', type=str)
	args = parser.parse_args()
	# choose which word2vec to use
	if args.task == 'taskA' or args.task == 'taskC':
		w2v = Word2Vec.load('word2vec_A_100.h5')
	elif args.task == 'taskB':
		w2v = Word2Vec.load('word2vec_B_100.h5')
	question, answer, label = read_data(args.data_path, w2v, args.task)
	model = build_model()
	adam = Adam(lr=1e-5, decay=1e-7)
	model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
	batch_size = 128
	# use data_generator to avoid out of memory
	model.fit_generator(data_generator(question,answer, label, w2v), steps_per_epoch=int((len(label)+batch_size-1)/batch_size), epochs=200)
	model.save(args.model_name)
	
	

if __name__ == '__main__':
	main()
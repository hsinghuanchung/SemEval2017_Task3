from keras import backend as K
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda, BatchNormalization, Flatten
from keras.layers.merge import add, concatenate
from keras.models import Model, load_model
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences
import sys
from keras.callbacks import ModelCheckpoint, EarlyStopping
import gensim
import numpy as np
K.set_learning_phase(0)


word2vec_model = gensim.models.Word2Vec.load('word2vec_model_taskA_128.model')

query = []
comment = []
y = []
x_query = []
x_comment = []

################## Data preprocess 
#############      Task B preprocess
if sys.argv[1] == 'B':
	file = open(sys.argv[2]).readlines()
	for sentence in file:
		sentence = sentence.strip().split(' EOS ')
		query.append(sentence[0].split(' '))
		
		sentence = sentence[1].split(' \t __label__')
		
		comment.append(sentence[0].split(' '))
		if sentence[1] == 'Irrelevant' : y.append(0)
		else : y.append(1)
##############     Task A preprocess
elif sys.argv[1] == 'A' or sys.argv[3] == 'C':
	file = open(sys.argv[2]).readlines()
	for sentence in file:
		sentence = sentence.strip().split(' EOS ')
		query.append(sentence[0].split(' '))
		
		sentence = sentence[1].split(' \t __label__')
		
		comment.append(sentence[0].split(' '))
		if sentence[1] == 'Good' : y.append(1)
		else : y.append(0)
################################

v = np.zeros(128)
for  q in query:
	temp = []
	for  w in q : temp.append(word2vec_model[w])
	temp = np.array(temp)
	
	x_query.append(temp)
x_query = pad_sequences(x_query, maxlen=50, dtype='int32', padding='post', truncating='post', value=v)

for  c in comment:
	temp = []
	for  w in c : temp.append(word2vec_model[w])
	temp = np.array(temp)
#	temp = pad_sequences(temp, maxlen=50, dtype='int32', padding='post', truncating='post', value=v)
	x_comment.append(temp)
x_comment = pad_sequences(x_comment, maxlen=50, dtype='int32', padding='post', truncating='post', value=v)



######################################################

################input and build model
q_input = Input(shape=(50, 128))
c_input = Input(shape=(50, 128))
################CNN layer
inner = Conv1D(16, 3, padding='same', name='conv11', kernel_initializer='he_normal')(q_input)  # (None, 128, 64, 64)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling1D(pool_size=2, name='max11')(inner)  # (None,64, 32, 64)

inner = Conv1D(32, 3, padding='same', name='conv22', kernel_initializer='he_normal')(inner)  # (None, 64, 32, 128)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling1D(pool_size=2, name='max22')(inner)  # (None, 32, 16, 128)

inner = Conv1D(64, 3, padding='same', name='conv33', kernel_initializer='he_normal')(inner)  # (None, 32, 16, 256)
inner = BatchNormalization()(inner)
inner1 = Activation('relu')(inner)

inner = Conv1D(16, 3, padding='same', name='conv1', kernel_initializer='he_normal')(c_input)  # (None, 128, 64, 64)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling1D(pool_size=2, name='max1')(inner)  # (None,64, 32, 64)

inner = Conv1D(32, 3, padding='same', name='conv2', kernel_initializer='he_normal')(inner)  # (None, 64, 32, 128)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling1D(pool_size=2, name='max2')(inner)  # (None, 32, 16, 128)

inner = Conv1D(64, 3, padding='same', name='conv3', kernel_initializer='he_normal')(inner)  # (None, 32, 16, 256)
inner = BatchNormalization()(inner)
inner2 = Activation('relu')(inner)


##############RNN layer
lstm_1 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(inner1)  # (None, 32, 512)
lstm_1b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(inner2)
lstm1_merged = concatenate([lstm_1, lstm_1b])  # (None, 32, 512)
lstm1_merged = BatchNormalization()(lstm1_merged)
inner = Flatten()(lstm1_merged)

 
inner = Dense(1, kernel_initializer='he_normal',name='dense2')(inner) #(None, 32, 63)
y_pred = Activation('sigmoid', name='fuck')(inner)
##############train and save best
model = Model([q_input, c_input], y_pred)
model.summary()

checkpoint = ModelCheckpoint(sys.argv[3], monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]
model.compile(loss='binary_crossentropy',optimizer="adam",metrics=['accuracy'])
model.fit([x_query, x_comment], y, batch_size=256,epochs=500,validation_split=.2, callbacks=callbacks_list)



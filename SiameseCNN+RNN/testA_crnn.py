from keras import backend as K
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda, BatchNormalization, Flatten
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
import gensim
import sys
import re
import xml.etree.ElementTree as ET
import math 
from keras.models import Sequential , load_model
from keras.layers.convolutional import Conv2D 
from keras.layers.pooling import MaxPooling2D , AveragePooling2D
from keras.optimizers import Adam , SGD
from keras.callbacks import ModelCheckpoint
import h5py

#########load word2vec model
word2vec_model = gensim.models.Word2Vec.load('word2vec_model_taskA_128.model')


x_query = []
x_comment = []
cdict = []
qdict = []
###########test_data preprocessing
###########using Element tree to read xml data
tree = ET.parse(sys.argv[1])
root = tree.getroot()
for child in root :
    	orgq_id = child.attrib["ORGQ_ID"]
    	relq_id = child[2].attrib["THREAD_SEQUENCE"]
    	orgq_subject = child[0].text
    	orgq_body = child[1].text 
    	relq_subject = child[2][0][0].text
    	relq_body = re.sub("[^a-zA-Z^0-9^ ]+" , '' , child[2][0][1].text)
    	
    	for i in range(1 , 11):
    		qdict.append(relq_id)
    		relc_id = child[2][i].attrib["RELC_ID"]
    		rel_comment = re.sub("[^a-zA-Z^0-9^ ]+" , '' ,child[2][i][0].text)
    		x_query.append(relq_body)
    		x_comment.append(rel_comment)
    		cdict.append(relc_id)
                

x_test_query = []
x_test_comment =[]
v = np.zeros(128)
#####################pading data
for  q in x_query:
	temp = []
	q = q.split(' ')
	for  w in q : 
	    if w in word2vec_model.wv.vocab : temp.append(word2vec_model[w])
	    else : temp.append(word2vec_model["oov"]) 
	temp = np.array(temp)
	x_test_query.append(temp)
x_test_query = pad_sequences(x_test_query, maxlen=50, dtype='int32', padding='post', truncating='post', value=v)

for  c in x_comment:
	temp = []
	c = c.split(' ')
	for  w in c : 
	    if w in word2vec_model.wv.vocab : temp.append(word2vec_model[w])
	    else : temp.append(word2vec_model["oov"]) 
	temp = np.array(temp)
	x_test_comment.append(temp)
x_test_comment = pad_sequences(x_test_comment, maxlen=50, dtype='int32', padding='post', truncating='post', value=v)
##################################################

############testing
model = load_model(sys.argv[2])
y_predict = model.predict([x_test_query , x_test_comment])
y_predict = y_predict.reshape(8800)
output_file = open(sys.argv[3] , 'w')
for i in range(len(x_comment)) : 
	if y_predict[i] >= 0.5 : quality = "true"
	else : quality = "false"
	output_file.write(qdict[i] + ' ' +  cdict[i] + ' ' +  '0' + ' ' + str(y_predict[i]) + ' '  + quality + '\n')


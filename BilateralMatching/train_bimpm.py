from util import DataManager
import numpy as np
import os, argparse
from sys import argv
import math
import tensorflow as tf
from keras import regularizers, initializers, constraints
from keras.layers.core import Permute, Flatten
from keras.models import Model, load_model
from keras.layers import Layer, Input, GRU, LSTM, Dense, Dropout, Bidirectional, concatenate, multiply, Lambda, average, add
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.backend import set_session
from sklearn.utils import class_weight
import gensim
import gensim.downloader as api 
from contextlib import redirect_stdout


gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
set_session(sess)

parser = argparse.ArgumentParser(description='Natural Language Matching')
parser.add_argument('modelname')
parser.add_argument('datapath')
parser.add_argument('--subtask', choices=['A', 'B', 'C'])
parser.add_argument('--worddim', default='100')
args = parser.parse_args()

def crop(dimension, start, end):
    """
        Crops (or slices) a Tensor on a given dimension from start to end
        example : to crop tensor x[:, :, 5:10], call slice(2, 5, 10) as you want to crop on the second dimension
        Arguments:
            dimension: Integer, the dimension which you want to crop
            start: Integer, the start index of the cropped tensor
            end: Integer, the end index of the cropped tensor
        Returns:
            Lambda(func)
    """
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)


def BiMPM(question_embedding_inputs, comment_embedding_inputs, dropout_rate):
    """
        The network architecture of the recurrent neural networks and bilateral matching layer.
        The input embeddings are each fed into a bidirectional GRU which returns a sequence, forming the context representation layer. Later, the two sequences in the context representation layer are fed into a matching layer. The matching layer outputs two matched sequences and feed the sequences into two bidirectional GRUs. Finally, I concatenate the outputs of the two GRUs and return it.

        Arguments:
            question_embedding_inputs: The embeddings of question sentence
            comment_embedding_inputs: The embeddings of comment sentence
            dropout_rate: Float, dropout rate
        Returns:
            RNN_output: A tensor which will be fed into a feed forward neural network and a softmax layer
    """
        
    Question_LSTM_cell = Bidirectional(GRU(64, 
                                       return_sequences=True, 
                                       dropout=dropout_rate))
    Comment_LSTM_cell = Bidirectional(GRU(64, 
                                      return_sequences=True, 
                                      dropout=dropout_rate))
    question_output = Question_LSTM_cell(question_embedding_inputs)
    comment_output = Comment_LSTM_cell(comment_embedding_inputs)
    qlist = []
    clist = []
    for i in range(40):
        qlist.append(crop(1, i, i + 1)(question_output))
        clist.append(crop(1, i, i + 1)(comment_output))
   
    matchqlist = []
    matchclist = []
    for i in range(40):
        inmatchqlist = []
        inmatchclist = []
        for j in range(40):
            inmatchqlist.append(multiply([qlist[i], clist[j]]))
            inmatchclist.append(multiply([clist[i], qlist[j]]))
        matchqlist.append(average(inmatchqlist))
        matchclist.append(average(inmatchclist))
    print("length of matchqlist", len(matchqlist)) 
    print("length of matchclist", len(matchclist)) 
    q = concatenate(matchqlist, axis=1)
    c = concatenate(matchclist, axis=1)
    FinalQ_GRU_cell = GRU(16,
                          return_sequences=False,
                          dropout=dropout_rate)
    FinalC_GRU_cell = GRU(16,
                          return_sequences=False,
                          dropout=dropout_rate)
    q = FinalQ_GRU_cell(q)
    c = FinalC_GRU_cell(c)
    RNN_output = concatenate([q,c])
    return RNN_output

def W2VRNN(word_index, embedding_matrix, modelname, worddim, subtask):
    """
        This function defines the model. The model has two inputs, representing the natural language sentence pair and one output. 
        Arguments:
            word_index: A dictionary mapping word to index
            embedding_matrix: A numpy array having a size of worddim * vocabulary size
            modelname: String, name of the model
            worddim: Integer, The dimension of word vectors
            subtask: String, "A" or "B" or "C"

        Returns:
            model: A keras model
    """
    print('start training w2v')
    question_inputs = Input(shape=(40,))
    comment_inputs = Input(shape=(40,))
    if subtask == 'B':
        question_embedding_inputs = Embedding(len(word_index), worddim, weights=[embedding_matrix], trainable=False)(question_inputs)
        comment_embedding_inputs = Embedding(len(word_index), worddim, weights=[embedding_matrix], trainable=False)(comment_inputs)
    elif subtask == 'A' or subtask == 'C':
        question_embedding_inputs = Embedding(len(word_index), worddim, weights=[embedding_matrix], trainable=True)(question_inputs)
        comment_embedding_inputs = Embedding(len(word_index), worddim, weights=[embedding_matrix], trainable=True)(comment_inputs)

    # RNN 
    # return_sequence = False
    
    dropout_rate = 0.5
    RNN_output = BiMPM(question_embedding_inputs, comment_embedding_inputs, dropout_rate)
        
    outputs = Dense(16, 
                    activation='relu',
                    kernel_regularizer=regularizers.l2(0.1))(RNN_output)
    outputs = Dense(2, activation='softmax')(outputs)
    model =  Model(inputs=[question_inputs, comment_inputs], outputs=outputs)
    model.summary()
    
    with open(os.path.join('subtask' + subtask, 'models', modelname, 'modelsummary.txt'), 'w') as f:
        with redirect_stdout(f):
            model.summary()
    # optimizer
    adam = Adam(lr=0.0005)
    print ('compile model...')
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy',])
    return model
    

def main():
    """ Main function of train_bimpm.py
    Arguments:
        modelname: Name your model!
        train_file: Training data file
        subtask: "A" or "B" or "C"
        worddim: The dimension of word embedding, 100 or 200
    Outputs:
        subtask + [subtask]/models/[modelname]/model.h5: Trained model
        subtask + [subtask]/models/[modelname]/idx2word.pkl: A dictionary mapping index to word
        subtask + [subtask]/models/[modelname]/word2idx.pkl: A dictionary mapping word to index
        subtask + [subtask]/models/[modelname]/modelsummary.txt: Summary of the model 
    """

    modelname = args.modelname
    train_file = args.datapath
    subtask = args.subtask
    worddim = args.worddim
    if not os.path.isdir(os.path.join('subtask' + subtask, 'models',modelname)):
        os.makedirs(os.path.join('subtask' + subtask, 'models',modelname))

    # Load and process training data
    dm = DataManager(subtask)
    word2idx_path = os.path.join('subtask' + subtask, 'models', modelname, 'word2idx.pkl')
    idx2word_path = os.path.join('subtask' + subtask, 'models', modelname, 'idx2word.pkl')
    if os.path.exists(word2idx_path) and os.path.exists(idx2word_path):
        dm.load_tokenizer(word2idx_path, idx2word_path)
    else: 
        dm.tokenize('glove-wiki-gigaword-' + worddim)
        dm.save_tokenizer(word2idx_path, idx2word_path)
    
    dm.add_data('train', train_file)
    dm.to_category()
    dm.to_sequence(40, 40)
    (train_Q, train_C), train_Y = dm.get_data('train')
    
    print(len(train_Q), len(train_Q[0]))
    print(len(train_C), len(train_C[0]))
    print(train_Y[0:3])
    label_sum = np.sum(train_Y, axis=0)
    print("label_sum", label_sum)
    class_weights = {0: label_sum[1], 1: label_sum[0]}
    print("class_weights", class_weights)
    
    word_index, embedding_matrix = dm.embedding_matrix('glove-wiki-gigaword-' + worddim, int(worddim))
    model = W2VRNN(word_index, embedding_matrix, modelname, int(worddim), subtask)
        
    earlystopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='max')
    model_path = os.path.join('subtask' + subtask, 'models', modelname, 'model.h5')
    checkpoint = ModelCheckpoint(filepath=model_path,
                                 verbose=1,
                                 save_best_only=True,
                                 monitor='val_acc',
                                 save_weights_only=False,
                                 mode='max')
    model.fit([train_Q, train_C], train_Y, epochs=40, batch_size=512, validation_split=0.1, shuffle=True, callbacks=[earlystopping, checkpoint], class_weight=class_weights)
    
    

if __name__ == '__main__':
   main()

import numpy as np
import re
import pickle
import operator
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import gensim
import gensim.downloader as api 


class DataManager:
    def __init__(self, subtask):
        """ self.data is a dictionary which stores training and testing data.
        Key:
            The key specifies the properties of different dataset
            (e.g. train_insult, train_noninsult, test_insult, test_noninsult)
        Value:
            The value is a list [X,Y], where...
            X is a list of comments,
            Y is a list of corresponding ground truth.
        """
        self.data = {}
        if subtask == 'A' or subtask == 'C':
            self.categorymap = {'__label__Bad': 0.0, '__label__Good': 1.0, '__label__PotentiallyUseful': 0.0}
        elif subtask == 'B':
            self.categorymap = {'__label__Irrelevant': 0.0, '__label__Relevant': 1.0, '__label__PerfectMatch': 1.0}
    def add_data(self, key, data_path):
        print('read ' + key)
        if key == 'train':
            X = []
            Y = []
            with open(data_path, 'r') as f:
                for idx, line in enumerate(f):
                    line = line.strip().split('\t')
                    X.append(line[0])
                    Y.append(self.categorymap[line[1].strip()])
            print(X[0:2])
            print(Y[0:2])
            
            self.data['train'] = [X, Y]                
        
        if key == 'test':
            qidlist = []
            X = []
            with open(data_path, "r") as f:
                for line in f:
                    line = line.strip().split('\t')
                    qidlist.append(line[0])
                    X.append(line[1])
            print(X[0:2])
            self.data['test'] = [X] 
            self.qidlist = qidlist

    # def tokenize(self, vocab_size):
    #     """ Assign each character with an index
    #     Arguments:
    #         vocab_size: The maximum number of characters to assign indices.
    #     """
    #     print('create new tokenizer')
    #     self.tokenizer = Tokenizer(num_words=vocab_size, lower=False, filters='')
    #     texts = self.data['train'][0]
    #     self.tokenizer.fit_on_texts(texts)

    #     self.tokenizer.word_index = {e: i for e, i in self.tokenizer.word_index.items() if i <= vocab_size}
    
    def tokenize(self, wordvec_path):
        """ Assign each character with an index
        Arguments:
            vocab_size: The maximum number of characters to assign indices.
        """
        print('create new tokenizer')
        vocabulary_word2index = {}
        vocabulary_index2word = {}
        model = api.load(wordvec_path)
        vocabulary_word2index['PAD_ID'] = 0
        vocabulary_index2word[0] = 'PAD_ID'
        vocabulary_word2index['OOV'] = 1 
        vocabulary_index2word[1] = 'OOV' 
        vocabulary_word2index['EOS'] = 2 # a special token for biLstTextRelation model. which is used between two sentences.
        vocabulary_index2word[2] = 'EOS'
        for i, vocab in enumerate(model.wv.vocab):
            vocabulary_word2index[vocab] = i + 3
            vocabulary_index2word[i + 3] = vocab
        
        self.word2idx = vocabulary_word2index
        self.idx2word = vocabulary_index2word

    
    def save_tokenizer(self, word2idx_path, idx2word_path):
        pickle.dump(self.word2idx, open(word2idx_path, 'wb'))
        pickle.dump(self.idx2word, open(idx2word_path, 'wb'))

    def load_tokenizer(self, word2idx_path, idx2word_path):
        print('load tokenizer from %s' % word2idx_path)
        self.word2idx = pickle.load(open(word2idx_path, 'rb'))
        self.idx2word = pickle.load(open(idx2word_path, 'rb'))

    def embedding_matrix(self, word2vec_model_path, worddim):
        print('making embedding matrix')
        word_index = self.word2idx
        index_word = self.idx2word
        vocab_size = len(word_index) 
 
        word2vec_model = api.load(word2vec_model_path)
        word2vec_dict = {}
        for word, vector in zip(word2vec_model.wv.vocab, word2vec_model.wv.vectors):
            word2vec_dict[word] = vector
        
        bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
        
        
        embedding_matrix = np.zeros((len(word_index), worddim))
        embedding_matrix[0] = np.zeros(worddim)
        embedding_matrix[1] = np.random.uniform(-bound, bound, worddim)
        embedding_matrix[2] = np.random.uniform(-bound, bound, worddim)

        for word, i in word_index.items():
            if word in word2vec_dict:
                embedding_matrix[i] = word2vec_dict[word]
            else:
                print(word)
                continue

        return word_index, embedding_matrix
    
    def to_sequence(self, qmaxlen, cmaxlen):
        """ Turn string to sequence of indices represented by a numpy array.
        Arguments:
            maxlen: Pad the word sequence to maxlen.
        """
        self.qmaxlen = qmaxlen
        self.cmaxlen = cmaxlen
        for key in self.data:
            question = []
            comment = []
            for sentence in self.data[key][0]:
                qclist = sentence.strip().split('EOS')
                qwordlist = qclist[0].strip().split(' ')
                cwordlist = qclist[1].strip().split(' ')
                senlist = []
                for w in qwordlist: 
                    if w in self.word2idx: senlist.append(self.word2idx[w])
                    else: senlist.append(1)
                question.append(senlist)
                senlist = []
                for w in cwordlist:
                    if w in self.word2idx: senlist.append(self.word2idx[w])
                    else: senlist.append(1)
                comment.append(senlist)
            self.data[key][0] = (np.array(pad_sequences(question, maxlen=qmaxlen, value=0.0)),
                                 np.array(pad_sequences(comment, maxlen=cmaxlen, value=0.0)))
    
    
    def to_category(self):
        for key in self.data:
            if len(self.data[key]) == 2:
                self.data[key][1] = np.array(to_categorical(self.data[key][1]))
    
    def get_data(self, key):
        if key == 'train':
            return self.data[key][0], self.data[key][1]
        if key == 'test':
            return self.data[key][0], self.qidlist

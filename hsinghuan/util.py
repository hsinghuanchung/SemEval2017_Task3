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
        """ 
        self.data is a dictionary which stores training and testing data.
        Key:
            The key specifying the properties of different dataset, "train" or "test"
        Value:
            The value is a list [X,Y], where...
            X is a list of question comment pairs or question question pairs
            Y is a list of corresponding label. Y is None if the key is "test".

        self.categorymap is a dictionary which maps label to float value
        """
        self.data = {}
        if subtask == 'A' or subtask == 'C':
            self.categorymap = {'__label__Bad': 0.0, '__label__Good': 1.0, '__label__PotentiallyUseful': 0.0}
        elif subtask == 'B':
            self.categorymap = {'__label__Irrelevant': 0.0, '__label__Relevant': 1.0, '__label__PerfectMatch': 1.0}
    
    def add_data(self, key, data_path):
        """
        Reads training or testing data and store it into self.data.
        Arguments:
            key: "train" or "test"
            data_path: The training data file or testing data file

        """
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

    
    def tokenize(self, wordvec_path):
        """ 
        Assign each word with an index. Make two dictionaries, self.word2idx and self.idx2word. One maps word to index, the other maps index to word.
        Arguments:
            wordvec_path: The pretrained word embedding. I use 'glove-wiki-gigaword-100' or 'glove-wiki-gigaword-200'.
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
        """
        Serialize and save self.word2idx and self.idx2word.
        Arguments:
            word2idx_path: The location to store word2idx
            idx2word_path: The location to store idx2word
        """
        pickle.dump(self.word2idx, open(word2idx_path, 'wb'))
        pickle.dump(self.idx2word, open(idx2word_path, 'wb'))

    def load_tokenizer(self, word2idx_path, idx2word_path):
        """
        Load and deserialize word2idx and idx2word.
        Arguments:
            word2idx_path: The location of word2idx.pkl
            idx2word_path: The location of idx2word.pkl
        """
        print('load tokenizer from %s' % word2idx_path)
        self.word2idx = pickle.load(open(word2idx_path, 'rb'))
        self.idx2word = pickle.load(open(idx2word_path, 'rb'))

    def embedding_matrix(self, word2vec_model_path, worddim):
        """
        Make an embedding_matrix according to pretrained word embeddings and word2idx
        Arguments:
            word2vec_model_path: The pretrained word embedding. I use 'glove-wiki-gigaword-100' or 'glove-wiki-gigaword-200' 
            worddim: 100 or 200
        Returns:
            word_index: self.word2idx
            embedding_matrix: A 2D numpy array
        """
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
        """ Turn strings in self.data to sequences of indices represented by a numpy array.
        Arguments:
            qmaxlen: The maximum length of question sentence.
            cmaxlen: The maximum length of comment sentence.
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
        """
        Turn integer label into numpy array
        e.g. 0 -> np.array([1,0]), 1 -> np.array([0,1])
        """
        for key in self.data:
            if len(self.data[key]) == 2:
                self.data[key][1] = np.array(to_categorical(self.data[key][1]))
    
    def get_data(self, key):
        """
        Get training or testing data
        Arguments:
            key: "train" or test"
        """
        if key == 'train':
            return self.data[key][0], self.data[key][1]
        if key == 'test':
            return self.data[key][0], self.qidlist

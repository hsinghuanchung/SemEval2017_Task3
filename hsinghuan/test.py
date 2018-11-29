import gensim
import gensim.downloader as api
import numpy as np
import argparse
import tensorflow as tf
from util import DataManager    
import os
from sys import argv
from keras.models import load_model
from keras.backend import set_session

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
set_session(sess)


parser = argparse.ArgumentParser(description='Natural Language Matching')
parser.add_argument('modelname')
parser.add_argument('datapath')
parser.add_argument('--subtask', choices=['A', 'B', 'C'])
args = parser.parse_args()


def outputA(qidlist, result, modelname):
    """
        The function which writes prediction file of subtaskA
        Arguments:
            qidlist: A list of RELC_ID
            result: 2D numpy array, the output of the model
            modelname: String, modelname
        Outputs:
            subtaskA/result/[modelname]/res.pred
    """
    
    outdir = os.path.join("subtaskA","result", modelname)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    with open(os.path.join(outdir, "res.pred"), "w") as f:
        for idx, r in enumerate(result):
            print(r)
            label = np.argmax(r)
            relq_id = qidlist[idx].split('_')[0] + '_' + qidlist[idx].split('_')[1] 
            f.write(relq_id)
            f.write(' ')
            f.write(qidlist[idx]) 
            f.write(' 0 ')
            f.write(str(r[1]))
            f.write(' ')
            if label == 1:
                f.write('true')
            elif label == 0:
                f.write('false')
            f.write('\n')

def outputB(qidlist, result, modelname):
    """
        The function which writes prediction file of subtaskB
        Arguments:
            qidlist: A list of RELC_ID
            result: 2D numpy array, the output of the model
            modelname: String, modelname
        Outputs:
            subtaskB/result/[modelname]/res.pred
    """
    outdir = os.path.join("subtaskB","result", modelname)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    with open(os.path.join(outdir, "res.pred"), "w") as f:
        for idx, r in enumerate(result):
            print(r)
            label = np.argmax(r)
            relq_id = qidlist[idx].split('_')[0] 
            f.write(relq_id)
            f.write(' ')
            f.write(qidlist[idx]) 
            f.write(' 0 ')
            f.write(str(r[1]))
            f.write(' ')
            if label == 1:
                f.write('true')
            elif label == 0:
                f.write('false')
            f.write('\n')

def outputC(qidlist, result, modelname):
    """
        The function which writes prediction file of subtaskB
        Arguments:
            qidlist: A list of RELC_ID
            result: 2D numpy array, the output of the model
            modelname: String, modelname
        Outputs:
            subtaskC/result/[modelname]/res.pred
    """
    outdir = os.path.join("subtaskC", "result", modelname)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    with open(os.path.join(outdir, "res.pred"), "w") as f:
        for idx, r in enumerate(result):
            print(r)
            label = np.argmax(r)
            orgq_id = qidlist[idx].split('_')[0] 
            f.write(orgq_id)
            f.write(' ')
            f.write(qidlist[idx]) 
            f.write(' 0 ')
            f.write(str(r[1]))
            f.write(' ')
            if label == 1:
                f.write('true')
            elif label == 0:
                f.write('false')
            f.write('\n')

def main():
    """ Main function of test.py
    Arguments:
        modelname: String, name of the model
        datapath: The testing file
        subtask: String, "A" or "B" or "C"
    Outputs:
        subtask + [subtask]/result/[modelname]/res.pred
    """ 
    modelname = args.modelname
    datapath = args.datapath
    subtask = args.subtask 
    dm = DataManager(subtask)
    dm.load_tokenizer(os.path.join("subtask" + subtask, "models", modelname, "word2idx.pkl"), os.path.join("subtask" + subtask, "models", modelname, "idx2word.pkl"))
    dm.add_data("test", datapath)
    dm.to_sequence(40, 40)
    (test_Q, test_C), qidlist = dm.get_data("test")
    print("test_Q", test_Q[0:2])
    print("test_C", test_C[0:2])
    print("qidlist", qidlist[0:2])
    model = load_model(os.path.join("subtask" + subtask, "models", modelname, "model.h5"))
    result = model.predict([test_Q, test_C], batch_size = 128, verbose=1)
    print("result", result[0:2])
    if subtask == "A":
        outputA(qidlist, result, modelname)
    elif subtask == "B":
        outputB(qidlist, result, modelname)
    elif subtask == "C":
        outputC(qidlist, result, modelname)    

if __name__ == "__main__":
    main()

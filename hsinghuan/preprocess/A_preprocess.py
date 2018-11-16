"""
    Preprocesses data for subtaskA then outputs dictionaries and a text file
    Dictionary format
        qdict:
            key: question id
            value: (subject, body)
        cdict:
            key: question id
            value: [(comment0, relevancy_label0), (comment1, relevancy_label1), ...]

    Text file format
        subject body [EOS] comment0 [tab] __label__relevancy_label0
        subject body [EOS] comment0 [tab] __label__relevancy_label1
"""


import re
import os
import string
import pickle
from sys import argv
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
from A_parse import parse2015, parse2016 



class Splitter(object):
    """
    split the document into sentences and tokenize each sentence
    """
    def __init__(self):
        self.splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def split(self,text):
        """
        out : ['What', 'can', 'I', 'say', 'about', 'this', 'place', '.']
        """
        # split into single sentence
        sentences = self.splitter.tokenize(text)
        # tokenization in each sentences
        tokens = [self.tokenizer.tokenize(sent) for sent in sentences]
        return tokens


class LemmatizationWithPOSTagger(object):
    def __init__(self):
        pass
    def get_wordnet_pos(self,treebank_tag):
        """
        return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v) 
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            # As default pos in lemmatization is Noun
            return wordnet.NOUN

    def pos_tag(self,tokens):
        # find the pos tagginf for each tokens [('What', 'WP'), ('can', 'MD'), ('I', 'PRP') ....
        pos_tokens = [nltk.pos_tag(token) for token in tokens]

        # lemmatization using pos tagg   
        # convert into feature set of [('What', 'What', ['WP']), ('can', 'can', ['MD']), ... ie [original WORD, Lemmatized word, POS tag]
        pos_tokens = [ [(word, lemmatizer.lemmatize(word,self.get_wordnet_pos(pos_tag)), [pos_tag]) for (word,pos_tag) in pos] for pos in pos_tokens]
        return pos_tokens

lemmatizer = WordNetLemmatizer()
splitter = Splitter()
lemmatization_using_pos_tagger = LemmatizationWithPOSTagger()
STOP = {'ve', 't', 'o', 'y', 'an', 'd', 'm', 'll', 's', 're'}
URL_PAT = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', re.UNICODE)

def cleanse(text):
    text = text.lower()
    text = URL_PAT.sub(' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = " ".join([w for w in text.split() if w not in STOP])
    
    lemmatizer = WordNetLemmatizer()
    splitter = Splitter()
    lemmatization_using_pos_tagger = LemmatizationWithPOSTagger()

    #step 1 split document into sentence followed by tokenization
    tokens = splitter.split(text)

    #step 2 lemmatization using pos tagger 
    lemma_pos_token = lemmatization_using_pos_tagger.pos_tag(tokens)
    text = []
    for sent in lemma_pos_token:
        text = text + [after for prev, after, token in sent]
    text = " ".join(text)
    return text



def readall2015():
    qdict = {}
    cdict = {}
    qdict, cdict = parse2015("../../IRIE2018_project_1/training_data/SemEval2015-Task3-CQA-QL-dev-reformatted-excluding-2016-questions-cleansed.xml", qdict, cdict)
    qdict, cdict = parse2015("../../IRIE2018_project_1/training_data/SemEval2015-Task3-CQA-QL-train-reformatted-excluding-2016-questions-cleansed.xml", qdict, cdict)
    qdict, cdict = parse2015("../../IRIE2018_project_1/training_data/SemEval2015-Task3-CQA-QL-test-reformatted-excluding-2016-questions-cleansed.xml", qdict, cdict)
    return qdict, cdict

def readall2016():
    qdict = {}
    cdict = {}

    qdict, cdict = parse2016("../../IRIE2018_project_1/training_data/SemEval2016-Task3-CQA-QL-train-part1.xml", qdict, cdict)
    qdict, cdict = parse2016("../../IRIE2018_project_1/training_data/SemEval2016-Task3-CQA-QL-train-part2.xml", qdict, cdict)
    qdict, cdict = parse2016("../../IRIE2018_project_1/training_data/SemEval2016-Task3-CQA-QL-test.xml", qdict, cdict)
    qdict, cdict = parse2016("../../IRIE2018_project_1/training_data/SemEval2016-Task3-CQA-QL-dev.xml", qdict, cdict)

    return qdict, cdict


def main():
    taskAdir = argv[1]
    qdict2015, cdict2015 = readall2015()
    qdict2016, cdict2016 = readall2016()
    print("Start preprocessing")
    for q in qdict2015.keys():
        subject, body = qdict2015[q]
        if subject == None: print(q)
        subject = cleanse(subject)
        body = cleanse(body)
        qdict2015[q] = (subject, body)
    
    for q in cdict2015.keys():
        for idx, (comment, label) in enumerate(cdict2015[q]):
            comment = cleanse(comment)
            cdict2015[q][idx] = (comment, label) 
    
    for q in qdict2016.keys():
        subject, body = qdict2016[q]
        subject = cleanse(subject)
        body = cleanse(body)
        qdict2016[q] = (subject, body)

    for q in cdict2016.keys():
        for idx, (comment, label) in enumerate(cdict2016[q]):
            comment = cleanse(comment)
            cdict2016[q][idx] = (comment, label) 
    print(len(qdict2015) + len(qdict2016)) 
    print(len(cdict2015) + len(cdict2016))
    
    print("Serializing")
    if os.path.isdir(taskAdir) == False:
        os.mkdirs(taskAdir)
    pickle.dump(qdict2015, open(os.path.join(taskAdir, "qdict2015.pkl"), "wb")) 
    pickle.dump(cdict2015, open(os.path.join(taskAdir, "cdict2015.pkl"), "wb")) 
    pickle.dump(qdict2016, open(os.path.join(taskAdir, "qdict2016.pkl"), "wb")) 
    pickle.dump(cdict2016, open(os.path.join(taskAdir, "cdict2016.pkl"), "wb")) 

    print("Write text dataset")
    with open(os.path.join(taskAdir, "data.txt"), "w") as f:
        for q in cdict2015:
            subject, body = qdict2015[q]
            for comment, label in cdict2015[q]:
                f.write(subject)
                f.write(" ")
                f.write(body)
                f.write(" EOS ")
                f.write(comment)
                f.write(" \t ")
                f.write("__label__" + label)
                f.write(" \n")
        for q in cdict2016:
            subject, body = qdict2016[q]
            for comment, label in cdict2016[q]:
                f.write(subject)
                f.write(" ")
                f.write(body)
                f.write(" EOS ")
                f.write(comment)
                f.write(" \t ")
                f.write("__label__" + label)
                f.write(" \n")
if __name__ == "__main__":
    main()

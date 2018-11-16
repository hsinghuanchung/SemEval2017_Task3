"""
    Preprocesses data for subtaskB then outputs dictionaries and a text file
    Dictionary format
        orgqdict:
            key: question id
            value: (subject, body)
        relqdict:
            key: question id
            value: [((subject0, body0), relevancy_label0), ((subject1, body1), relevancy_label1), ...]

    Text file format
        orgq_subject0 orgq_body0 [EOS] relq_subject0 relq_body0 [tab] __label__relevancy_label0
        orgq_subject0 orgq_body0 [EOS] relq_subject1 relq_body1 [tab] __label__relevancy_label1
"""

import re
import os
import string
import pickle
import nltk
from sys import argv
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
from B_parse import parse2016 

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



def readall2016():
    orgqdict = {}
    relqdict = {}

    orgqdict, relqdict = parse2016("../../IRIE2018_project_1/training_data/SemEval2016-Task3-CQA-QL-train-part1.xml", orgqdict, relqdict)
    orgqdict, relqdict = parse2016("../../IRIE2018_project_1/training_data/SemEval2016-Task3-CQA-QL-train-part2.xml", orgqdict, relqdict)
    orgqdict, relqdict = parse2016("../../IRIE2018_project_1/training_data/SemEval2016-Task3-CQA-QL-test.xml", orgqdict, relqdict)
    orgqdict, relqdict = parse2016("../../IRIE2018_project_1/training_data/SemEval2016-Task3-CQA-QL-dev.xml", orgqdict, relqdict)
    
    return orgqdict, relqdict

def main():
    subtaskBdir = argv[1]
    orgqdict, relqdict = readall2016()
    print("Preprocessing")
    for q in orgqdict.keys():
        orgq_subject, orgq_body = orgqdict[q]
        orgq_subject = cleanse(orgq_subject)
        orgq_body = cleanse(orgq_body)
        orgqdict[q] = (orgq_subject, orgq_body)
    for q in relqdict.keys():
        for idx, ((relq_subject, relq_body), label) in enumerate(relqdict[q]):
            relq_subject = cleanse(relq_subject)
            relq_body = cleanse(relq_body)
            relqdict[q][idx] = ((relq_subject, relq_body), label) 
    
    print("Serializing")
    if os.path.isdir(subtaskBdir) == False:
        os.makedirs(subtaskBdir)
    pickle.dump(orgqdict, open(os.path.join(subtaskBdir, "orgqdict.pkl"), "wb")) 
    pickle.dump(relqdict, open(os.path.join(subtaskBdir, "relqdict.pkl"), "wb")) 
    
    print("Write text dataset")
    with open(os.path.join(subtaskBdir, "data.txt"), "w") as f:
        for q in orgqdict:
            orgsubject, orgbody = orgqdict[q]
            for (relsubject, relbody), label in relqdict[q]:
                f.write(orgsubject)
                f.write(" ")
                f.write(orgbody)
                f.write(" EOS ")
                f.write(relsubject)
                f.write(" ")
                f.write(relbody)
                f.write(" \t ")
                f.write("__label__" + label)
                f.write(" \n")

if __name__ == "__main__":
    main()

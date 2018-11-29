import xml.etree.ElementTree as ET
import re
from sys import argv
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet




def parsetest(filename, orgqdict, relqdict, orgqlist):
    tree = ET.parse(filename)
    root = tree.getroot()

    for child in root:
        orgqid = child.attrib["ORGQ_ID"]
        orgq_subject = child[0].text if child[0].text != None else ""
        orgq_body = child[1].text if child[1].text != None else ""
        thread =  child[2]
        relqid = thread.attrib["THREAD_SEQUENCE"]
        relq_subject = child[2][0][0].text if child[2][0][0].text != None else ""
        relq_body = child[2][0][1].text if child[2][0][1].text != None else ""
        
        if orgqid not in orgqdict:
            orgqlist.append(orgqid)
            orgqdict[orgqid] = (orgq_subject, orgq_body)
        if orgqid not in relqdict:
            relqdict[orgqid] = [(relqid, (relq_subject, relq_body))]
        else:
            relqdict[orgqid].append((relqid, (relq_subject, relq_body)))

    return orgqdict, relqdict, orgqlist 

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

if __name__ == "__main__":
    orgqdict, relqdict, orgqlist = {}, {}, []
    orgqdict, relqdict, orgqlist = parsetest(argv[1], orgqdict, relqdict, orgqlist)
    print(orgqdict["Q388"])
    print(relqdict["Q388"])
    
    with open(argv[2], "w") as f:
        for orgqid in orgqlist:
            orgq_subject, orgq_body = orgqdict[orgqid]
            orgq_subject, orgq_body = cleanse(orgq_subject), cleanse(orgq_body)
            relq = relqdict[orgqid]
            for relqid, (relq_subject, relq_body) in relq:
                relq_subject = cleanse(relq_subject)
                relq_body = cleanse(relq_body)
                f.write(relqid)
                f.write("\t")
                f.write(orgq_subject)
                f.write(" ")
                f.write(orgq_body)
                f.write(" EOS ")
                f.write(relq_subject)
                f.write(" ")
                f.write(relq_body)
                f.write("\n")
    


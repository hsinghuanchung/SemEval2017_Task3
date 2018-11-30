import sys
import codecs
import math
import xml.etree.ElementTree as ET
#from A_preprocess import readall2016

qdict = {}  #original query id to original query body dictionary
cdict = {}  # original  query  id to (related comment id ,related comment body)  dictionary

# parameters of BM25 model
K1 = 1      
b = 0.75
doc_len = {}      #length of comment
avg_doc_len = 1   #average length of comment 
N = 0        #total number of comments
Ni = {}       # total number of comments that a term appears in 

################ building cdict and qdict #########################################

def build_dict(filename):
    tree = ET.parse(filename)    # read in xml file 
    root = tree.getroot()
    for child in root :
    	orgq_id = child.attrib["ORGQ_ID"]
    	relq_id = child[2].attrib["THREAD_SEQUENCE"]
    	orgq_subject = child[0].text
    	orgq_body = child[1].text 
    	relq_subject = child[2][0][0].text
    	relq_body = child[2][0][1].text
    	qdict[orgq_id] = (orgq_subject , orgq_body )
    	comment_list = []
    	for i in range(1 , 11):
    		relc_id = child[2][i].attrib["RELC_ID"]
    		rel_comment = child[2][i][0].text.replace(',', '').replace('.', '').replace('?', '').replace('-', '').replace('#', '')
    		comment_list.append((relc_id , rel_comment))

    	if orgq_id in cdict.keys() :  cdict[orgq_id] += comment_list
    	else : 	cdict[orgq_id] = comment_list
    		
################ calculating term frequency #########################################
def term_freq(term , doc):
	temp = doc.split(' ')
	count = 0
	for w in temp:
		if w == term : count += 1 

	return count


################# scoring of a query and a comment #########################################
def scoring(query , doc):
	w = query.split(' ')
	score = 0
	for i in w:
		tf = term_freq(i , doc)
		n = (1 + K1) * tf
		d =((doc_len[doc] / avg_doc_len) * b + (1 - b)) * K1 + tf
		bb = n / d
		if i in Ni.keys() : 
			nn = math.log((N - Ni[i] + 0.5) / (Ni[i] + 0.5))
		else :
		    nn = math.log((N + 0.5) / 0.5) 	
		score += bb * nn
	return score
####################### main function #####################################		
def main():
	global N
	build_dict(sys.argv[1])
	total_len = 0	
      	

	for doc_list in cdict.values() :
	    for doc in doc_list: 
	    	N += 1
	    	total_len += doc[1].count(' ') + 1
	    	doc_len[doc[1].lower()] = doc[1].count(' ') + 1
	    	temp = doc[1].lower().split(' ')
	    	for w in set(temp):
	    		if w in Ni.keys() : Ni[w] += 1
	    		else : Ni[w] = 1

	avg_doc_len = total_len / N

###################   writing   output file ####################################################	
	output_file = open(sys.argv[2] , 'w')

	for key in qdict.keys() :
		score = []
		for  c in cdict[key]: 
			score.append((scoring(qdict[key][1].lower().replace(',', '').replace('.', '').replace('?', '').replace('-', '').replace('#', '') , c[1].lower()) , c[0]))
		#score.sort(reverse = True)
		for i in range(100):
			quality = 'false'
			if score[i][0] >= 0.5 : quality = 'true'
			output_file.write(key + ' ' +  str(score[i][1]) + ' ' +  '0' + ' ' + str(score[i][0]) + ' '  + quality + '\n')

if __name__ == "__main__":
    main()

   
	



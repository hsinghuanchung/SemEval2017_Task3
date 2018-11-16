import xml.etree.ElementTree as ET
from sys import argv

def parse2016(filename, orgqdict, relqdict):
    """
        orgqdict
            key: orgq_id, value: (orgqsubject, orgqbody)
        relqdict
            key: orgq_id, value: [((relqsubject, relqbody), relevancy label) ... ]
    """
    tree = ET.parse(filename)
    root = tree.getroot()

    for child in root:
        # Each child represents a new (original question, related question) pair
        orgq_id = child.attrib["ORGQ_ID"]
        
        orgq_subject = child[0].text if child[0].text != None else ""
        orgq_body = child[1].text if child[1].text != None else ""
        
        relq_subject = child[2][0][0].text if child[2][0][0].text != None else ""
        relq_body = child[2][0][1].text if child[2][0][1].text != None else ""
        
        label = child[2][0].attrib["RELQ_RELEVANCE2ORGQ"]
        
    
        if orgq_id not in orgqdict:
            orgqdict[orgq_id] = (orgq_subject, orgq_body)

        if orgq_id not in relqdict:
            relqdict[orgq_id] = [((relq_subject, relq_body), label)]
        else:
            relqdict[orgq_id].append( ((relq_subject, relq_body), label) )

    return orgqdict, relqdict

def main():
    orgqdict = {}
    relqdict = {}
    orgqdict, relqdict = parse2016("IRIE2018_project_1/training_data/SemEval2016-Task3-CQA-QL-train-part1.xml", orgqdict, relqdict)

if __name__ == "__main__":
    main()

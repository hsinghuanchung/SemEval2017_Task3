import xml.etree.ElementTree as ET
from sys import argv

def parse2015(filename, qdict, cdict):
    tree = ET.parse(filename)
    root = tree.getroot()

    for child in root:
        thread =  child.attrib["THREAD_SEQUENCE"]
        comment = []
        for grandchild in child:
            if grandchild.tag == "RelQuestion":
                q_subject = grandchild[0].text if grandchild[0].text != None else ""
                q_body = grandchild[1].text if grandchild[1].text != None else ""
            elif grandchild.tag == "RelComment":
                c_text = grandchild[0].text
                c_label = grandchild.attrib["RELC_RELEVANCE2RELQ"]
                comment.append((c_text, c_label))
        qdict[thread] = (q_subject, q_body)
        cdict[thread] = comment

    return qdict, cdict
def parse2016(filename, qdict, cdict):
    """
        Add original question, related question, comments of related question pairs and original, comments of related question pairs, each having different labels.
        
        Arguments:
            filename: Filenames of SemEval2016 files
            qdict: A dictionary with key being question id and value being (question subject, question body) tuple
            cdict: A dictionary with key being question id and value being a list of (comment text, relevancy label) tuple
        Returns:
            qdict and cdict with new information
    """
    
    tree = ET.parse(filename)
    root = tree.getroot()

    for child in root:
        # Each child represents a new (original question, related question) pair
        orgq_id = child.attrib["ORGQ_ID"]
        relq_id = child[2].attrib["THREAD_SEQUENCE"]
        orgq_comment = []
        relq_comment = []
        # get orgq_comment, relq_comment
        orgq_subject = child[0].text if child[0].text != None else ""
        orgq_body = child[1].text if child[1].text != None else ""
        DUPLICATE = True if "SubtaskA_Skip_Because_Same_As_RelQuestion_ID" in child[2].attrib else False 
        for rel in child[2]:
            if rel.tag == "RelQuestion":
                relq_subject = rel[0].text if rel[0].text != None else ""
                relq_body = rel[1].text if rel[1].text != None else ""
            elif rel.tag == "RelComment":
                c_text = rel[0].text
                orgq_c_label = rel.attrib["RELC_RELEVANCE2ORGQ"]
                orgq_comment.append((c_text, orgq_c_label))
                relq_c_label = rel.attrib["RELC_RELEVANCE2RELQ"]
                relq_comment.append((c_text, relq_c_label))

        if DUPLICATE is False:
            qdict[relq_id] = (relq_subject, relq_body)
            cdict[relq_id] = relq_comment
        
        if (orgq_id in qdict) != (orgq_id in cdict):
            print("WARNING qdict inconsistent with cdict")
        elif orgq_id not in qdict:
            qdict[orgq_id] = (orgq_subject, orgq_body)
            cdict[orgq_id] = relq_comment
        else:
            cdict[orgq_id] = cdict[orgq_id] + orgq_comment
    
    return qdict, cdict



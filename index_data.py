import sys
import pickle
reload(sys)
sys.setdefaultencoding("utf-8")
import os
from elasticsearch import Elasticsearch

INDEX_NAME = 'microarchitecture'

es = Elasticsearch(["http://localhost:9200"])

def index_name(doc):
    return INDEX_NAME

def index_document(doc_obj, _id):
    index = index_name(doc_obj)
    if index:
        es.index(index, "doc", doc_obj, id=_id)

with open("data/data.tsv", "rb") as n:
    bug_data = n.readlines()


for i in range(len(bug_data)):
    if i == 0:
        continue
    line = bug_data[i].replace("\n", "").split("\t")
    #print "LINE", line
    
    try:
        bug_obj = {}
        bug_obj["core"] = line[0]
        bug_obj["manufacturer"] = line[1]
        bug_obj["chip"] = line[2]
        bug_obj["details"] = line[3]
        bug_obj["workaround"] = line[4]
        bug_obj["failure"] = line[5]
        print "BUG OBJ", bug_obj
        index_document(bug_obj, bug_obj)
    except:
        print "Skipped"

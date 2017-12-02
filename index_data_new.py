import sys
import pickle
reload(sys)
sys.setdefaultencoding("utf-8")
import os
from elasticsearch import Elasticsearch

INDEX_NAME = 'microarchitecture_new'

es = Elasticsearch(["http://localhost:9200"])

def index_name(doc):
    return INDEX_NAME

def index_document(doc_obj, _id):
    index = index_name(doc_obj)
    if index:
        es.index(index, "doc", doc_obj, id=_id)

with open("data/newData.tsv", "rb") as n:
    bug_data = n.readlines()


for i in range(len(bug_data)):
    if i == 0:
        continue
    line = bug_data[i].strip().split("\t")
    if len(line) > 1:
        category = line[0]
        detail = line[1]
        print category, detail
        
        bug_obj = {}
        bug_obj["category"] = category
        bug_obj["detail"] = detail
        print "BUG OBJ", bug_obj
        try:
            index_document(bug_obj, bug_obj)
        except:
            print "SKIPPED"

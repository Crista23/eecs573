from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

vectorizer = TfidfVectorizer(stop_words='english')
INDEX ="microarchitecture_new"

es = Elasticsearch(["http://localhost:9200"])

def cluster_documents(k, document_list):
	X = vectorizer.fit_transform(document_list)
	true_k = k
	model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
	model.fit(X)
	print("Top terms per cluster:")
	order_centroids = model.cluster_centers_.argsort()[:, ::-1]
	terms = vectorizer.get_feature_names()
	for i in range(true_k):
		print("Cluster %d:" % i),
		for ind in order_centroids[i, :10]:
			print(' %s' % terms[ind]),
		print

query_text = {"query": { "match_all": {} } }
docs_retrieved = es.search(INDEX, body=query_text)
total_found_documents = docs_retrieved["hits"]["total"]
print "TOTAL DOCUMENTS", total_found_documents

docs_per_category = defaultdict(list)
bug_categories = []
bug_details = []
for doc in scan(es, query=query_text, index=INDEX):
	print "DOC", doc
	doc_category = doc["_source"]["category"]
	#print "category", doc_category
	doc_detail = doc["_source"]["detail"]
	#print "!!!doc_detail", doc_detail
	docs_per_category[doc_category].append(doc_detail)
	bug_categories.append(doc_category)
	bug_details.append(doc_detail)

#print 80 * "---"
#print "BUG CATEGORIES"
#print "CATEGORIES", len(bug_categories)
#print bug_categories
#cluster_documents(5, bug_categories, "overall_clusters")
#print "DETAILS", bug_details

#cluster_documents(5, bug_details)
#cluster_documents(10, bug_details)
#cluster_documents(15, bug_details)
#cluster_documents(20, bug_details)

print docs_per_category
for category, errors in docs_per_category.iteritems():
	print "category=", category
	cluster_documents(20, errors)




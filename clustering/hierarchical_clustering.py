from scipy.cluster.hierarchy import ward, dendrogram

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib as mpl
mpl.use('Agg')

from sklearn.manifold import MDS

from sklearn.cluster import KMeans
import nltk, re

from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
stemmer = SnowballStemmer("english")
wordnet_lemmatizer = WordNetLemmatizer()

es = Elasticsearch(["http://localhost:9200"])
INDEX ="microarchitecture_new"

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    #stems = [stemmer.stem(t) for t in filtered_tokens]
    lemmatized_tokens = [ wordnet_lemmatizer.lemmatize(t) for t in filtered_tokens ]
    return " ".join(lemmatized_tokens)


query_text = {"query": { "match_all": {} } }
docs_retrieved = es.search(INDEX, body=query_text)
total_found_documents = docs_retrieved["hits"]["total"]
print "TOTAL DOCUMENTS", total_found_documents

bug_details = []
titles = []
i = 0
for doc in scan(es, query=query_text, index=INDEX):
	doc_category = doc["_source"]["category"]
	titles.append(doc_category)
	doc_detail = doc["_source"]["detail"]
	clean_doc_detail = tokenize_and_stem(doc_detail)
	#clean_doc_detail = doc_detail
	bug_details.append(clean_doc_detail)
	i += 1
	if  i == 100:
		break
tfidf_matrix = tfidf_vectorizer.fit_transform(bug_details)
dist = 1 - cosine_similarity(tfidf_matrix)
num_clusters = 5
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()


print "CLUSTERS", clusters

print("Top terms per cluster:")
print()
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

names = []
terms = tfidf_vectorizer.get_feature_names()
for i in range(num_clusters):
	print("Cluster %d:" % i),
	top_terms = []
	for ind in order_centroids[i, :5]:
		print(' %s' % terms[ind]),
		top_terms.append(terms[ind])
	names.append(" ".join(top_terms))
    	

linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

fig, ax = plt.subplots(figsize=(25, 20)) # set size
ax = dendrogram(linkage_matrix, orientation="left", labels=terms, leaf_font_size=20)

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off',
    labelsize='large')

plt.tight_layout() #show plot with tight layout

#uncomment below to save figure
plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters
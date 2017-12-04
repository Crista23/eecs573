from gensim.models import Word2Vec

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib as mpl
mpl.use('Agg')

import nltk, re
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import PCA

from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
wordnet_lemmatizer = WordNetLemmatizer()

es = Elasticsearch(["http://localhost:9200"])
INDEX ="microarchitecture_new"

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word not in stop]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    #stems = [stemmer.stem(t) for t in filtered_tokens]
    lemmatized_tokens = [ wordnet_lemmatizer.lemmatize(t) for t in filtered_tokens ]
    return lemmatized_tokens


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
	#if  i == 10:
	#	break

model = Word2Vec(bug_details, min_count=200)

X = model[model.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)

plt.scatter(result[:, 0], result[:, 1])
words = list(model.vocab)
for i, word in enumerate(words):
	plt.annotate(word, xy=(result[i, 0], result[i, 1]), size=8, ha='left', va='bottom')

plt.xlim(-1, 1)
plt.ylim(-1, 1)

plt.tight_layout() #show plot with tight layout
plt.savefig("word2vec_vectors.png") #show the plot

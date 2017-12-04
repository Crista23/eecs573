from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib as mpl
mpl.use('Agg')

from sklearn.manifold import MDS
import pandas as pd

from sklearn.cluster import KMeans

import pickle

tfidf_vectorizer = TfidfVectorizer(stop_words='english')

es = Elasticsearch(["http://localhost:9200"])
INDEX ="microarchitecture_new"

query_text = {"query": { "match_all": {} } }
docs_retrieved = es.search(INDEX, body=query_text)
total_found_documents = docs_retrieved["hits"]["total"]
print "TOTAL DOCUMENTS", total_found_documents

bug_details = []
titles = []
for doc in scan(es, query=query_text, index=INDEX):
	doc_category = doc["_source"]["category"]
	titles.append(doc_category)
	doc_detail = doc["_source"]["detail"]
	bug_details.append(doc_detail)

tfidf_matrix = tfidf_vectorizer.fit_transform(bug_details)
dist = 1 - cosine_similarity(tfidf_matrix)
num_clusters = 5
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

print "CLUSTERS", clusters

"""
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

print "MDS"
pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
print "POS"
xs, ys = pos[:, 0], pos[:, 1]
print "xs=", xs
print "ys=", ys

print "Done!"

with open('xs.pickle', 'wb') as handle:
    pickle.dump(xs, handle)

with open('ys.pickle', 'wb') as handle:
    pickle.dump(ys, handle)
"""

with open("xs.pickle", "rb") as handle:
	xs = pickle.load(handle)

with open("ys.pickle", "rb") as handle:
	ys = pickle.load(handle)

## Visualization ##
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

cluster_names = {0: 'erratum, intel, performance, bit, error, software, event, processor, msr, incorrect', 
                 1: 'commercially, available, observed, erratum, intel, software, impact, behavior, operation, hang', 
                 2: 'pcie, link, state, power, package, processor, c6, exit, reset, erratum', 
                 3: 'instruction, page, vm, memory, bit, address, fault, exception, execution, set', 
                 4: 'hang, behavior, unpredictable, result, cause, erratum, check, machine, processor, lead'}

df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles))
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(25, 15)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
    
ax.legend(numpoints=1)  #show legend with only 1 point

#add label in x,y position with the label as the film title
#for i in range(len(df)):
#    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)

plt.tight_layout() #show plot with tight layout
plt.savefig("plot_clusters_nolabels.png") #show the plot

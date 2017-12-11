import csv
import pprint as pp
from pattern.text.en import singularize
import bokeh.plotting as bp
import lda.datasets
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
plt.switch_backend('agg')
plt.rc('font', size=16)
import matplotlib as mpl

mpl.use('Agg')

# https://shuaiw.github.io/2016/12/22/topic-modeling-and-tsne-visualzation.html
def load_csv_file(filename):
    stop_wds = set(stopwords.words('english'))
    add_stop_wds = {u'1',u'2',u'3',u'4',u'5',u'6',u'7',u'8',u'9',u'0',u'will',u'can',u'could',u'have',u'has',u'been', u'a', u'b', u'c', u'd',
                    u'e', u'f', u'g', u'h', u'i', u'j', u'k', u'l', u'm', u'n', u'o', u'p', u'q', u'r', u's', u't', u'u', u'v', u'w', u'x', u'y',
                    u'z', u'may', u'might', u'details', u'none',u'due', u'e', u'errors', u'error',u'erratum',u'data',u'datum'}

    stop_wds.update(add_stop_wds)

    stop_char = [chr(c) for c in range(256)]
    stop_char = [x for x in stop_char if not x.isalnum()]
    stop_char.remove(' ')
    stop_char.remove('_')
    stop_char = ''.join(stop_char)

    tokenizer = RegexpTokenizer(r'\w+')
    docs = []
    dictionary = dict()
    word_cnt = 0
    vocabulary = []

    prefix = []

    with open(filename) as tsvfile:
        content = csv.reader(tsvfile, delimiter='\t')

        for line in content:

            # first filter the dataset
            #if not line[0].startswith('workaround'):
            #    continue
            if line[0] == "":
                continue
            if line[1].startswith('none identified'):
                continue
            if line[1].strip() == "":
                continue
            tokens = filter(None, tokenizer.tokenize(line[1].lower().translate(None, stop_char)))
            terms  = []

            for i in tokens:
                try:
                    if not unicode(i) in stop_wds:
                       terms.append(singularize(i))
#                         terms.append(i)
                except:
                    continue
            if len(terms) == 0:
                continue

            if line[0].startswith("title"): prefix.append(0)
            if line[0].startswith("problem"): prefix.append(1)
            if line[0].startswith("implication"): prefix.append(2)
            if line[0].startswith("workaround"): prefix.append(3)



            for each_term in terms:
                if dictionary.has_key(each_term):
                    continue
                dictionary[each_term] = word_cnt
                vocabulary.append(each_term)
                word_cnt += 1

            docs.append(terms)

    doc_term_matrix = np.zeros((len(docs), len(vocabulary)))
    for i in range(len(docs)):
        for each_term in docs[i]:
            doc_term_matrix[i, dictionary[each_term]] += 1
    print len(prefix), len(doc_term_matrix)
    return (np.array(vocabulary), doc_term_matrix.astype(int), np.array(prefix))

def save_results():
    pass
    return

vocabulary, doc_term_matrix, prefix = load_csv_file('newdata.tsv')
print doc_term_matrix.shape


model = lda.LDA(n_topics=7, n_iter=2000, random_state=1)
doc_topic_matrix = model.fit_transform(doc_term_matrix)
topic_word = model.topic_word_
n_top_words = 10
topic_summary = [] # topic words for each topic
for i, topic_dist in enumerate(topic_word):
    topic_summary = vocabulary[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    # for i, each in enumerate(topic_summary):
    #     if each == 'bio': topic_summary[i] = 'bios'
    #     if each == 'addres': topic_summary[i] = 'address'
    print('Topic {}: {}'.format(i, ' '.join(topic_summary)))
word_topic_matrix = np.transpose(topic_word)

#find the average topic of each type of doc
print doc_topic_matrix.shape
for i in range(4):
    print type(doc_topic_matrix)
    prefix_type_idx = np.where(prefix == i)
    print('type ', format(i))
    print np.average(doc_topic_matrix[prefix_type_idx[0],:], axis = 0)


#####################################################################################################################
# filter words with strong topics
threshold = 0.001
filter_idx = np.amax(word_topic_matrix, axis=1) > threshold  # idx of doc that above the threshold
word_topic_matrix = word_topic_matrix[filter_idx]
new_vocab = vocabulary[filter_idx]
word_top1_topic = []  # top 1 topic for each word
word_top1_topic_mat = np.zeros(word_topic_matrix.shape) # top 1 topic matrix for each word
for i in xrange(word_topic_matrix.shape[0]):
    word_top1_topic.append(word_topic_matrix[i].argmax())
    word_top1_topic_mat[i, word_top1_topic[i]] = word_topic_matrix[i, word_top1_topic[i]]

# t-SNE embedding
tsne_model = TSNE(n_components=2, verbose=2,learning_rate = 500, random_state=0, init='pca')
tsne_lda_word = tsne_model.fit_transform(word_topic_matrix)


# visualization
topic_names = {  0: 'Topic 0:  ', 1: 'Topic 1:  ', 2: 'Topic 2:  ', 3: 'Topic 3:  ', 4: 'Topic 4:  ', 5: 'Topic 5:  ',
                 6: 'Topic 6:  ', 7: 'Topic 7:  ', 8: 'Topic 8:  ', 9: 'Topic 9:  '}
colormap = np.array([
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"
])
df = pd.DataFrame(dict(x=tsne_lda_word[:, 0], y=tsne_lda_word[:, 1], label=word_top1_topic))
groups = df.groupby('label')
fig, ax = plt.subplots(figsize=(25, 15))  # set size
ax.margins(0.02)


# annotate the point with top5 words in each topics
topic_top5_word = []
topn_words = 5
print word_topic_matrix.shape[1]
for i in xrange(word_top1_topic_mat.shape[1]):
    topic_top5_word.append(np.argsort(word_top1_topic_mat[:, i])[:-(topn_words+1):-1])

for i, each_topic in enumerate(topic_top5_word):
    tmp_str = [new_vocab[idx] for idx in each_topic]
    for id, each in enumerate(tmp_str):
        if each == 'bio': tmp_str[id] = 'bios'
        if each == 'addres': tmp_str[id] = 'address'
    topic_names[i] += ' '.join(tmp_str)
    # for idx in each_topic:
        # ax.annotate(new_vocab[idx], (tsne_lda_word[idx,0],tsne_lda_word[idx,1]))

for idx, each in groups:
    ax.plot(each.x, each.y, marker='o', linestyle='', ms=10,
            label=topic_names[idx], color=colormap[idx],
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    ax.tick_params(axis='y', which='both', left='off', top='off', labelleft='off')

ax.legend(numpoints=1)
plt.tight_layout()
plt.savefig("plot_topics.png")




###########################################################################################################################
# visualization for the docs
# threshold = 0.2
# filter_doc_idx = np.amax(doc_topic_matrix, axis=1) > threshold  # idx of doc that above the threshold
# doc_topic_matrix = doc_topic_matrix[filter_doc_idx]
# doc_top1_topic = []  # top 1 topic for each word
# for i in xrange(doc_topic_matrix.shape[0]):
#     doc_top1_topic.append(doc_topic_matrix[i].argmax())
#
# # t-SNE embedding
# tsne_model_2 = TSNE(n_components=2, verbose=2,learning_rate = 1000, random_state=0, init='pca')
# tsne_lda_doc = tsne_model_2.fit_transform(doc_topic_matrix)
#
#
#
# # visualization
# doc_names = {  0: 'Topic 0 docs', 1: 'Topic 1 docs', 2: 'Topic 2 docs', 3: 'Topic 3 docs', 4: 'Topic 4 docs', 5: 'Topic 5 docs',
#                  6: 'Topic 6 docs', 7: 'Topic 7 docs', 8: 'Topic 8 docs', 9: 'Topic 9 docs'}
# colormap = np.array([
#     "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
#     "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
#     "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
#     "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"
# ])
# df2 = pd.DataFrame(dict(x=tsne_lda_doc[:, 0], y=tsne_lda_doc[:, 1], label=doc_top1_topic))
# groups2 = df2.groupby('label')
# fig2, ax2 = plt.subplots(figsize=(25, 15))  # set size
# ax2.margins(0.02)
#
# for idx, each in groups2:
#     ax2.plot(each.x, each.y, marker='o', linestyle='', ms=7,
#             label=doc_names[idx], color=colormap[idx],
#             mec='none')
#     ax2.set_aspect('auto')
#     ax2.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
#     ax2.tick_params(axis='y', which='both', left='off', top='off', labelleft='off')
#
# ax2.legend(numpoints=1)
# plt.tight_layout()
# plt.savefig("plot_docs.png")
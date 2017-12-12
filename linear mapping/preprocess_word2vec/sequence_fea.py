from gensim.models import word2vec
import numpy as np
model = word2vec.Word2Vec.load('word2vec.model')

invalid_doc_id = set()
def extract_seq_fea(sequence):
    features = np.array([model.wv[word] for word in sequence])
    #max_fea = np.amax(features, axis=0)
    #min_fea = np.amin(features, axis=0)
    ave_fea = np.average(features, axis=0)
    #con_fea = np.concatenate((max_fea, min_fea, ave_fea))
    #return con_fea.tolist()
    return ave_fea

# with open('train_title.txt') as f:
#     corpus_fea = []
#     for line in f:  corpus_fea.append(extract_seq_fea(line.strip().split(' ')))
#     with open('train_title_fea.txt', 'w') as wf:
#         for each in corpus_fea:
#             wf.write(' '.join(str(e) for e in each)+'\n')


# with open('train_problem.txt') as f:
#     corpus_fea = []
#     for line in f:  corpus_fea.append(extract_seq_fea(line.strip().split(' ')))
#     with open('train_problem_fea.txt', 'w') as wf:
#         for each in corpus_fea:
#             wf.write(' '.join(str(e) for e in each)+'\n')
#
# with open('train_implication.txt') as f:
#     corpus_fea = []
#     for line in f:  corpus_fea.append(extract_seq_fea(line.strip().split(' ')))
#     with open('train_implication_fea.txt', 'w') as wf:
#         for each in corpus_fea:
#             wf.write(' '.join(str(e) for e in each)+'\n')

# with open('train_workaround.txt') as f:
#     corpus_fea = []
#     for line in f:  corpus_fea.append(extract_seq_fea(line.strip().split(' ')))
#     with open('train_workaround_fea.txt', 'w') as wf:
#         for each in corpus_fea:
#             wf.write(' '.join(str(e) for e in each)+'\n')

# with open('test_title.txt') as f:
#     corpus_fea = []
#     for line in f:  corpus_fea.append(extract_seq_fea(line.strip().split(' ')))
#     with open('test_title_fea.txt', 'w') as wf:
#         for each in corpus_fea:
#             wf.write(' '.join(str(e) for e in each)+'\n')

# with open('test_problem.txt') as f:
#     corpus_fea = []
#     for line in f:  corpus_fea.append(extract_seq_fea(line.strip().split(' ')))
#     with open('test_problem_fea.txt', 'w') as wf:
#         for each in corpus_fea:
#             wf.write(' '.join(str(e) for e in each)+'\n')

f_tr_p = open('train_problem.txt')
f_tr_i = open('train_implication.txt')


process_pool_tr_p = []
process_pool_tr_i = []


for line in f_tr_p:
    process_pool_tr_p.append(line.strip().split(' '))

for line in f_tr_i:
    process_pool_tr_i.append(line.strip().split(' '))

n = len(process_pool_tr_i)

for i in xrange(n):
    try:
        if process_pool_tr_i[i][0] == "" or process_pool_tr_p[i][0] == "":
            del process_pool_tr_p[i]
            del process_pool_tr_i[i]
    except:
        break

corpus_fea_tr_p = []
corpus_fea_tr_i = []

for each in process_pool_tr_p:
    corpus_fea_tr_p.append(extract_seq_fea(each))

for each in process_pool_tr_i:
    corpus_fea_tr_i.append(extract_seq_fea(each))

f_tr_i.close()
f_tr_p.close()



with open('train_problem_fea.txt', 'w') as wf:
    for each in corpus_fea_tr_p:
        wf.write(' '.join(str(e) for e in each)+'\n')


with open('train_implication_fea.txt', 'w') as wf:
    for each in corpus_fea_tr_i:
        wf.write(' '.join(str(e) for e in each)+'\n')


with open('test_problem.txt') as f:
    corpus_fea = []
    for line in f:  corpus_fea.append(extract_seq_fea(line.strip().split(' ')))
    with open('test_problem_fea.txt', 'w') as wf:
        for each in corpus_fea:
            wf.write(' '.join(str(e) for e in each)+'\n')



with open('filtered_train_problem.txt', 'w') as wf:
    for each in process_pool_tr_p:
        wf.write(' '.join(str(e) for e in each)+'\n')


with open('filtered_train_implication.txt', 'w') as wf:
    for each in process_pool_tr_i:
        wf.write(' '.join(str(e) for e in each)+'\n')
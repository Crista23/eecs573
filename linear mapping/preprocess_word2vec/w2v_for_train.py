from gensim.models import word2vec
import pprint

def get_corpus():
    corpus = []
    with open('train_title.txt') as f:
        for line in f:
            if line == '':
                continue
            corpus.append(line.strip().split(' '))
    with open('train_problem.txt') as f:
        for line in f:
            if line == '':
                continue
            corpus.append(line.strip().split(' '))
    with open('train_implication.txt') as f:
        for line in f:  
            if line == '':
                continue
            corpus.append(line.strip().split(' '))
    with open('train_workaround.txt') as f:
        for line in f:
            if line == '':
                continue
            corpus.append(line.strip().split(' '))
    with open('test_title.txt') as f:
        for line in f: 
            if line == '':
                continue
            corpus.append(line.strip().split(' '))
    with open('test_problem.txt') as f:
        for line in f:
            if line == '':
                continue
            corpus.append(line.strip().split(' '))
    with open('test_implication.txt') as f:
        for line in f:
            if line == '':
                continue
            corpus.append(line.strip().split(' '))
    with open('test_workaround.txt') as f:
        for line in f:  
            if line == '':
                continue
            corpus.append(line.strip().split(' '))
    return corpus

my_corpus = get_corpus()

model = word2vec.Word2Vec(my_corpus, alpha=0.025, min_alpha = 0.001, workers=1, size=100,window=7, min_count=0, iter=200)
model.save('word2vec.model')
model = word2vec.Word2Vec.load('word2vec.model')
pprint.pprint(model.most_similar(positive=['2133mhz'], topn = 5))

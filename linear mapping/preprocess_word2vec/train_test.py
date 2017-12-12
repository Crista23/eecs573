import string
import csv

train_idx = []
test_idx = []

with open('train_idx.txt') as f:
    line = f.readline()
    train_idx = filter(None, line.split('\t'))
    train_idx = map(int, train_idx)
    print len(train_idx)

with open('test_idx.txt') as f:
    line = f.readline()
    test_idx = filter(None, line.split('\t'))
    test_idx = map(int, test_idx)
    print len(test_idx)

stop_char = [chr(c) for c in range(256)]
stop_char = [x for x in stop_char if not x.isalnum()]
stop_char.remove(' ')
stop_char.remove('_')
stop_char = ''.join(stop_char)
title = []
problem = []
implication = []
workaround = []

with open('newdata.tsv') as tsvfile:
    content = csv.reader(tsvfile, delimiter='\t')
    for line in content:
        line[1] = filter(None, line[1].lower().translate(None, stop_char))
        if line[0].startswith('title'):
            title.append(line[1]+'\n')
        if line[0].startswith('problem'):
            problem.append(line[1] + '\n')
        if line[0].startswith('implication'):
            implication.append(line[1] + '\n')
        if line[0].startswith('workaround'):
            workaround.append(line[1] + '\n')

train_title = [title[x-1] for x in train_idx]
train_problem = [problem[x-1] for x in train_idx]
train_implication = [implication[x-1] for x in train_idx]
train_workaround = [workaround[x-1] for x in train_idx]

with open('train_title.txt','w') as f:
    for str in train_title:
        f.write(str)
with open('train_problem.txt','w') as f:
    for str in train_problem:
        f.write(str)
with open('train_implication.txt','w') as f:
    for str in train_implication:
        f.write(str)
with open('train_workaround.txt','w') as f:
    for str in train_workaround:
        f.write(str)


test_title = [title[x-1] for x in test_idx]
test_problem = [problem[x-1] for x in test_idx]
test_implication = [implication[x-1] for x in test_idx]
test_workaround = [workaround[x-1] for x in test_idx]

with open('test_title.txt','w') as f:
    for str in test_title:
        f.write(str)
with open('test_problem.txt','w') as f:
    for str in test_problem:
        f.write(str)
with open('test_implication.txt','w') as f:
    for str in test_implication:
        f.write(str)
with open('test_workaround.txt','w') as f:
    for str in test_workaround:
        f.write(str)


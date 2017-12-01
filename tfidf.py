import string
import csv
import math
from textblob import TextBlob as tb
import sys

def tf(word, blob):
    return (float)(blob.words.count(word)) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    divisor = 1
    n_contain = n_containing(word, bloblist)
    if n_contain > 0:
        divisor = n_contain
    return math.log((float)(len(bloblist)) / divisor)

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

reload(sys)
sys.setdefaultencoding('utf8')

data = {}
currCore = ""
# Read all the data in
file = open("data/newData.tsv", "r")
for line in file:
    line = line.strip().split("\t")
    core = line[0]
    manu = line[1]
    chip = line[2]
    detail = line[3].lower()
    workaround = line[4]
    failure = line[5]
    if chip <> "":
        if chip in data:
            data[chip] += "\n" + detail
        else:
            data[chip] = detail

dataList = []
for core in data:
    dataList.append(tb(data[core]))

for i, d in enumerate(dataList):
    print("Top words in document {}".format(i + 1))
    scores = {word: tfidf(word, d, dataList) for word in d.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:10]:
        print("{}\t{}".format(word, round(score, 5)))


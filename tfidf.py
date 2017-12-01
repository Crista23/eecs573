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

data = []
currCore = ""
# Read all the data in
file = open("data/NewData.tsv", "r")
for line in file:
    line = line.strip().split("\t")
    if len(line) > 1:
      category = line[0]
      detail = line[1]
      if category == "title":
        data.append(detail)

print len(data)
'''
for i, d in enumerate(dataList):
    print("Top words in document {}".format(i + 1))
    scores = {word: tfidf(word, d, dataList) for word in d.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:10]:
        print("{}\t{}".format(word, round(score, 5)))
'''

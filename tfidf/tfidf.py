import string
import csv
import math
from textblob import TextBlob as tb
from nltk.corpus import stopwords
import re
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
file = open("../data/NewDataTag.tsv", "rb")
for line in file:
    line = line.strip().split("\t")
    if len(line) > 1:
      category = line[0]
      detail = line[1].encode('ascii', errors='ignore')
      output = ""
      for word in detail.split():
          pair = word.split("|")
          if pair[0] not in stopwords.words("english"):
              output += word + " "
      if category == "problem":
        data.append(tb(output.strip()))
file.close()

file = open("tfidf_results.txt", "w")
for i, d in enumerate(data):
    file.write("Top words in document {}".format(i + 1))
    file.write("\n")
    scores = {word: tfidf(word, d, data) for word in d.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:11]:
        file.write("{}\t{}".format(word, round(score, 5)))
        file.write("\n")
file.close()

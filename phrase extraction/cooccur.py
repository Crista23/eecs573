import string
import sys
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.util import ngrams
import re

reload(sys)
sys.setdefaultencoding('utf8')

store = False
data = []

# Read all the data in
file = open("../data/NewData.tsv", "rb")
for line in file:
    line = line.strip().split("\t")
    if len(line) > 1:
        category = line[0]
        detail = line[1].encode('ascii', errors='ignore')
        if category == "problem":
            if "msr" in detail:
                store = True
            else:
                store = False
        if category == "workaround" and store == True:
            output = re.sub(r'[^\w\s]',' ', detail)
            output = nltk.word_tokenize(output)
            output = ngrams(output, 17)
            data.append(output)
file.close()


results = {}
for line in data:
    for word in line:
        if word in results:
            results[word] += 1
        else:
            results[word] = 1

sorted = sorted(results.items(), key=lambda x: x[1], reverse=True)
for word in sorted:
    if word[1] > 3:
        output = ""
        for w in word[0]:
            output += w + " "
        print output.strip()

import string
import sys
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.util import ngrams
import re
import difflib

reload(sys)
sys.setdefaultencoding('utf8')

data = {"pcie" : [], "gp" : [], "pebs" : [], "vm" : [], "bios" : [], "msr" : []}
file = open("../data/NewData.tsv", "rb")
for line in file:
    line = line.strip().split("\t")
    if len(line) > 1:
        category = line[0]
        detail = line[1].encode('ascii', errors='ignore').lower()
        detail = detail.split(".")
        if category == "title":
            for sen in detail:
                for key in data:
                    if key in sen:
                        sen = re.sub(r'[^\w\s]',' ', sen).strip()
                        data[key].append(sen)
file.close()

def overlap(s1, s2):
    s = difflib.SequenceMatcher(None, s1, s2)
    pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2)) 
    return s1[pos_a:pos_a+size]

cluster = {}
sens = data["pcie"]

for i in range(len(sens)):
    if i < len(sens):
        c1 = sens[i]
        cluster[c1] = []
        j = len(sens) - 1

        while j > i:
            c2 = sens[j]
            over = overlap(c1, c2)
            lenOver = len(over)
            lenMin = min(len(c1), len(c2))
            if (float)(lenOver) / lenMin >= 0.5:
                cluster[c1].append(c2)
                sens.remove(c2)
            j -= 1
   
for key in cluster:
    l = len(cluster[key])
    if l > 4:
        print l, "\t", key

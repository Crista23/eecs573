import string
import difflib

data = {"pcie" : [], "gp" : [], "pebs" : [], "vm" : [], "bios" : [], "msr" : []}

file = open("cooccur.tsv", "r")
for line in file:
    line = line.strip().split("\t")
    if line[1] == "workaround":
        data[line[0]].append(line[2])
file.close()

def overlap(s1, s2):
    s = difflib.SequenceMatcher(None, s1, s2)
    pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2)) 
    return s1[pos_a:pos_a+size]

for key in data:
    sens = data[key]
    cluster = {}
    print key

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

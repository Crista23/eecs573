import string
import operator

data = {}
file = open("workaround.txt", "r")
for line in file:
    line = line.lower().strip().split("\t")
    if len(line) > 1:
        word = line[0]
        if word in data:
            data[word] += 1
        else:
            data[word] = 1
file.close()

results = {}
sortData = sorted(data.items(), key=operator.itemgetter(1), reverse=True)
for key in sortData:
    if key[1] <= 50:
        pair = key[0].split("|")
        if len(pair) > 1:
            if pair[1] in results:
                results[pair[1]].append(pair[0])
            else:
                results[pair[1]] = [pair[0]]

for key in results:
    print key
    output = ""
    for word in results[key]:
        output += word + ", "
    print output.strip() + "\n"

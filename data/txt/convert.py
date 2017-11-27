import string
import re

data = []
bug = {"title" : "", "problem" : "", "implication" : "", "workaround" : ""}
t = "knl"
str = ""

file = open("25.txt", "r")
for line in file:
    line = line.lower().strip()
    if t + "1" in line or t + "2" in line or t + "3" in line or t + "4" in line or t + "5" in line or t + "6" in line or t + "7" in line or t + "8" in line or t + "9" in line:
        print line
        bug["title"] = (line.split("\t")[1]).strip()
    elif "problem:" in line:
        if str <> "":
            bug["title"] += str
        bug["problem"] = (line.split("problem:")[1]).strip()
        str = ""
    elif "implication:" in line:
        if str <> "":
            bug["problem"] += str
        bug["implication"] = (line.split("implication:")[1]).strip()
        str = ""
    elif "workaround:" in line:
        if str <> "":
            bug["implication"] += str
        bug["workaround"] = (line.split("workaround:")[1]).strip()
        str = ""
    elif "status:" in line:
        data.append(bug)
        bug = {"title" : "", "problem" : "", "implication" : "", "workaround" : ""}
        str = ""
    else:
        str += " " + line
file.close()

file = open("result.tsv", "w")
for b in data:
    b["title"] = re.sub(r'\s+', " ", b["title"])
    b["problem"] = re.sub(r'\s+', " ", b["problem"])
    b["implication"] = re.sub(r'\s+', " ", b["implication"])
    b["workaround"] = re.sub(r'\s+', " ", b["workaround"])
    
    file.write("title\t" + b["title"] + "\n")
    file.write("problem\t" + b["problem"] + "\n")
    file.write("implication\t" + b["implication"] + "\n")
    file.write("workaround\t" + b["workaround"] + "\n\n")
file.close()

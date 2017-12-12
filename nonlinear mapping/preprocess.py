import string
import numpy as np
import re
import csv

# dict = {'1':2,'2':3,'3':4,'4':5,'5':6,'6':7,'7':8,'8':9,'9':10,'0':11,'-':12}
# i_dict=[]
# src_list = []
# tgt_list = []
# with open('test_data') as f:
#     st = 13
#     lines= f.readlines()
#     for each_line in lines:
#         each_parts = re.split('\xe3\x80\x92|\xe3\x80\x80|\n', each_line)
#         src_str = list(each_parts[1])
#         tgt_str = [each_parts[2][i:i+3] for i in range(0, len(each_parts[2]), 3)]
#         src_list.append([dict[x] for x in src_str])
#         print src_str, tgt_str




def build_dict_src_tgt_from_tsv(src = 'problem', tgt = 'implication', filename = 'newdata.tsv'):
    dict = {}
    i_dict={}
    src_list = []
    tgt_list = []

    stop_char = [chr(c) for c in range(256)]
    stop_char = [x for x in stop_char if not x.isalnum()]
    stop_char.remove(' ')
    stop_char.remove('_')
    stop_char = ''.join(stop_char)

    with open(filename) as tsvfile:
        st = 2
        content = csv.reader(tsvfile, delimiter='\t')
        for line in content:
            if line[0].startswith(src):
                src_str = filter(None, line[1].lower().translate(None, stop_char).split(' '))
                pass
                for each in src_str:
                    if dict.has_key(each):
                        continue
                    else:
                        dict[each] = st
                        i_dict[st] = each
                        st+=1
                src_list.append([dict[x] for x in src_str])

            if line[0].startswith(tgt):
                tgt_str = filter(None, line[1].lower().translate(None, stop_char).split(' '))
                pass
                for each in tgt_str:
                    if dict.has_key(each):
                        continue
                    else:
                        dict[each] = st
                        i_dict[st] = each
                        st+=1
                tgt_list.append([dict[x] for x in tgt_str])
    dict['<stop>'] = 1
    dict['<pad>'] = 0
    i_dict[1] = '<stop>'
    i_dict[0] = '<pad>'

    return (dict, i_dict, src_list, tgt_list)



(dict, i_dict, src_list, tgt_list) = build_dict_src_tgt_from_tsv()


print src_list[3]
print tgt_list[3]

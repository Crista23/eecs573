
test_id = 1
sim_id = [1783 , 1295,  1733]

test_doc=[]
te_wf_p=open('test_problem.txt')
lines = te_wf_p.readlines()
test_doc.append(lines[test_id].strip())
te_wf_p.close()

te_wf_i=open('test_implication.txt')
lines = te_wf_i.readlines()
test_doc.append(lines[test_id].strip())
te_wf_i.close()


print('problem: ' + test_doc[0])
print('implication: ' + test_doc[1])
print("\nsimilar:")
for each_id in sim_id:
    tr_doc = []

    tr_wf_p=open('filtered_train_problem.txt')
    lines = tr_wf_p.readlines()
    tr_doc.append(lines[each_id].strip())
    tr_wf_p.close()

    tr_wf_i=open('filtered_train_implication.txt')
    lines = tr_wf_i.readlines()
    tr_doc.append(lines[each_id].strip())
    tr_wf_i.close()


    print('problem: ' + tr_doc[0])
    print('implication: ' + tr_doc[1])
    print('\n')

print('\n\n')

from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

vectorizer = TfidfVectorizer(stop_words='english')

def cluster_documents(k, document_list):
	X = vectorizer.fit_transform(document_list)
	true_k = k
	model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
	model.fit(X)
	print("Top terms per cluster:")
	order_centroids = model.cluster_centers_.argsort()[:, ::-1]
	terms = vectorizer.get_feature_names()
	for i in range(true_k):
		print("Cluster %d:" % i),
		for ind in order_centroids[i, :10]:
			print(' %s' % terms[ind]),
		print


with open("data/ARM7and9.csv") as f:
	ARM7and9_lines = f.readlines()
print len(ARM7and9_lines)
#print ARM7and9_lines

error_descriptions = []
arm = defaultdict(list)
for i in range(len(ARM7and9_lines)):
	if i ==0:
		continue
	line = ARM7and9_lines[i].replace("\n", "").split("\t")
	print "******LINE", line
	chip_id = line[0]
	details_core = line[1]
	print "!!!!!!!!DETAILS CORE", details_core
	errata_id = line[2]
	module_criticality = line[3]
	workaround = line[4]
	revisions_impacted = line[5]
	failure_fix_status = line[6]
	masks_affected = line[7]
	manufacturer = line[8]
	d = {"details_core": details_core, "errata_id": errata_id, "module_criticality": module_criticality, "workaround": workaround, "revisions_impacted": revisions_impacted, "failure_fix_status": failure_fix_status, "masks_affected": masks_affected, "manufacturer": manufacturer}
	arm[chip_id].append(d)
	#error_descriptions.append(details_core)
	#error_descriptions.append(workaround)
	error_descriptions.append(failure_fix_status)

print "ERROR DESCRIPTIONS", error_descriptions
#cluster documents overall
cluster_documents(5, error_descriptions)

"""
#cluster documents per chip id
for chip_id, values in arm.iteritems():
	print "chip_id=", chip_id
	chip_errors = []
	for item in values:
		chip_errors.append(d["details_core"])
	#print "chip_errors", chip_errors
	cluster_documents(1, chip_errors)
"""

with open("data/CortexA8.csv") as f:
	CortexA8_lines = f.readlines()
print len(CortexA8_lines)

error_descriptions = []
for i in range(len(CortexA8_lines)):
	if i ==0:
		continue
	line = CortexA8_lines[i].replace("\n", "").split("\t")
	print "******LINE", line
	revisions_impacted = line[0]
	manufacturer = line[1]
	chip = line[2]
	details = line[3]
	workaround = line[4]
	revisions_impacted = line[5]
	failure_fix_status = line[6]
	core = line[7]
	#d = {"details_core": details_core, "errata_id": errata_id, "module_criticality": module_criticality, "workaround": workaround, "revisions_impacted": revisions_impacted, "failure_fix_status": failure_fix_status, "masks_affected": masks_affected, "manufacturer": manufacturer}
	#arm[chip_id].append(d)
	#error_descriptions.append(details)
	#error_descriptions.append(workaround)
	error_descriptions.append(failure_fix_status)

print "ERROR DESCRIPTIONS", error_descriptions
#cluster documents overall
cluster_documents(5, error_descriptions)


with open("data/CortexA9.csv") as f:
	CortexA9_lines = f.readlines()
print len(CortexA9_lines)

error_descriptions = []
for i in range(len(CortexA9_lines)):
	#if i ==0:
	#	continue
	line = CortexA9_lines[i].replace("\n", "").split("\t")
	print "******LINE", line
	criticality = line[0]
	manufacturer = line[1]
	chip = line[2]
	linux_bsp_status = line[3]
	revisions_impacted = line[4]
	workaround = line[5]
	failure = line[6]
	details = line[7]
	fix_status = line[8]
	core = line[9]
	error_category = line[10]
	#d = {"details_core": details_core, "errata_id": errata_id, "module_criticality": module_criticality, "workaround": workaround, "revisions_impacted": revisions_impacted, "failure_fix_status": failure_fix_status, "masks_affected": masks_affected, "manufacturer": manufacturer}
	#arm[chip_id].append(d)
	error_descriptions.append(details)
	#error_descriptions.append(workaround)
	#error_descriptions.append(failure)

print "ERROR DESCRIPTIONS", error_descriptions
#cluster documents overall
cluster_documents(5, error_descriptions)



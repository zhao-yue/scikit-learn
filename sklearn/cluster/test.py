import numpy as np
import time
import datetime
import sys
import csv
from sklearn.cluster.k_means_ import KMeans,k_means
from mpi4py import MPI
start_time0 = time.time()
maxi=1000
filename='../../../onemillion.csv'
if(len(sys.argv)>2):
	filename=sys.argv[2]
if(len(sys.argv)>1):
	maxi=int(sys.argv[1])

with open(filename,'r') as f:
	reader = csv.reader(f)
	data = list(reader)
comm = MPI.COMM_WORLD
end_time0 = time.time()	
start_time1 = time.time()
data.pop(0)
data=data[0:maxi]
data=np.array(data).astype(np.float)
end_time1 = time.time()
start_time2 = time.time()
kmeans = KMeans(n_clusters=3, max_iter=600,random_state=35732643,n_init=1,verbose=False,precompute_distances=False,init='k-means++',algorithm='full').fit(data)
end_time2 = time.time()
#print(kmeans.labels_)
#print(kmeans.cluster_centers_)
'''
if comm.Get_rank()==0:
	np.savetxt("output12.txt",kmeans.labels_,fmt='%d')
'''
comm.Barrier()
if comm.Get_rank()==0:
	print ("Open file: %s sec; Preprocess data:%s sec \nExecution time parallel %s sec" % (end_time0-start_time0,end_time1-start_time1,end_time2 - start_time2))
	print ()

''' # A very simple test
X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print(kmeans.labels_)
print(kmeans.cluster_centers_)
'''

import numpy as np
import time
import datetime
import sys
import csv
from sklearn.cluster.k_means_ import KMeans,k_means
from mpi4py import MPI
import math
start_time1 = time.time()
maxi=1000
filename='../../../onethousand.csv'
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
if(len(sys.argv)>2):
	filename=sys.argv[2]
if(len(sys.argv)>1):
	maxi=int(sys.argv[1])
sample_size = math.ceil(maxi/comm.Get_size())
sample_from = comm.Get_rank()*sample_size
sample_to = sample_from+sample_size
if sample_to > maxi:
	sample_to = maxi

data = np.zeros((sample_to-sample_from, 3), dtype=np.float)
j=0
'''
with open(filename) as fp:
    for i, line in enumerate(fp):
        if i>=sample_from+1 and i < sample_to+1: #The first row is column name
            j = j+1
            data[j]=float(line.strip())
        elif i >= sample_to+1:
            break
'''

with open(filename,'r') as f:
	reader = csv.reader(f)
	reader.__next__()
	for row in reader:
		if reader.line_num-2 >=sample_from and reader.line_num-2 < sample_to: #The first row is column name	
			data[j]=row
			j = j+1	
end_time1 = time.time()
init_array = np.array([[-0.5,-0.3,0.2], [0,0.3,-0.1], [0.1,-0.5,-0.5]])
start_time2 = time.time()
kmeans = KMeans(n_clusters=3, max_iter=600,random_state=35732643,n_init=1,verbose=True,precompute_distances=False,init=init_array,algorithm='full').fit(data)
end_time2 = time.time()
#print(rank, "labels",kmeans.labels_)
print(rank,"main centers\n",kmeans.cluster_centers_)
'''
if comm.Get_rank()==0:
	np.savetxt("output1.txt",kmeans.labels_,fmt='%d')
if comm.Get_rank()==1:
	np.savetxt("output2.txt",kmeans.labels_,fmt='%d')
if comm.Get_rank()==2:
	np.savetxt("output3.txt",kmeans.labels_,fmt='%d')
if comm.Get_rank()==3:
	np.savetxt("output4.txt",kmeans.labels_,fmt='%d')
'''
'''
recvbuf=None
if rank == 0:
    recvbuf = np.empty([1000,], dtype=np.int32)
comm.Gather(kmeans.labels_, recvbuf, root=0)
if rank ==0:
	np.savetxt("output4.txt",recvbuf,fmt='%d')
'''
comm.Barrier()
if comm.Get_rank()==0:
	print ("Prepare: %s sec \nExecution time parallel %s sec" % (end_time1-start_time1,end_time2 - start_time2))
	print ()

''' # A very simple test
X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print(kmeans.labels_)
print(kmeans.cluster_centers_)
'''


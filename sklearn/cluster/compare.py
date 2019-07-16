import numpy as np
import time
import datetime
import sys
import csv
from sklearn.cluster.k_means_ import KMeans,k_means
from mpi4py import MPI
from sklearn.metrics.cluster import adjusted_rand_score

a = np.loadtxt("output1.txt",dtype=np.int32)
b = np.loadtxt("output00.txt",dtype=np.int32)
print('adjusted_rand_score',adjusted_rand_score(a,b))



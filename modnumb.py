

import numpy as np
import time
import pandas as pd
import os
import numba
from numba import jit
features = pd.read_csv('data/data copy/TCGA-PANCAN-HiSeq-801x20531/data.csv', index_col=0).to_numpy()
@jit
def compute_distances(xs, num_clusters,assignments):
    N = xs.shape[0]  # num sample points
    d = xs.shape[1]  # dimension of space
    np.random.seed(0)
    cids = np.random.choice(N, (num_clusters,), replace=False)
    centroids  = xs[cids, :]
    # Compute distances from sample points to centroids
    # all  pair-wise _squared_ distances
    cdists = np.zeros((N, num_clusters))
    for i in range(N):
        xi = xs[i, :]
        for c in range(num_clusters):
            cc  = centroids[c, :]
            dist = 0
            for j in range(d):
                dist += (xi[j] - cc[j]) ** 2
            cdists[i, c] = dist
    return cdists, assignments
@jit
def expectation_step(xs, num_clusters,cdists, assignments):
    N = xs.shape[0]  # num sample points
    d = xs.shape[1]  # dimension of space
    np.random.seed(0)
    cids = np.random.choice(N, (num_clusters,), replace=False)
    centroids  = xs[cids, :]
    num_changed_assignments = 0
    

    for i in range(N):
            # pick closest cluster
       
        cmin = 0
        mindist = np.inf
        for c in range(num_clusters):
            if cdists[i, c] < mindist:
                cmin = c
                mindist = cdists[i, c]
        if assignments[i] != cmin:
            num_changed_assignments += 1
        assignments[i] = cmin
    return num_changed_assignments, assignments

@jit
def maximization_step(xs, num_clusters, assignments):
    N = xs.shape[0]  # num sample points
    d = xs.shape[1]  # dimension of space
    np.random.seed(0)
    cids = np.random.choice(N, (num_clusters,), replace=False)
    centroids  = xs[cids, :]
    # Maximization step: Update centroid for each cluster

    for c in range(num_clusters):
        
        newcent =np.zeros(1,)
        clustersize = 0
        for i in range(N):
            if assignments[i] == c:
                newcent = newcent + xs[i, :]
                clustersize += 1
        newcent = newcent / clustersize
        centroids[c, :]  = newcent
    return centroids

def kmeans(xs, num_clusters=4):
    N = xs.shape[0]  # num sample points
    d = xs.shape[1]  # dimension of space
    assignments = np.zeros(N, dtype=np.uint8)

    
    while True:
        start=time.perf_counter()
        cdists, assignments= compute_distances(features, 4, assignments)
        num_changed_assignments, assignments = expectation_step(features, 4, cdists, assignments)
        centroids = maximization_step(features,4, assignments)
        if num_changed_assignments == 0:
            break
        end = time.perf_counter()
        
    return centroids, assignments
centroids, assignments = kmeans(features, 4)  





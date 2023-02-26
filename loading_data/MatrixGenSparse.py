# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a script file to generate the plot.
"""

# imports
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
import time
from itertools import repeat, chain
import networkx as nx
from scipy import sparse
import random
import scipy.sparse as ss
import sys

size_of_ground_set = 1000000

directory = "data_exp2_split/"


def save_csr_matrix(filename, matrix):
    """Save compressed sparse row (csr) matrix to file.

    Based on http://stackoverflow.com/a/8980156/232571

    """
    assert filename.endswith('.npz')
    attributes = {
        'data': matrix.data,
        'indices': matrix.indices,
        'indptr': matrix.indptr,
        'shape': matrix.shape,
    }
    #np.savez(filename, matrix)
    ss.save_npz(filename, matrix, compressed=False)

def load_csr_matrix(filename):
    """Load compressed sparse row (csr) matrix from file.

    Based on http://stackoverflow.com/a/8980156/232571

    """
    assert filename.endswith('.npz')
    loader = np.load(filename)
    args = (loader['data'], loader['indices'], loader['indptr'])
    matrix = ss.csr_matrix(args, shape=loader['shape'])
    return matrix


#########################################################################

BA_fname = "ba"+str(1)+"m_sparse_1_1.npz"
pathBA = directory+BA_fname
print("Starting BA Graph Gen...")
G = nx.barabasi_albert_graph(size_of_ground_set, 500, seed=42)
try:
    G.remove_edges_from(G.selfloop_edges())
except:
    G.remove_edges_from(nx.selfloop_edges(G))
print("BA Graph Generated")
BA = nx.to_scipy_sparse_matrix(G, format='csr') 
BA.setdiag(1)

print(BA.shape)
save_csr_matrix(pathBA, BA)
loaded_csr_matrix = ss.load_npz(pathBA)
print("Loaded Matrix: ",loaded_csr_matrix.shape)
BA = []

#########################################################################

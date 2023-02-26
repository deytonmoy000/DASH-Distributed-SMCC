import numpy as np
import networkx as nx
from scipy import sparse
import sys
import pandas as pd

in_file = sys.argv[1]
out_file = sys.argv[2]

print("Opening ", in_file)
A = np.loadtxt(in_file, delimiter=',')
print(A.shape)

sA = sparse.csr_matrix(A)
print(sA.shape)

# IF compressed = True; application may crash when data is decompressed during experiments
sparse.save_npz(out_file, sA, compressed=False) 
print("Saved to ", out_file)

sparse_matrix_input = sparse.load_npz(out_file)
print("Loading ", out_file)
print(sparse_matrix_input.shape)

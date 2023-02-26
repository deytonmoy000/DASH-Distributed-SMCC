import numpy as np
import networkx as nx
from scipy import sparse
import sys
import pandas as pd
import random

in_file = sys.argv[1]
out_file = sys.argv[2]
nSplits = sys.argv[3]

out_file_full = out_file+".npz"


G = nx.read_weighted_edgelist(in_file)
print(G.number_of_nodes(), G.number_of_edges())

try:
    G.remove_edges_from(G.selfloop_edges())
except:
    G.remove_edges_from(nx.selfloop_edges(G))

# G_relabelled = nx.convert_node_labels_to_integers(G)
# print(G_relabelled.number_of_nodes(), G_relabelled.number_of_edges())
V = list(range(G.number_of_nodes()))
random.Random(42).shuffle(V)
V_split = np.array_split(V, nSplits)

sparse_matrix = nx.to_scipy_sparse_matrix(G, format='csr')
sparse.save_npz(out_file, sparse_matrix, compressed=True)

for i in range(nSplits):
	out_file_split_i = out_file + str(i+1) + ".npz"
	V_split_i = list(V_split[i])
	sparse_matrix_i = sparse_matrix[V_split_i,:]
	sparse.save_npz(out_file_split_i, sparse_matrix_i, compressed=True)

sparse_matrix_input = sparse.load_npz(out_file)

print("Full Sparse matrix size:", sparse_matrix_input.shape)

for i in range(nSplits):
	out_file_split_i = out_file + str(i+1) + ".npz"
	sparse_matrix_input_i = sparse.load_npz(out_file_split_i)
	print(i+1, "th Split Sparse matrix size:", sparse_matrix_input_i.shape)
#sparse.save_npz('/tmp/sparse_matrix.npz', )
#nx.readwrite.adjlist.write_adjlist(G, out_file, delimiter=',')

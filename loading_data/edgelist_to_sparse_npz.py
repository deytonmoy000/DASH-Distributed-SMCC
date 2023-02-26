import numpy as np
import networkx as nx
from scipy import sparse
import sys
import pandas as pd

in_file = sys.argv[1]
out_file = sys.argv[2]

edgelist = pd.read_csv(in_file)
G = nx.from_pandas_edgelist(edgelist, source='FromNodeId', target='ToNodeId', edge_attr=None, create_using=None)
print(G.number_of_nodes(), G.number_of_edges())

try:
    G.remove_edges_from(G.selfloop_edges())
except:
    G.remove_edges_from(nx.selfloop_edges(G))

remove = [node for node,degree in dict(G.degree()).items() if degree < 1]

G.remove_nodes_from(remove)
G_relabelled = nx.convert_node_labels_to_integers(G)
print(G_relabelled.number_of_nodes(), G_relabelled.number_of_edges())
sparse_matrix = nx.to_scipy_sparse_matrix(G_relabelled, format='csr')
sparse.save_npz(out_file, sparse_matrix, compressed=True)
sparse_matrix_input = sparse.load_npz(out_file)

print(sparse_matrix_input.shape)
#sparse.save_npz('/tmp/sparse_matrix.npz', )
#nx.readwrite.adjlist.write_adjlist(G, out_file, delimiter=',')

import scipy.sparse as ss
import sys
import numpy as np 
in_file = sys.argv[1]
out_file = sys.argv[2]
def load_csr_matrix(filename):
    """Load compressed sparse row (csr) matrix from file.

       Based on http://stackoverflow.com/a/8980156/232571

    """
    assert filename.endswith('.npz')
    loader = np.load(filename)
    args = (loader['data'], loader['indices'], loader['indptr'])
    matrix = ss.csr_matrix(args, shape=loader['shape'])
    return matrix
sparse_matrix = load_csr_matrix(in_file)

ss.save_npz(out_file, sparse_matrix, compressed=False)
X = ss.load_npz(out_file)
print("Edges: ",X.count_nonzero())
print("Shape: ",X.shape)


import scipy.sparse as ss
import sys
import numpy as np 
in_file = sys.argv[1]
def load_csr_matrix(filename):
    """Load compressed sparse row (csr) matrix from file.

       Based on http://stackoverflow.com/a/8980156/232571

    """
    assert filename.endswith('.npz')
    loader = np.load(filename)
    args = (loader['data'], loader['indices'], loader['indptr'])
    matrix = ss.csr_matrix(args, shape=loader['shape'])
    return matrix
#sparse_matrix = load_csr_matrix(in_file)
sparse_matrix = ss.load_npz(in_file)
print("Edges: ",sparse_matrix.count_nonzero())
print("Shape: ",sparse_matrix.shape)


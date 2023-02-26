import numpy as np
import pandas as pd
import scipy.sparse as ss

def read_data_file_as_coo_matrix(filename='edges.txt'):
    "Read data file and return sparse matrix in coordinate format."
    data = pd.read_csv(filename, sep=',', header=None, dtype=np.uint64)
    rows = data[0]  # Not a copy, just a reference.
    cols = data[1]
    ones = np.ones(len(rows), np.uint32)
    matrix = ss.coo_matrix((ones, (rows, cols))) #for twitter(cols, rows) /// for friendster (rows,cols)
    return matrix

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
    np.savez(filename, **attributes)

def load_csr_matrix(filename):
    """Load compressed sparse row (csr) matrix from file.

    Based on http://stackoverflow.com/a/8980156/232571

    """
    assert filename.endswith('.npz')
    loader = np.load(filename)
    args = (loader['data'], loader['indices'], loader['indptr'])
    matrix = ss.csr_matrix(args, shape=loader['shape'])
    return matrix

import sys
if __name__ == '__main__':
    in_file = sys.argv[1]
    out_file = sys.argv[2]

    coo_matrix = read_data_file_as_coo_matrix(in_file)
    csr_matrix = coo_matrix.tocsr()
    print(csr_matrix.shape)
    #save_csr_matrix(out_file, csr_matrix)
    #loaded_csr_matrix = load_csr_matrix(out_file)
    
    ss.save_npz(out_file, csr_matrix, compressed=False)
    loaded_csr_matrix = ss.load_npz(out_file)
    print(loaded_csr_matrix.shape)
    # Comparison based on http://stackoverflow.com/a/30685839/232571
    assert (csr_matrix != loaded_csr_matrix).nnz == 0

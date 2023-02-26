import scipy.sparse
import sys

in_file = sys.argv[1]
sparse_matrix = scipy.sparse.load_npz(in_file)
dense_matrix = sparse_matrix.todense()
print(dense_matrix.shape)


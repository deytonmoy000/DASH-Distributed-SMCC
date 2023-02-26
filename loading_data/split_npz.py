import scipy.sparse as ss
import sys
import random
import numpy as np

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
    #loader = np.load(filename)
    #args = (loader['data'], loader['indices'], loader['indptr'])
    #matrix = ss.csr_matrix(args, shape=loader['shape'])
    matrix = ss.load_npz(filename)
    return matrix

directory="data_exp2_split/"
in_file = sys.argv[1]
out_file = sys.argv[2]
sparse_matrix = load_csr_matrix(in_file)
print(sparse_matrix.shape)
n = sparse_matrix.shape[0]

V = [ele for ele in range(n)]
random.Random(42).shuffle(V)
nSplits = int(sys.argv[3])

# Random partitioning of the entire data
# V_split = np.array_split(V,nSplits)

# Assinging elements to 'nSplits' machines uniformly at random
random.seed(42)
V_split = [[] for i in range(nSplits)]
for ele in V:
    x = random.randint(0, nSplits-1)
    V_split[x].append(ele)

for i in range(nSplits):
    X = sparse_matrix[V_split[i],:]
    print(i+1,":-",X.shape)
    out_filename = directory+out_file +"_"+ str(nSplits) +"_" +str(i+1)+".npz"
    save_csr_matrix(out_filename, X)
    Y = ss.load_npz(out_filename)
    print("Loaded Matrix: ",Y.shape)

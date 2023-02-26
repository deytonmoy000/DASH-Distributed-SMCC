"""
@author: Adam Breuer (FAST, Breuer et. al. 2019)
(Includes minor modifications )

"""


import numpy as np
from scipy import sparse



class InfluenceMaxSparse:  

    
    def __init__(self, A, p, nThreads):
        """
        class VertexCover:
            INPUTS and INSTANCE VARIABLES:
                A: (2D Numpy Array) a 2D SYMMETRIC np.array where each element in {0,1}, representing the adjacency matrix of the graph
                Note that if A has self-loops (diag elements) they will be IGNORED
                p: float in (0,1) the probability that a neighbor influences another
                groundset: a list of the ground set of elements with indices 1:nrow(A)
        """

        self.groundset = range(A.shape[0])
        self.A         = A
        self.p         = p
        self.name      = "InfMax"
        self.nThreads  = nThreads
        assert(sparse.issparse(A))
        # assert( np.sum(A - A.T) == 0 )
        # assert( np.all(A.diagonal() == 1) ) 




    def value(self, S):
        if not len(S):
            return(0)

        # assert(len(S)==len(np.unique(S))) # DELETE ME

        S = np.sort(list(set(S)))

        colsums = (self.A[S,:]).sum(axis=0)
        prob_influenced = 1.0 - (1.0 - self.p)**np.asarray(colsums).flatten()
        prob_influenced[S] = 1.0
        return np.sum( prob_influenced )

    


    def marginalval(self, S, T):
        if not len(S):
            return(0)
        if not len(T):
            return self.value(S)

        # S = list(np.unique(S))
        # T = list(np.unique(T))
        # assert(len(S)==len(np.unique(S)))
        # assert(len(T)==len(np.unique(T)))

        # if len(set(S).intersection(T)):
        #     print('!!SETS S AND T OVERLAP IN MARGINALVAL!!')
        #     S = np.sort(list(set(S) - set(T)))
        #     # raise(Exception)

        return self.value(list(set().union(S, T))) - self.value(T)


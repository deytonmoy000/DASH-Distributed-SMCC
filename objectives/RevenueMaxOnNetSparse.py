"""
@author: Adam Breuer (FAST, Breuer et. al. 2019)
(Includes minor modifications )

"""

import numpy as np
from scipy import sparse


class RevenueMaxOnNetSparse:  

    
    def __init__(self, A, alpha, nThreads):
        """
        Revenue Maximization (influence maximization with a concave function over influence):
            INPUTS and INSTANCE VARIABLES:
                A: (2D Numpy Array) a 2D SYMMETRIC np.array where each element is a weight, representing the adjacency matrix of the graph
                groundset: a list of the ground set of elements with indices 1:nrow(A)
                alpha: (float)  The coefficient of the square root on the value function

        """

        self.groundset = range(A.shape[0])
        self.A         = A
        self.alpha     = alpha 
        self.name      = "RevMax"
        self.nThreads  = nThreads
        assert(sparse.issparse(A))
        # if sparse.issparse(A):
        #     # assert( np.sum(A - A.T) == 0 )
        #     assert( np.all(A.diagonal() == 0) ) 
        # else:
        #     # assert( (A == A.T).all() )
        #     assert( (sum(np.diag(A)) == 0) )




    def value(self, S):
        if not len(S):
            return(0)
        S = np.sort(list(set(S)))
        colsums = (self.A[S,:]).sum(axis=0)
        return np.sum( np.asarray(colsums).flatten()**self.alpha )


    def marginalval(self, S, T):
        """ Marginal value of adding set S to current set T for function above """
        if not len(S):
            return(0)
        if not len(T):
            return self.value(S)

        SuT = list(set().union(S, T))
        return( self.value(SuT) - self.value(T) ) 

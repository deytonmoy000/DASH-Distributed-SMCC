"""
@author: Adam Breuer (FAST, Breuer et. al. 2019)
(Includes minor modifications )

"""


import numpy as np
from scipy import sparse



class NetCoverSparse:  

    
    def __init__(self, A, nThreads, V=[]):
        """
        class NetCoverSparse:
            INPUTS and INSTANCE VARIABLES:
                A: (a sparse CSR array representing a SYMMETRIC network where each element in {0,1}
                Note that if A has self-loops (diag elements) they will be IGNORED
                groundset: a list of the ground set of elements with indices 1:nrow(A)
        """

        
        self.groundset      = range(A.shape[0])
        if(V==[]):
            self.realset    = [ ele for ele in self.groundset ]
        else:
            self.realset    = V
        self.A              = A.tocsr()
        self.Cov            = []
        self.name           = "NetCov"
        self.nThreads       = nThreads
        
        assert(sparse.issparse(A))
        # assert( np.sum(A - A.T) == 0 )
        # assert( np.all(A.diagonal() == 1) ) 




    def value(self, S):
        if not len(S):
            return(0)

        S = list(np.unique(S))
        # assert(len(S)==len(np.unique(S))) # DELETE ME

        S = np.sort(list(set(S)))
        return self.A[S,:].max(0).sum()


    

    def marginalval(self, S, T):
        # Fast approach
        if not len(S):
            return(0)
        if not len(T):
            return self.value(S)

        S = list(np.unique(S))
        # T = list(np.unique(T))
        
        # assert(len(S)==len(np.unique(S)))
        # assert(len(T)==len(np.unique(T)))

        # if len(set(S).intersection(T)):
        #     print('!!SETS S AND T OVERLAP IN MARGINALVAL!!')
        #     S = np.sort(list(set(S) - set(T)))
        #     # raise(Exception)

        return self.value(list(set().union(S, T))) - self.value(T)
    
    
    """
    Optimized gain
    """
    
    def value_opt(self, S):
        if not len(self.Cov):
            return(0)
    
        assert(len(self.Cov)==len(np.unique(self.Cov)))
    
        try:
            return len(self.Cov)
    
        except:
            print('Cov', self.Cov)
            raise(Exception)
    
    
    def marginalval_opt(self, S, T, Cov):
        # Fast approach
        if not len(S):
            return(0)
        
        # Cov = np.unique(np.array(np.nonzero(self.A[T,:]))[1]).tolist()
        # S_cov = np.array(np.nonzero(self.A[S,:])).flatten().tolist()
        S_cov = np.unique(np.array(np.nonzero(self.A[S,:]))[1]).tolist()
        
        if not len(Cov):
            return len(S_cov)

        totalCov = list( set().union(Cov, S_cov, S) );
        
        return len(totalCov) - len(Cov)
    
    
    
    def updateCov(self, S):
        
        Cov = []
        if not len(S):
            return Cov
        
        # S_cov = np.array(np.nonzero(self.A[S,:])).flatten().tolist()
        S_cov = np.unique(np.array(np.nonzero(self.A[S,:]))[1]).tolist()
        
        
        Cov = list( set().union(S_cov, S) );
        
        return Cov
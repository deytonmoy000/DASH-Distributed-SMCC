#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''
import concurrent.futures
import os
# TO USE SPECIFIED NUMBER OF THREADS *nThreads* and not parallelize internally by python
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from datetime import datetime
import numpy as np
import random
from queue import PriorityQueue
# from scipy import sparse

try:
    from mpi4py import MPI
except ImportError:
    MPI = None 
if not MPI:
    print("MPI not loaded from the mpi4py package. Serial implementations will function, \
            but parallel implementations will not function.")

def check_inputs(objective, k):
    '''
    Function to run basic tests on the inputs of one of our optimization functions:
    '''
    # objective class contains the ground set and also value, marginalval methods
    assert( hasattr(objective, 'groundset') )
    assert( hasattr(objective, 'value') )
    assert( hasattr(objective, 'marginalval') )
    # k is greater than 0
    # print("k=",k, " len(objective.groundset) =",len(objective.groundset)  )
    assert( k>0 )
    # k is smaller than the number of elements in the ground set
    assert( k<=len(objective.groundset) )
    # the ground set contains all integers from 0 to the max integer in the set
    assert( np.array_equal(objective.groundset, list(range(np.max(objective.groundset)+1) )) )

def sample_seq( X, k, randstate ):
    if len(X) <= k:
        randstate.shuffle(X)
        return X
    Y = list(randstate.choice(X, k, replace=False));
    randstate.shuffle(Y);
    return Y;

def parallel_margvals_returnvals_fls(objective, L, N, comm, rank, size):
    '''
    Parallel-compute the marginal value f(element)-f(L) of each element in set N. Version for stochasticgreedy_parallel.
    Returns the ordered list of marginal values corresponding to the list of remaining elements N
    '''
    # Scatter the roughly equally sized subsets of remaining elements for which we want marg values
    comm.barrier()
    N_split_local = np.array_split(N, size)[rank]

    # Compute the marginal addition for each elem in N, then add the best one to solution L; remove it from remaining elements N
    valL = objective.value( L );
    ele_vals_local_vals = [ objective.value( list( set().union([elem], L ) ) ) - valL for elem in N_split_local ]
    
    # Gather the partial results to all processes
    ele_vals = comm.allgather(ele_vals_local_vals)
    return [val for sublist in ele_vals for val in sublist]


'''
Thread Implementations of Utility functions
'''
def margvals_returnvals(objective, L, N):
    '''
    Compute the marginal value f(element)-f(L) of each element in set N. Version for stochasticgreedy_parallel.
    Returns the ordered list of marginal values corresponding to the list of remaining elements N
    '''
    ele_vals_local_vals = [ objective.marginalval( [elem], L ) for elem in N ]
    
    return ele_vals_local_vals

def parallel_margvals_returnvals_thread(objective, L, N, nthreads=16):
    '''
    Parallel-compute the marginal value f(element)-f(L) of each element in set N. Version for stochasticgreedy_parallel.
    Returns the ordered list of marginal values corresponding to the list of remaining elements N
    '''
    #Dynamically obtaining the number of threads fr parallel computation
    # nthreads_rank = multiprocessing.cpu_count()
    nthreads_obj = objective.nThreads
    N_split_local = np.array_split(N, nthreads_obj)
    #with concurrent.futures.ProcessPoolExecutor(max_workers=nthreads) as executor:
    with concurrent.futures.ThreadPoolExecutor(max_workers=nthreads_obj) as executor:
        futures = [executor.submit(margvals_returnvals, objective, L, split) for split in N_split_local]
        return_value = [f.result() for f in futures]
    A = []
    for i in range(len(return_value)):
        A.extend(return_value[i])
    # return return_value
    return A

    # return return_value

def parallel_margvals_returnvals(objective, L, N, comm, rank, size):
    '''
    Parallel-compute the marginal value f(element)-f(L) of each element in set N. Version for stochasticgreedy_parallel.
    Returns the ordered list of marginal values corresponding to the list of remaining elements N
    '''
    # Scatter the roughly equally sized subsets of remaining elements for which we want marg values
    comm.barrier()
    N_split_local = np.array_split(N, size)[rank]

    ele_vals_local_vals = parallel_margvals_returnvals_thread(objective, L, list(N_split_local), nthreads=16)
    # Compute the marginal addition for each elem in N, then add the best one to solution L; remove it from remaining elements N
    # ele_vals_local_vals = [ objective.marginalval( [elem], L ) for elem in N_split_local ]
    
    # Gather the partial results to all processes
    ele_vals = comm.allgather(ele_vals_local_vals)
    return [val for sublist in ele_vals for val in sublist]

    # return ele_vals

def val_of_sets_returnvals(objective, sets):
    '''
    Compute the marginal value f(element)-f(L) of each element in set N. Version for stochasticgreedy_parallel.
    Returns the ordered list of marginal values corresponding to the list of remaining elements N
    '''
    set_vals_local_vals = [ objective.value( myset ) for myset in sets ]
    
    return set_vals_local_vals

def parallel_val_of_sets_thread(objective, list_of_sets, nthreads=16):
    nthreads_rank = objective.nThreads
    ''' Parallel-compute the value f(S) of each set (sublist) in list_of_sets, return ordered list of corresponding values f(S) '''
    list_split_local = np.array_split(list_of_sets, nthreads_rank)
    # Reduce the partial results to the root process
    #with concurrent.futures.ProcessPoolExecutor(max_workers=nthreads) as executor:
    with concurrent.futures.ThreadPoolExecutor(max_workers=nthreads_rank) as executor:
        futures = [executor.submit(val_of_sets_returnvals, objective, split) for split in list_split_local]
        return_value = [f.result() for f in futures]
    A = []
    for i in range(len(return_value)):
        A.extend(return_value[i])
    # return return_value
    return A
    # return return_value

def parallel_val_of_sets_returnvals(objective, N, comm, rank, size):
    '''
    Parallel-compute the marginal value f(element)-f(L) of each element in set N. Version for stochasticgreedy_parallel.
    Returns the ordered list of marginal values corresponding to the list of remaining elements N
    '''
    # Scatter the roughly equally sized subsets of remaining elements for which we want marg values
    comm.barrier()
    N_split_local = np.array_split(N, size)[rank]
    ele_vals_local_vals = parallel_val_of_sets_thread(objective, list(N_split_local), nthreads=16)
    # Compute the marginal addition for each elem in N, then add the best one to solution L; remove it from remaining elements N
    # ele_vals_local_vals = [ objective.marginalval( [elem], L ) for elem in N_split_local ]
    
    # Gather the partial results to all processes
    ele_vals = comm.allgather(ele_vals_local_vals)
    return [val for sublist in ele_vals for val in sublist]




'''
Additional Utility functions
'''
def make_lmbda(eps, k, n):
    '''
    Generate the vector lambda based on LINE 9 in ALG. 1
    '''
    idx = 1;
    lmbda = [ idx ];
    while (idx < n - 1):
        if ( idx < k ):
            newidx = np.floor(idx*(1 + eps));
            if (newidx == idx):
                newidx = idx + 1;
            if (newidx > k):
                newidx = k;
        else:
            newidx = np.floor(idx + eps * k);
            if (newidx == idx):
                newidx = idx + 1;
            if (newidx > n - 1):
                newidx = n - 1;

        lmbda.append( int(newidx) );
        idx = newidx;
    return lmbda

def adaptiveAdd_thread(lmbda, idcs, V, S, objective, eps, k, tau):
    B = []
    if ( len( idcs ) > 0 ):
        pos=lmbda[ idcs[ 0 ] ];
        ppos=0
        if pos > 1:
            ppos=lmbda[ idcs[0] - 1 ];

        tmpS = list( set().union( V[0 : ppos], S) );
        valTmpS = objective.value( tmpS );
        Ti=list(set(V[ppos:pos]))

        for idx in range(1,len(idcs) + 1):
            tmpS = list( set(tmpS) | set( Ti) );
            
            gain= objective.value( tmpS ) - valTmpS;
            if (tau == 0):
                thresh = len(Ti)*(1-eps)*valTmpS / np.float(k);
            else:
                thresh = len(Ti)*(1-eps)*tau;
                
            if (gain >= thresh):
                B.append(True)
            else:
                B.append(False)

            valTmpS = valTmpS + gain;
            if (idx >= len(idcs)):
                posEnd = lmbda[ idcs[ -1 ] ];
            else:
                posEnd = lmbda[ idcs[ idx ] ];
            Ti=V[ lmbda[ idcs[ idx - 1] ]: posEnd ];

    
    # Gather the partial results to all processes
    

    return B

def parallel_adaptiveAdd_thread(lmbda, V, S, objective, eps, k, tau = 0):
    
    '''
    Parallel-compute the marginal value of block  of elements and set the corresponding flag to True based on condition in LINE 13 of ALG. 1
    '''
    #Dynamically obtaining the number of threads fr parallel computation
    # nthreads_rank = multiprocessing.cpu_count()
    nthreads = objective.nThreads
    idcs = np.array_split( range( len( lmbda ) ), nthreads )
    #lmbda_split_local = np.array_split(lmbda, size)[rank]
    #with concurrent.futures.ProcessPoolExecutor(max_workers=nthreads) as executor:
    with concurrent.futures.ThreadPoolExecutor(max_workers=nthreads) as executor:
        futures = [executor.submit(adaptiveAdd_thread, lmbda, split, V, S, objective, eps, k, tau) for split in idcs]
        return_value = [f.result() for f in futures]

    B = []
    for i in range(len(return_value)):
        B.extend(return_value[i])
    return B

def parallel_adaptiveAdd(lmbda, V, S, objective, eps, k, comm, rank, size, tau = 0):
    
    '''
    Parallel-compute the marginal value of block  of elements and set the corresponding flag to True based on condition in LINE 13 of ALG. 1
    '''
    
    # Scatter the roughly equally sized subsets of remaining elements for which we want marg values
    comm.barrier()
    # idcs = np.array_split( range( len( lmbda ) ), size )[ rank ]
    lmbda_split_local = np.array_split(lmbda, size)[rank]

    B = parallel_adaptiveAdd_thread(list(lmbda_split_local), V, S, objective, eps, k, tau)

    # Gather the partial results to all processes
    B_vals = comm.allgather(B)

    return [val for sublist in B_vals for val in sublist]
    

# Utility functions for RDASH and GDASH

def LAT_SingleNode(V, S, V_all, V_ground, q, objective, tau, eps, delta, k, pastGains):
    
    '''
    The parallelizable greedy algorithm LAT using Single Node execution for Fixed Threshold 'tau'
    PARALLEL IMPLEMENTATION
    
    INPUTS:
    list V - contains the elements currently in the groundset
    list S - contains the elements currently in the solution set
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    float tau -- the fixed threshold
    float eps -- the error tolerance between 0 and 1
    float delta -- the probability tolerance between 0 and 1
    int k -- the cardinality constraint (must be k>0)
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())
    randstate -- random seed to use when drawing samples 
    pastGains -- each element is the current marginal gain of the corresponding element in groundset

    OUTPUTS:
    list pastGains -- each element is the current marginal gain of the corresponding element in groundset
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    list S -- the solution, where each element in the list is an element with marginal values > tau.

    '''  
    
    queries = 0    
    if(len(S)==k):
        return [pastGains, S, queries]
    ell = int(np.ceil( (4.0 + 8.0/ eps ) * np.log( len(V) / delta ) ) ) + 1;
    V = list( np.sort( list( set(V)-set(S) ) ) );
    for itr in range( ell+1 ): # The additional iteration for ParallelAlgo_PGB
        s = np.min( [k - len(S), len(V) ] );

        q.shuffle(V_ground)
        tmp = [ele for ele in V_ground if ele in V]
        V = tmp

        # seq = sample_seq( V, s, randstate );
        seq = V[0:s]

        lmbda = make_lmbda( eps, s, s );
        
        B = parallel_adaptiveAdd_thread(lmbda, seq, S, objective, eps, k, tau)
        #Added query increment
        queries += len(lmbda)
        
        # ## PREVIOUS VERSION of LAT Ran the Following ()
        # lmbda_star = lmbda[0]
        # if len(B) > 1:
        #     for i in range(1,len(B)):
        #         if(B[i]):
        #             if (i == len(B) - 1):
        #                 lmbda_star = lmbda[-1];
        #         else:
        #             lmbda_star = lmbda[i]
        #             break;

        # T = set(seq[0:lmbda_star]);

        
        # for i in range(lmbda_star, len(B)):
        #     if (B[i]):
        #         T = set().union(T, seq[ lmbda[ i - 1 ] : lmbda[ i ] ]);
        

        ### UPDATED Version stops at the first FALSE in Lambda or selects the last value in LAMBDA
        lmbda_star = lmbda[-1]
        if len(B) > 1:
            for i in range(1,len(B)):
                if(B[i]):
                    continue
                else:
                    lmbda_star = lmbda[i]
                    break;

        T = set(seq[0:lmbda_star]);

        T = list( set(T)-set(S) )
        # S= list(set().union(S, T))
        S.extend(T)
        V = list( np.sort( list( set(V)-set(S) ) ) );

        #Filter
        gains = parallel_margvals_returnvals_thread(objective, S, V, 1)
        #Added query increment
        queries += len(V)
        V_ids = [V_all.index(elem) for elem in V]

        for ps in range( len(gains )):
            pastGains[ V_ids[ps] ] = gains[ ps ];
        
        V_above_thresh = np.where(gains >= tau)[0]
        V = [ V[idx] for idx in V_above_thresh ];
        if (len(V) == 0):
            break;
        if (len(S) == k):
            break;

    if (len(V) > 0 and len(S) < k and (itr==ell)): 
        print( "LAT has failed. This should be an extremely rare event. Terminating program..." );
        print( "V: ", V );
        # print( "S: ", S );
        exit(1);
            
    return [pastGains, S, queries];

def LAG_SingleNode(objective, k, eps, V, V_all, q, C, seed=42, stop_if_approx=False, nthreads=16, alpha=0, Gamma=0):

    '''
    The algorithm LAG using Single Node execution for Submodular Mazimization.
    PARALLEL IMPLEMENTATION (Multithread)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    bool stop_if_approx: determines whether we exit as soon as we find OPT that reaches approx guarantee or keep searching for better solution
    nthreads -- Number of threads to use on each machine
    
    OUTPUTS:
    list S -- the solution, where each element in the list is an element in the solution set
    ''' 
    
    # eps_FLS = 0.21
    if(len(C) > 0):
        V = list( set().union( C, V)); 
    if(k >= len(V)):
        return V
    # Gamma, sol, pastGains = ParallelLinearSeq_SingleNode_ParallelAlgo(objective, k, eps_FLS, V, seed, True, nthreads=16, return_all=True );
    pastGains = parallel_margvals_returnvals_thread(objective, [], V, nthreads)
    
    if (alpha==0 or Gamma==0):
        alpha = 1.0 / k
        valtop = np.max( pastGains);
        Gamma = valtop
    
    S = []
    #I1 = make_I(eps, k)
    
    tau = Gamma / (alpha * np.float(k));
    # taumin = Gamma / (3.0 * np.float(k));
    taumin = (eps * Gamma) / np.float(k);
    print( "LAG-- Gamma:", Gamma, "  alpha:", alpha,   "  tau:", tau, "  taumin:", taumin, "  |V|:", len(V), "  k:", k);
    # if (tau > valtop / np.float(k)):
    #     tau = valtop / np.float(k);
    
    #pastGains = np.inf*np.ones(len(V));
    while (tau > taumin):
        
        tau = np.min( [tau, np.max( pastGains ) * (1.0 - eps)] );
        if (tau <= 0.0):
            tau = taumin;
        V_above_thresh_ids = np.where(pastGains >= tau)[0]
        V_above_thresh = [ V[ele] for ele in V_above_thresh_ids ];
        V_above_thresh = list( set(V_above_thresh) - set(S) );

        print( "LAG: Before LAT-- |V|:", len(V_above_thresh), "  |S|:", len(S), "  tau:", tau );
        currGains = parallel_margvals_returnvals_thread(objective, S, V_above_thresh, 1)
        
        for ps in range( len(V_above_thresh )):
            pastGains[ V_above_thresh_ids[ps]] = currGains[ ps ];
        V_above_thresh_id = np.where(currGains >= tau)[0]
        V_above_thresh = [ V_above_thresh[ele] for ele in V_above_thresh_id ];
        if (len(V_above_thresh) > 0):
            [pastGains, S, queries_tmp] = LAT_SingleNode( V_above_thresh, S, V, V_all, q, objective, tau, eps / np.float(3), 1.0 / ((np.log( alpha / 3 ) / np.log( 1 - eps ) ) + 1) , k, pastGains );
            #Added query increment
            S_ids = [V.index(elem) for elem in S]   
            for ele in S_ids:
                pastGains[ele] = 0;
        print( "LAG: After LAT-- |V|:", len(V_above_thresh), "  |S|:", len(S), "  tau:", tau );
        if (len(S) >= k):
            break;
    print("LAG: After WHILE-- |V|:", len(V_above_thresh), "  |S|:", len(S), "  tau:", tau, "  k:", k )
    if(len(S)>k):
        Ap = S[len(S) - k : len(S)]
    else:
        Ap = S;
    
    return Ap

def Dist_LAG(objective, k, eps, V, V_all, q, comm, rank, size, C=[], p_root=0, seed=42, stop_if_apx=False, nthreads=16):

    check_inputs(objective, k)
    comm.barrier()
    V_split_local = V[rank] #np.array_split(V, size)[rank]
    
    # # Compute the marginal addition for each elem in N, then add the best one to solution L; remove it from remaining elements N
    # ele_A_local_vals = [ LAG_SingleNode(objective, k, eps, list(V_split_local.flatten()), C, seed, False, nthreads)]
    ele_A_local_vals = [ LAG_SingleNode(objective, k, eps, V_split_local, V_all, q, C, seed, False, nthreads)]
    ## return AlgRel()

    # # Gather the partial results to all processes
    ele_vals = comm.allgather(ele_A_local_vals)
    
    return [val for sublist in ele_vals for val in sublist]

# RDASH (Algorithm 1)
def RDASH(objective, k, eps, comm, rank, size, p_root=0, seed=42, stop_if_aprx=False, nthreads=16):
    '''
    The parallelizable distributed greedy algorithm RDASH. Uses multiple machines to obtain solution (Algorithm 1)
    PARALLEL IMPLEMENTATION (Multithread)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    bool stop_if_aprx: determines whether we exit as soon as we find OPT that reaches approx guarantee or keep searching for better solution
    nthreads -- Number of threads to use on each machine

    OUTPUTS:
    list S -- the solution, where each element in the list is an element in the solution set.
    float f(S) -- the value of the solution
    time -- runtime of the algorithm
    time_dist -- runtime spent on the distributed part of the algorithm
    time_post -- runtime spent on the post processing part of the algorithm
    ''' 

    comm.barrier()
    p_start = MPI.Wtime()
    
    # # Each processor gets the same random state 'q' so they can independently draw IDENTICAL random sequences .
    q = np.random.RandomState(42)

    random.seed(seed)
    S = []
    time_rounds = [0]
    query_rounds = [0]
    queries = 0
    V_all = [ ele for ele in objective.groundset ];
    # random.Random(seed).shuffle(V)
    V = [[] for i in range(size)]
    for ele in objective.groundset:
        x = random.randint(0, size-1)
        V[x].append(ele)
    p_start_dist = MPI.Wtime()
    S_DistGB_split = Dist_LAG(objective, k, eps, V, V_all, q, comm, rank, size, [], p_root, seed, False, nthreads)

    S_DistGB = []
    S_DistGB_all = []
    for i in range(len(S_DistGB_split)):
        S_DistGB.extend(list(S_DistGB_split[i]))
        S_tmp = S_DistGB_split[i]
        if(len(S_tmp)>k):
            S_DistGB_all.append(list(S_tmp[len(S_tmp) - k : len(S_tmp)]))
        else:
            S_DistGB_all.append(list(S_tmp))
    S_DistGB = list(np.unique(S_DistGB))
    
    
    p_stop_dist = MPI.Wtime()
    time_dist = (p_stop_dist - p_start_dist)

    p_start_post = MPI.Wtime()

    S_DistGreedy_split_vals = parallel_val_of_sets_thread(objective, S_DistGB_all, objective.nThreads)
    
    S_p = np.argmax(S_DistGreedy_split_vals)
    
    S_p_val = np.max(S_DistGreedy_split_vals)

    T = LAG_SingleNode(objective, k, eps, S_DistGB, V_all, q, [], seed, stop_if_aprx, nthreads=objective.nThreads)

    if(objective.value(T)>S_p_val):
        S = T
    else:
        S = list(S_DistGB_all[S_p])
    # print(S)
    p_stop = MPI.Wtime()
    time_post = (p_stop - p_start_post)
    time = (p_stop - p_start)
    valSol = objective.value( S )
    # Added increment
    queries += 1
    if rank == p_root:
        print ('DASH:', valSol, time, 'with k=', k, 'n=', len(objective.groundset), 'eps=', eps, '|S|=', len(S))
    
    return valSol, time, time_dist, time_post, S

# GDASH  (Algorithm 4)
def GDASH(objective, k, eps, comm, rank, size, p_root=0, seed=42, nthreads=16):
    

    '''
    The parallelizable distributed greedy algorithm GDASH. Uses multiple machines to obtain solution (Algorithm 4)
    PARALLEL IMPLEMENTATION (Multithread)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    nthreads -- Number of threads to use on each machine
    
    OUTPUTS:
    list S -- the solution, where each element in the list is an element in the solution set.
    float f(S) -- the value of the solution
    time -- runtime of the algorithm
    ''' 
    comm.barrier()

    p_start = MPI.Wtime()
    
    S = []
    C = []
    C_r = []
    # random.shuffle(V)
    
    check_inputs(objective, k)
    comm.barrier()
    V_all = [ ele for ele in objective.groundset ];
    
    # # Each processor gets the same random state 'q' so they can independently draw IDENTICAL random sequences .
    q = np.random.RandomState(42)
    
    random.seed(seed)

    for r in range(int(1/eps)): 
        V = [[] for i in range(size)]
        for ele in objective.groundset:
            x = random.randint(0, size-1)
            V[x].append(ele)

        S_DistGB_split = Dist_LAG(objective, k, eps, V, V_all, q, comm, rank, size, C_r, p_root, seed, False, nthreads)

        S_AlgSol = []
        S_AlgRel = []
        for i in range(len(S_DistGB_split)):
            S_AlgRel.extend(list(S_DistGB_split[i]))
            S_AlgSol.append(list(S_DistGB_split[i]))
            
        S_AlgRel = list(np.unique(S_AlgRel))
        
        S_AlgSol_split_vals = parallel_val_of_sets_thread(objective, S_AlgSol)
        
        S_p = np.argmax(S_AlgSol_split_vals)
        S_p_val = np.max(S_AlgSol_split_vals)

        if(S_p_val > objective.value(S)):
            S = list(S_AlgSol[S_p])
        
        ##added a variable to update C
        C_r = list( set().union( C_r, list(S_AlgRel)));
        
        # C = list( set().union( C, list(C_r)));
        
    p_stop = MPI.Wtime()
    time = (p_stop - p_start)
    valSol =  objective.value( S )
    print ('GDASH:', valSol, time, 'with k=', k, 'n=', objective.A.shape[1], 'eps=', eps, '|S|=', len(S))
        
    return valSol, time, S #time_dist, time_post, 



# DAT (Algorithm 3)
def DAT_binning(objective, Sol_all, taus_all, valtop, eps, k):
    # valSets = parallel_val_of_sets_thread(objective, S_DistLAT_all)
    taus = [(valtop*pow(1+eps,i)/k) for i in range(int(np.ceil((np.log(k)/np.log(1+eps))))-1)]
    valSets = [valx*k/valtop for valx in taus_all]
    valSets = [int(np.log(x)/np.log(1+eps)) if x >= 1 else -1 for x in valSets ]
    bins = [[] for i in range(int(np.ceil((np.log(k)/np.log(1+eps))))-1)]
    for i,x in enumerate(valSets):
        if(x>=0):
            bins[x].append(Sol_all[i])
    return bins, taus

def LAT_SingleThread(V, S, V_ground, q, objective, tau, eps, delta, k, pastGains, seed=42 ):
    
    '''
    The greedy algorithm LAT for Fixed Threshold 'tau' running on a single thread
    IMPLEMENTATION
    
    INPUTS:
    list V - contains the elements currently in the groundset
    list S - contains the elements currently in the solution set
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    float tau -- the fixed threshold
    float eps -- the error tolerance between 0 and 1
    float delta -- the probability tolerance between 0 and 1
    int k -- the cardinality constraint (must be k>0)
    randstate -- random seed to use when drawing samples 
    pastGains -- each element is the current marginal gain of the corresponding element in groundset

    OUTPUTS:
    list pastGains -- each element is the current marginal gain of the corresponding element in groundset
    list S -- the solution, where each element in the list is an element with marginal values > tau.

    '''  
    if(len(S)==k or len(V)==0):
        return S
    # randstate = np.random.RandomState(seed)
    ell = int(np.ceil( (4.0 + 8.0/ eps ) * np.log( len(V) / delta ) ) ) + 1;
    V = list( np.sort( list( set(V)-set(S) ) ) );
    for itr in range( ell + 1):
        s = np.min( [k - len(S), len(V) ] );
        
        q.shuffle(V_ground)
        tmp = [ele for ele in V_ground if ele in V]
        V = tmp

        # seq = sample_seq( V, s, randstate );
        seq = V[0:s]

        lmbda = make_lmbda( eps, s, s );
        idcs = [ele for ele in range(len(lmbda))]
        B = adaptiveAdd_thread(lmbda, idcs, seq, S, objective, eps, k, tau)
        #Added query increment
        lmbda_star = lmbda[0]
        if len(B) > 1:
            for i in range(1,len(B)):
                if(B[i]):
                    if (i == len(B) - 1):
                        lmbda_star = lmbda[-1];
                else:
                    lmbda_star = lmbda[i]
                    break;

        #Add elements
        #T= parallel_pessimistically_add_x_seq(objective, S, seq, tau, comm, rank , size );
        T = set(seq[0:lmbda_star]);
        
        ### PREVIOUS VERSION of LAT Ran the Following (Current Version stops at the first FALSE in Lambda; hence commented)
        # for i in range(lmbda_star, len(B)):
        #     if (B[i]):
        #         T = set().union(T, seq[ lmbda[ i - 1 ] : lmbda[ i ] ]);

        S= list(set().union(S, T))
        V = list( np.sort( list( set(V)-set(S))));
        
        gains = margvals_returnvals(objective, S, V)
                
        for ps in range( len(gains )):
            pastGains[ ps ] = gains[ ps ];
        
        V_above_thresh = np.where(gains >= tau)[0]
        V = [ V[idx] for idx in V_above_thresh ];
        if (len(V) == 0):
            break;
        if (len(S) == k):
            break;

    if (len(V) > 0 and len(S) < k):
        # if (rank == 0):
        print( "LAT has failed. This should be an extremely rare event. Terminating program..." );
        exit(1);
            
    return S;

def LAT_SingleThread_MultiThresh(V, S, V_ground, q, objective, taus, eps, delta, k, pastGains, seed=42 ):
    '''
    Helper function to run all different threshold guesses sequentially using a single thread
    '''
    A = []
    max_i = np.argmax(pastGains)
    # A.append([max_i])
    for j in range(len(taus)):
        sol_j = LAT_SingleThread(V, S, V_ground, q, objective, taus[j], eps, delta, k, pastGains, seed=42 )
        A.append(sol_j)
    return A

def LAT_SingleThread_MultiAll(V_all, S_all, V_ground, q, objective, taus, eps, delta, k, seed=42 ):
    '''
    Helper function to run all different threshold guesses sequentially using a single thread
    '''
    A = []
    for j in range(len(taus)):
        Gains = parallel_margvals_returnvals_thread(objective, S_all[j], V_all[j], nthreads=objective.nThreads)
        sol_j = LAT_SingleThread(V_all[j], S_all[j], V_ground, q, objective, taus[j], eps, delta, k, Gains, seed=42 )
        A.append(sol_j)
    return A

def Dist_LAT_SingleNode_GuessOPT(objective, k, eps, delta, V, V_all, q, taus, pastGains, seed=42, stop_if_apx=False, nthreads=16):
    '''
    Helper function to run all different threshold guesses in parallel on each machine
    '''
    #Dynamically obtaining the number of threads fr parallel computation
    # nthreads_rank = multiprocessing.cpu_count()
    nthreads_obj = objective.nThreads
    taus_split = np.array_split(taus, nthreads_obj)
    #with concurrent.futures.ProcessPoolExecutor(max_workers=nthreads) as executor:
    with concurrent.futures.ThreadPoolExecutor(max_workers=nthreads_obj) as executor:
        futures = [executor.submit(LAT_SingleThread_MultiThresh, V, [], V_all, q, objective, split, eps, delta, k, pastGains, seed) for split in taus_split ]
        return_value = [f.result() for f in futures]
    A = []
    for i in range(len(return_value)):
        A.extend(return_value[i])
    # return return_value
    return A

    # return return_value

def DAT_GuessOPT_SingleNode_postProc(objective, k, eps, delta, V_all, taus, S_all, V_ground, q, seed=42, stop_if_apx=False, nthreads=16):
    '''
    Helper function to run all different threshold guesses in parallel on each machine
    '''
    #Dynamically obtaining the number of threads fr parallel computation
    # nthreads_rank = multiprocessing.cpu_count()
    nthreads_obj = objective.nThreads
    taus_split = np.array_split(taus, nthreads_obj)
    V_all_split = np.array_split(V_all, nthreads_obj)
    S_all_split = np.array_split(S_all, nthreads_obj)
    #with concurrent.futures.ProcessPoolExecutor(max_workers=nthreads) as executor:
    with concurrent.futures.ThreadPoolExecutor(max_workers=nthreads_obj) as executor:
        futures = [executor.submit(LAT_SingleThread_MultiAll, V_all_split[i], S_all_split[i], V_ground, q, objective, taus_split[i], eps, delta, k, seed) for i in range(len(taus_split)) ]
        return_value = [f.result() for f in futures]
    A = []
    for i in range(len(return_value)):
        A.extend(return_value[i])
    # return return_value
    return A

    # return return_value

def Dist_LAT_GuessOPT(objective, k, eps, delta, V, V_all, q, comm, rank, size, p_root=0, seed=42, stop_if_apx=False, nthreads=16):

    check_inputs(objective, k)
    comm.barrier()
    V_split_local = V[rank] # np.array_split(V, size)[rank]
    
    # Gains = parallel_margvals_returnvals_thread(objective, [], list(V_split_local.flatten()), nthreads=objective.nThreads)
    Gains = parallel_margvals_returnvals_thread(objective, [], V_split_local, nthreads=objective.nThreads)
    
    res = [0]*3

    valtop = np.max( Gains);
    valtop_all = comm.allgather(valtop)

    taus = [(valtop*pow(1+eps,i)/k) for i in range(0, int(np.log(k)/np.log(1+eps)))]
    taus_all = comm.allgather(taus)

    # ele_A_local_vals = [Dist_LAT_SingleNode_GuessOPT(objective, k, eps, delta, list(V_split_local.flatten()), taus, Gains, seed=42, stop_if_apx=False, nthreads=objective.nThreads)]
    ele_A_local_vals = [Dist_LAT_SingleNode_GuessOPT(objective, k, eps, delta, V_split_local, V_all, q, taus, Gains, seed=42, stop_if_apx=False, nthreads=objective.nThreads)]
    
    # # Gather the partial results to all processes
    Sol_all = comm.allgather(ele_A_local_vals)
    
    res[0] = [val for val in valtop_all] # valtop_all
    res[1] = [val for val in taus_all] # for val in sublist] # taus_all
    res[2] = [val for sublist in Sol_all for val in sublist] # ele_vals

    return res

def DAT_GuessOPT(objective, k, eps, comm, rank, size, p_root=0, seed=42, stop_if_aprx=False, nthreads=16):
    '''
    The parallelizable distributed greedy algorithm DAT. Uses multiple machines to obtain solution (Algorithm 3)
    The algorithm proceeds with no knowledge of OPT

    PARALLEL IMPLEMENTATION (Multithread)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    float delta -- the probability tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    bool stop_if_aprx: determines whether we exit as soon as we find OPT that reaches approx guarantee or keep searching for better solution
    nthreads -- Number of threads to use on each machine

    OUTPUTS:
    list S -- the solution, where each element in the list is an element in the solution set.
    float f(S) -- the value of the solution
    time -- runtime of the algorithm
    time_dist -- runtime spent on the distributed part of the algorithm
    time_post -- runtime spent on the post processing part of the algorithm
    ''' 

    comm.barrier()
    p_start = MPI.Wtime()
    S = []
    
    # # Each processor gets the same random state 'q' so they can independently draw IDENTICAL random sequences .
    q = np.random.RandomState(42)
    
    random.seed(seed)
    V_all = [ ele for ele in objective.groundset ];
    # random.Random(seed).shuffle(V)
    V = [[] for i in range(size)]
    for ele in objective.groundset:
        x = random.randint(0, size-1)
        V[x].append(ele)
    
    delta = 1
    # Get the solution of parallel LAT
    p_start_dist = MPI.Wtime()
    
    S_DistLAT_split = Dist_LAT_GuessOPT(objective, k, eps, delta, V, V_all, q, comm, rank, size, p_root=0, seed=42, stop_if_apx=False, nthreads=objective.nThreads)
    valtop_all = S_DistLAT_split[0]
    # Max singleton
    valtop = np.max(valtop_all)
    
    tauDist = S_DistLAT_split[1]
    SolsDist = S_DistLAT_split[2]
    Sol_all = []
    taus_all = []
    for i in range(len(tauDist)):
        taus_all.extend(tauDist[i])
        Sol_all.extend(SolsDist[i])

    S_DistLAT = []
    S_DistLAT_all = []
    
    p_stop_dist = MPI.Wtime()
    time_dist = (p_stop_dist - p_start_dist)

    p_start_post = MPI.Wtime()

    S_Bin, taus = DAT_binning(objective, Sol_all, taus_all, valtop, eps, k)

    S_all = [list(set().union(*S_Bin[i])) for i in range(len(S_Bin))]

    A_all = []
    flag = 0
    for x in range(len(S_Bin)):
        A_x = []
        if(S_Bin[x]):
            A_x = random.Random(42).choice(S_Bin[x])
        if(len(A_x)==k):
            flag+=1
        A_all.append(A_x)
    
    if(flag!=len(A_all)):    
        T_all = DAT_GuessOPT_SingleNode_postProc(objective, k, eps, delta, S_all, taus, A_all, V_all, q)
    else:
        T_all = A_all
    T_x_all = parallel_val_of_sets_thread(objective, T_all)
    
    T_argmax = np.argmax(T_x_all)
    
    S = list(T_all[T_argmax])
    # print(S)
    p_stop = MPI.Wtime()
    time_post = (p_stop - p_start_post)
    time = (p_stop - p_start)
    valSol = objective.value( S )
    # Added increment
    if rank == p_root:
        print ('DAT_GuessOPT:', valSol, time, 'with k=', k, 'n=', len(objective.groundset), 'eps=', eps, '|S|=', len(S))
    
    return valSol, time, time_dist, time_post, S


'''
Journal Version Additional Algorithms
'''
#############################################################

# Utility functions for LDASH

def adaptiveAdd_thread_LAS(idcs, V, S, objective, eps, k):
    B = []
    if ( len( idcs ) > 0 ):
        
        S_union_T = list( set().union( V[0 : idcs[0]], S) ); # T_{i-1} = V[0 : idcs[i]]
        valS_union_T = objective.value( S_union_T );

        for v_i in idcs:
            S_union_T = list( set().union( [V[v_i]], S_union_T) );
            gain= objective.value( S_union_T ) - valS_union_T;
            thresh = valS_union_T / np.float(k);
            
            if (gain >= thresh):
                B.append(True)
            else:
                B.append(False)
            
            valS_union_T = valS_union_T + gain;
    
    # Gather the partial results to all processes
    
    return B

def parallel_adaptiveAdd_thread_LAS(V, S, objective, eps, k):
    
    '''
    Parallel-compute the marginal value of block  of elements and set the corresponding flag to True based on condition in LINE 13 of ALG. 1
    '''
    #Dynamically obtaining the number of threads fr parallel computation
    # nthreads_rank = multiprocessing.cpu_count()
    nthreads = objective.nThreads
    idcs = np.array_split( range( len( V ) ), nthreads )
    #lmbda_split_local = np.array_split(lmbda, size)[rank]
    #with concurrent.futures.ProcessPoolExecutor(max_workers=nthreads) as executor:
    with concurrent.futures.ThreadPoolExecutor(max_workers=nthreads) as executor:
        futures = [executor.submit(adaptiveAdd_thread_LAS, split, V, S, objective, eps, k) for split in idcs]
        return_value = [f.result() for f in futures]

    B = []
    for i in range(len(return_value)):
        B.extend(return_value[i])
    return B

def parallel_adaptiveAdd_LAS(V, S, objective, eps, k, comm, rank, size):
    
    '''
    Parallel-compute the marginal value of block  of elements and set the corresponding flag to True based on condition in LINE 13 of ALG. 1
    '''
    
    # Scatter the roughly equally sized subsets of remaining elements for which we want marg values
    comm.barrier()
    # idcs = np.array_split( range( len( lmbda ) ), size )[ rank ]
    V_split_local = np.array_split(V, size)[rank]

    B = parallel_adaptiveAdd_thread_LAS(list(V_split_local), V, S, objective, eps, k)

    # Gather the partial results to all processes
    B_vals = comm.allgather(B)

    return [val for sublist in B_vals for val in sublist]

def LAS_SingleNode(V_N, objective, k, eps, q, a, C, seed=42, nthreads=16):
    '''
    The parallelizable greedy algorithm LAS (Low-Adapive-Sequencing) using Single Node execution (OPTIMIZED IMPLEMENTATION)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k           -- the cardinality constraint (must be k>0)
    float eps       -- the error tolerance between 0 and 1
    int a           -- the max singleton assigned to every machine
    comm            -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank        -- the processor's rank (comm.Get_rank())
    int size        -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed            -- random seed to use when drawing samples
    bool stop_if_approx -- determines whether we exit as soon as we find OPT that reaches approx guarantee or keep searching for better solution

    OUTPUTS:
    float f(S)                  -- the value of the solution
    int queries                 -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time                  -- the processing time to optimize the function.
    list S                      -- the solution, where each element in the list is an element in the solution set.
    list of lists S_rounds      -- each element is a list containing the solution set S at the corresponding round.
    list of lists time_rounds   -- each element is a list containing the time at the corresponding round
    list of lists query_rounds  -- each element is a list containing the number of queries at the corresponding round
    list singletonVals          -- each element is the current marginal gain of the corresponding element in groundset
    '''  

    
    print( "\n --- Running LAS on N_i ---"); 

    if(len(C) > 0):
        V_N = list( set().union( C, V_N)); 
    if(k >= len(V_N)):
        return V_N

    n = len(V_N)    

    # Each processor gets its own random state so they can independently draw IDENTICAL random sequences a_seq.
    #proc_random_states = [np.random.RandomState(seed) for processor in range(size)]
    randstate = np.random.RandomState(seed)
    queries = 0
    #initialize S to max singleton
    S = [a]
    
    # q.shuffle(V_ground)
    currGains = parallel_margvals_returnvals_thread(objective, S, V_N, nthreads)
    
    queries += len(V_N)
    
    # S = [np.argmax(currGains)];
    singletonIdcs = np.argsort(currGains);
    singletonVals = currGains;
    currGains = np.sort(currGains);
    
    print( "\n --- Running first on the top 3K max singleton ---"); 

    #run first considering the top 3K singletons as the universe
    V = np.array(V_N)[ singletonIdcs[-3*k:] ]
    
    currGains = currGains[-3*k:];
    valtopk = np.sum( currGains[-k:] );


    while len(V) > 0:
        # Filtering
        t = objective.value(S)/np.float(k)
        queries += 1
        
        
        #do lazy discard first, based on last round's computations
        V_above_thresh = np.where(currGains >= t)[0]
        V = list( set([ V[idx] for idx in V_above_thresh ] ));

        #now recompute requisite gains
        print( "|V|:", len(V), "  |S|:", len(S), "  t:", t );
        print( "starting pmr..." );
        
        # currGains = parallel_margvals_returnvals(objective, S, [ele for ele in V], comm, rank, size)
        currGains = parallel_margvals_returnvals_thread(objective, S, V, nthreads)
        print("done.");
        queries += len(V)
        
        
        V_above_thresh = np.where(currGains >= t)[0]
        V = [ V[idx] for idx in V_above_thresh ];
        currGains = [ currGains[idx] for idx in V_above_thresh ];
        ###End Filtering
        
        print( "R1 After Filtering-- |V|:", len(V), "  |S|:", len(S), "  t:", t );

        if (len(V) == 0):
            break;
        
        # Random Permutation
        q.shuffle(V)
        
        print("R1 starting adaptiveAdd...");
        B = parallel_adaptiveAdd_thread_LAS(V, S, objective, eps, k)
        print("done.");
        
        queries += len( V );
        
        i_star = 0
        
        for i in range(1, len(B)+1):
            if(i <= k):
                if(np.sum(B[0:i]) >= (1-eps)*i):
                    i_star = i
            else:
                if((np.sum(B[i-k:i]) >= (1-eps)*k) and (np.sum(B[0:i-k]) >= (1-eps)*(i-k))):
                    i_star = i
        
        print( "R1 i_star: " , i_star);

        # S = list( set().union( V[0:i_star], S) ); # T_{i-star} = V[0 : i-star]
        T = list(set(V[0:i_star]) - set(S))
        S.extend(T)

    print( "--- Completed run on top 3K singletons --- ");   
    #run second on the entire universe

    t = objective.value(S) / np.float( k );
    queries += 1
    print( "--- Running on the entire universe with t=", t); 
    
    V_above_thresh = np.where(singletonVals >= t)[0]
    V = [ V_N[idx] for idx in V_above_thresh ]; 
    currGains = [ singletonVals[idx] for idx in V_above_thresh ];
    
    print( "R2 After pre-lazy discard |V| (>= t):", len(V)); 

    while len(V) > 0:
        # Filtering
        t = objective.value(S)/np.float(k)
        queries += 1
        
        #do lazy discard first, based on last round's computations
        V_above_thresh = np.where(currGains >= t)[0]
        V = list( set([ V[idx] for idx in V_above_thresh ] ));

        print( "R2 |V|:", len(V), "  |S|:", len(S), "  t:", t );
        #now recompute requisite gains
        print( "R2 starting pmr..." );
        currGains = parallel_margvals_returnvals_thread(objective, S, V, nthreads)
    
        #Added query increment
        queries += len(V)
        
        #currGains = parallel_margvals_returnvals(objective, S, [ele for ele in V], comm, rank, size)
        # if ( rank == 0):
        #     print( "done.");
        
        V_above_thresh = np.where(currGains >= t)[0]
        V = [ V[idx] for idx in V_above_thresh ];
        currGains = [ currGains[idx] for idx in V_above_thresh ];

        if (len(V) == 0):
            break;
        
        print( "R2 After Filtering-- |V|:", len(V), "  |S|:", len(S), "  t:", t );

        # Random Permutation
        q.shuffle(V)
        
        print("R2 starting adaptiveAdd...");
        B = parallel_adaptiveAdd_thread_LAS(V, S, objective, eps, k)
        print("R2 done.");
        
        queries += len( V );
        
        i_star = 0
        
        for i in range(1, len(B)+1):
            if(i <= k):
                if(np.sum(B[0:i]) >= (1-eps)*i):
                    i_star = i
            else:
                if((np.sum(B[i-k:i]) >= (1-eps)*k) and (np.sum(B[0:i-k]) >= (1-eps)*(i-k))):
                    i_star = i

        print( "R2 i_star: " , i_star);

        # S = list( set().union( V[0:i_star], S) ); # T_{i-star} = V[0 : i-star]
        T = list(set(V[0:i_star]) - set(S))
        S.extend(T)
    
    print( "--- Completed run (R2) on entire universe --- \n"); 

    if (len(V) > 0 and len(S) < k): 
        print( "LAS has failed. This should be an extremely rare event. Terminating program..." );
        print( "V: ", V );
        # print( "S: ", S );
        exit(1);
    
    return S

def Dist_LAS(objective, k, eps, V, q, a, comm, rank, size, C=[], p_root=0, seed=42, nthreads=16):

    check_inputs(objective, k)
    comm.barrier()
    V_split_local = V[rank] #np.array_split(V, size)[rank]
    
    # # Compute the marginal addition for each elem in N, then add the best one to solution L; remove it from remaining elements N
    # ele_A_local_vals = [ LAG_SingleNode(objective, k, eps, list(V_split_local.flatten()), C, seed, False, nthreads)]
    ele_A_local_vals = [ LAS_SingleNode(V_split_local, objective, k, eps, q, a, C, seed, nthreads)]
    ## return AlgRel()

    # # Gather the partial results to all processes
    ele_vals = comm.allgather(ele_A_local_vals)
    
    return [val for sublist in ele_vals for val in sublist]

# Linear-DASH (Algorithm LDASH (Algorithm LDASH with LAS in the distributed setting + LAS in post processing))

def LDASH(objective, k, eps, comm, rank, size, p_root=0, seed=42, nthreads=16):
    '''
    The parallelizable distributed linear-time greedy algorithm L-DASH. Uses multiple machines to obtain solution (Algorithm 1)
    PARALLEL IMPLEMENTATION (Multithread)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    bool stop_if_aprx: determines whether we exit as soon as we find OPT that reaches approx guarantee or keep searching for better solution
    nthreads -- Number of threads to use on each machine

    OUTPUTS:
    list S -- the solution, where each element in the list is an element in the solution set.
    float f(S) -- the value of the solution
    time -- runtime of the algorithm
    time_dist -- runtime spent on the distributed part of the algorithm
    time_post -- runtime spent on the post processing part of the algorithm
    ''' 

    comm.barrier()
    p_start = MPI.Wtime()
    
    # # Each processor gets the same random state 'q' so they can independently draw IDENTICAL random sequences .
    q = np.random.RandomState(42)

    random.seed(seed)

    S = []
    time_rounds = [0]
    query_rounds = [0]
    queries = 0
    V_all = [ ele for ele in objective.groundset ];
    
    # random.Random(seed).shuffle(V)
    V = [[] for i in range(size)]
    a, valA = 0, 0
    
    # Randomly assigning elements to all the machines 
    for ele in objective.groundset:
        # valE = objective.value([ele])
        
        # if valE >= valA:
        #     a = ele
        #     valA = valE
           
        x = random.randint(0, size-1)
        V[x].append(ele)
    
    # Obtaining the max singleton 'a'
    currGains = parallel_margvals_returnvals(objective, S, objective.groundset, comm, rank, size)
    # queries += len(objective.groundset)
    a = np.argmax(currGains);

    
    # Removing the max singleton from the groundset
    for i in range(size):
        V[i] = list(set(V[i]) - set([a]))
    
    p_start_dist = MPI.Wtime()
    S_DistGB_split = Dist_LAS(objective, k, eps, V, q, a, comm, rank, size, [], p_root, seed, nthreads)

    S_DistGB = []
    S_DistGB_all = []
    for i in range(len(S_DistGB_split)):
        S_DistGB.extend(list(S_DistGB_split[i]))
        S_tmp = S_DistGB_split[i]
        if(len(S_tmp)>k):
            S_DistGB_all.append(list(S_tmp[len(S_tmp) - k : len(S_tmp)]))
        else:
            S_DistGB_all.append(list(S_tmp))
    S_DistGB = list(np.unique(S_DistGB))
    
    
    p_stop_dist = MPI.Wtime()
    time_dist = (p_stop_dist - p_start_dist)

    p_start_post = MPI.Wtime()

    S_DistGreedy_split_vals = parallel_val_of_sets_thread(objective, S_DistGB_all, objective.nThreads)
    
    S_star = np.argmax(S_DistGreedy_split_vals)
    
    S_star_val = np.max(S_DistGreedy_split_vals)

    T = LAS_SingleNode(S_DistGB, objective, k, eps, q, a, [], seed, nthreads=objective.nThreads)

    if(len(T)>k):
        T_star = T[len(T) - k : len(T)]
    else:
        T_star = T

    if(objective.value(T_star)>S_star_val):
        S = T_star
    else:
        S = list(S_DistGB_all[S_star])
    
    print(S)

    p_stop = MPI.Wtime()
    time_post = (p_stop - p_start_post)
    time = (p_stop - p_start)
    valSol = objective.value( S )
    # Added increment
    queries += 1
    if rank == p_root:
        print ('LDASH:', valSol, time, 'with k=', k, 'n=', len(objective.groundset), 'eps=', eps, '|S|=', len(S))
    
    return valSol, time, time_dist, time_post, S

#############################################################

def LAS_SingleNode_localMax(V_N, objective, k, eps, q, C, seed=42, nthreads=16):
    '''
    The parallelizable greedy algorithm LAS (Low-Adapive-Sequencing) using Single Node execution (OPTIMIZED IMPLEMENTATION)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k           -- the cardinality constraint (must be k>0)
    float eps       -- the error tolerance between 0 and 1
    int a           -- the max singleton assigned to every machine
    comm            -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank        -- the processor's rank (comm.Get_rank())
    int size        -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed            -- random seed to use when drawing samples
    bool stop_if_approx -- determines whether we exit as soon as we find OPT that reaches approx guarantee or keep searching for better solution

    OUTPUTS:
    float f(S)                  -- the value of the solution
    int queries                 -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time                  -- the processing time to optimize the function.
    list S                      -- the solution, where each element in the list is an element in the solution set.
    list of lists S_rounds      -- each element is a list containing the solution set S at the corresponding round.
    list of lists time_rounds   -- each element is a list containing the time at the corresponding round
    list of lists query_rounds  -- each element is a list containing the number of queries at the corresponding round
    list singletonVals          -- each element is the current marginal gain of the corresponding element in groundset
    '''  

    
    print( "\n --- Running LAS on N_i ---"); 

    if(len(C) > 0):
        V_N = list( set().union( C, V_N)); 
    if(k >= len(V_N)):
        return V_N

    n = len(V_N)    

    # Each processor gets its own random state so they can independently draw IDENTICAL random sequences a_seq.
    #proc_random_states = [np.random.RandomState(seed) for processor in range(size)]
    randstate = np.random.RandomState(seed)
    queries = 0
    
    
    # q.shuffle(V_ground)
    currGains = parallel_margvals_returnvals_thread(objective, [], V_N, nthreads)
    
    #initialize S to max singleton
    # S = [a]
    S = [V_N[np.argmax(currGains)]];

    queries += len(V_N)
       
    singletonIdcs = np.argsort(currGains);
    singletonVals = currGains;
    currGains = np.sort(currGains);
    
    print( "\n --- Running first on the top 3K max singleton ---"); 

    #run first considering the top 3K singletons as the universe
    V = np.array(V_N)[ singletonIdcs[-3*k:] ]
    
    currGains = currGains[-3*k:];
    valtopk = np.sum( currGains[-k:] );


    while len(V) > 0:
        # Filtering
        t = objective.value(S)/np.float(k)
        queries += 1
        
        
        #do lazy discard first, based on last round's computations
        V_above_thresh = np.where(currGains >= t)[0]
        V = list( set([ V[idx] for idx in V_above_thresh ] ));

        #now recompute requisite gains
        print( "|V|:", len(V), "  |S|:", len(S), "  t:", t );
        print( "starting pmr..." );
        
        # currGains = parallel_margvals_returnvals(objective, S, [ele for ele in V], comm, rank, size)
        currGains = parallel_margvals_returnvals_thread(objective, S, V, nthreads)
        print("done.");
        queries += len(V)
        
        
        V_above_thresh = np.where(currGains >= t)[0]
        V = [ V[idx] for idx in V_above_thresh ];
        currGains = [ currGains[idx] for idx in V_above_thresh ];
        ###End Filtering
        
        print( "R1 After Filtering-- |V|:", len(V), "  |S|:", len(S), "  t:", t );

        if (len(V) == 0):
            break;
        
        # Random Permutation
        q.shuffle(V)
        
        print("R1 starting adaptiveAdd...");
        B = parallel_adaptiveAdd_thread_LAS(V, S, objective, eps, k)
        print("done.");
        
        queries += len( V );
        
        i_star = 0
        

        # ## PREVIOUS VERSION of LAS used the following
        # for i in range(1, len(B)+1):
        #     if(i <= k):
        #         if(np.sum(B[0:i]) >= (1-eps)*i):
        #             i_star = i
        #     else:
        #         if((np.sum(B[i-k:i]) >= (1-eps)*k) and (np.sum(B[0:i-k]) >= (1-eps)*(i-k))):
        #             i_star = i

        ### UPDATED VERSION of Conditions for LAS
        for i in range(1, len(B)+1):
            if(i <= k):
                if(np.sum(B[0:i]) < (1-eps)*i):
                    i_star = i-1
                    break
            elif(i <= len(B)):
                if((np.sum(B[i-k:i]) < (1-eps)*k) or (np.sum(B[0:i-k]) < (1-eps)*(i-k))):
                    i_star = i-1
                    break
            else:
                i_star = len( B )

        print( "R1 i_star: " , i_star);

        # S = list( set().union( V[0:i_star], S) ); # T_{i-star} = V[0 : i-star]
        T = list(set(V[0:i_star]) - set(S))
        S.extend(T)

    print( "--- Completed run on top 3K singletons --- ");   
    #run second on the entire universe

    t = objective.value(S) / np.float( k );
    queries += 1
    print( "--- Running on the entire universe with t=", t); 
    
    V_above_thresh = np.where(singletonVals >= t)[0]
    V = [ V_N[idx] for idx in V_above_thresh ]; 
    currGains = [ singletonVals[idx] for idx in V_above_thresh ];
    
    print( "R2 After pre-lazy discard |V| (>= t):", len(V)); 

    while len(V) > 0:
        # Filtering
        t = objective.value(S)/np.float(k)
        queries += 1
        
        #do lazy discard first, based on last round's computations
        V_above_thresh = np.where(currGains >= t)[0]
        V = list( set([ V[idx] for idx in V_above_thresh ] ));

        print( "R2 |V|:", len(V), "  |S|:", len(S), "  t:", t );
        #now recompute requisite gains
        print( "R2 starting pmr..." );
        currGains = parallel_margvals_returnvals_thread(objective, S, V, nthreads)
    
        #Added query increment
        queries += len(V)
        
        #currGains = parallel_margvals_returnvals(objective, S, [ele for ele in V], comm, rank, size)
        # if ( rank == 0):
        #     print( "done.");
        
        V_above_thresh = np.where(currGains >= t)[0]
        V = [ V[idx] for idx in V_above_thresh ];
        currGains = [ currGains[idx] for idx in V_above_thresh ];

        if (len(V) == 0):
            break;
        
        print( "R2 After Filtering-- |V|:", len(V), "  |S|:", len(S), "  t:", t );

        # Random Permutation
        q.shuffle(V)
        
        print("R2 starting adaptiveAdd...");
        B = parallel_adaptiveAdd_thread_LAS(V, S, objective, eps, k)
        print("R2 done.");
        
        queries += len( V );
        
        i_star = 0
        
        # ## PREVIOUS VERSION of LAS used the following
        # for i in range(1, len(B)+1):
        #     if(i <= k):
        #         if(np.sum(B[0:i]) >= (1-eps)*i):
        #             i_star = i
        #     else:
        #         if((np.sum(B[i-k:i]) >= (1-eps)*k) and (np.sum(B[0:i-k]) >= (1-eps)*(i-k))):
        #             i_star = i

        ### UPDATED VERSION of Conditions for LAS
        for i in range(1, len(B)+1):
            if(i <= k):
                if(np.sum(B[0:i]) < (1-eps)*i):
                    i_star = i-1
                    break
            elif(i <= len(B)):
                if((np.sum(B[i-k:i]) < (1-eps)*k) or (np.sum(B[0:i-k]) < (1-eps)*(i-k))):
                    i_star = i-1
                    break
            else:
                i_star = len( B )

        print( "R2 i_star: " , i_star);

        # S = list( set().union( V[0:i_star], S) ); # T_{i-star} = V[0 : i-star]
        T = list(set(V[0:i_star]) - set(S))
        S.extend(T)
    
    print( "--- Completed run (R2) on entire universe --- \n"); 

    if (len(V) > 0 and len(S) < k): 
        print( "LAS has failed. This should be an extremely rare event. Terminating program..." );
        print( "V: ", V );
        # print( "S: ", S );
        exit(1);
    
    return S

def Dist_LAS_localMax(objective, k, eps, V, q, comm, rank, size, C=[], p_root=0, seed=42, nthreads=16):

    check_inputs(objective, k)
    comm.barrier()
    V_split_local = V[rank] #np.array_split(V, size)[rank]
    
    # # Compute the marginal addition for each elem in N, then add the best one to solution L; remove it from remaining elements N
    # ele_A_local_vals = [ LAG_SingleNode(objective, k, eps, list(V_split_local.flatten()), C, seed, False, nthreads)]
    ele_A_local_vals = [ LAS_SingleNode_localMax(V_split_local, objective, k, eps, q, C, seed, nthreads)]
    ## return AlgRel()

    # # Gather the partial results to all processes
    ele_vals = comm.allgather(ele_A_local_vals)
    
    return [val for sublist in ele_vals for val in sublist]

# Linear-DASH_localMax (Algorithm LDASH with S=[max singleton in N_i] instead of global max singleton)

def LDASH_localMax(objective, k, eps, comm, rank, size, p_root=0, seed=42, nthreads=16):
    '''
    The parallelizable distributed linear-time greedy algorithm L-DASH. Uses multiple machines to obtain solution (Algorithm 1)
    PARALLEL IMPLEMENTATION (Multithread)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    bool stop_if_aprx: determines whether we exit as soon as we find OPT that reaches approx guarantee or keep searching for better solution
    nthreads -- Number of threads to use on each machine

    OUTPUTS:
    list S -- the solution, where each element in the list is an element in the solution set.
    float f(S) -- the value of the solution
    time -- runtime of the algorithm
    time_dist -- runtime spent on the distributed part of the algorithm
    time_post -- runtime spent on the post processing part of the algorithm
    ''' 

    comm.barrier()
    p_start = MPI.Wtime()
    
    # # Each processor gets the same random state 'q' so they can independently draw IDENTICAL random sequences .
    q = np.random.RandomState(42)

    random.seed(seed)

    S = []
    time_rounds = [0]
    query_rounds = [0]
    queries = 0
    V_all = [ ele for ele in objective.groundset ];
    
    # random.Random(seed).shuffle(V)
    V = [[] for i in range(size)]
    # a, valA = 0, 0
    
    # Randomly assigning elements to all the machines 
    for ele in objective.groundset:
        # valE = objective.value([ele])
        
        # if valE >= valA:
        #     a = ele
        #     valA = valE
           
        x = random.randint(0, size-1)
        V[x].append(ele)
    
    # # Obtaining the max singleton 'a'
    # currGains = parallel_margvals_returnvals(objective, S, objective.groundset, comm, rank, size)
    # # queries += len(objective.groundset)
    # a = np.argmax(currGains);

    
    # # Removing the max singleton from the groundset
    # for i in range(size):
    #     V[i] = list(set(V[i]) - set([a]))
    
    p_start_dist = MPI.Wtime()
    S_DistGB_split = Dist_LAS_localMax(objective, k, eps, V, q, comm, rank, size, [], p_root, seed, nthreads)

    S_DistGB = []
    S_DistGB_all = []
    for i in range(len(S_DistGB_split)):
        S_DistGB.extend(list(S_DistGB_split[i]))
        S_tmp = S_DistGB_split[i]
        if(len(S_tmp)>k):
            S_DistGB_all.append(list(S_tmp[len(S_tmp) - k : len(S_tmp)]))
        else:
            S_DistGB_all.append(list(S_tmp))
    S_DistGB = list(np.unique(S_DistGB))
    
    
    p_stop_dist = MPI.Wtime()
    time_dist = (p_stop_dist - p_start_dist)

    p_start_post = MPI.Wtime()

    S_DistGreedy_split_vals = parallel_val_of_sets_thread(objective, S_DistGB_all, objective.nThreads)
    
    S_star = S_DistGB_split[0] # np.argmax(S_DistGreedy_split_vals)
    
    S_star_val = objective.value(S_star) # np.max(S_DistGreedy_split_vals)

    T = LAS_SingleNode_localMax(S_DistGB, objective, k, eps, q, [], seed, nthreads=objective.nThreads)

    if(len(T)>k):
        T_star = T[len(T) - k : len(T)]
    else:
        T_star = T

    if(objective.value(T_star)>S_star_val):
        S = T_star
    else:
        S = list(S_DistGB_all[S_star])
    
    print(S)

    p_stop = MPI.Wtime()
    time_post = (p_stop - p_start_post)
    time = (p_stop - p_start)
    valSol = objective.value( S )
    # Added increment
    queries += 1
    if rank == p_root:
        print ('LDASH-localMax:', valSol, time, 'with k=', k, 'n=', len(objective.groundset), 'eps=', eps, '|S|=', len(S))
    
    return valSol, time, time_dist, time_post, S

#############################################################

# Linear-DASH_LAG (Algorithm LDASH with LAS in the distributed setting + LAG in post processing)

def LDASH_LAG(objective, k, eps, comm, rank, size, p_root=0, seed=42, nthreads=16):
    '''
    The parallelizable distributed linear-time greedy algorithm L-DASH. Uses multiple machines to obtain solution (Algorithm 1)
    PARALLEL IMPLEMENTATION (Multithread)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    bool stop_if_aprx: determines whether we exit as soon as we find OPT that reaches approx guarantee or keep searching for better solution
    nthreads -- Number of threads to use on each machine

    OUTPUTS:
    list S -- the solution, where each element in the list is an element in the solution set.
    float f(S) -- the value of the solution
    time -- runtime of the algorithm
    time_dist -- runtime spent on the distributed part of the algorithm
    time_post -- runtime spent on the post processing part of the algorithm
    ''' 

    comm.barrier()
    p_start = MPI.Wtime()
    
    # # Each processor gets the same random state 'q' so they can independently draw IDENTICAL random sequences .
    q = np.random.RandomState(42)

    random.seed(seed)

    S = []
    time_rounds = [0]
    query_rounds = [0]
    queries = 0
    V_all = [ ele for ele in objective.groundset ];
    
    # random.Random(seed).shuffle(V)
    V = [[] for i in range(size)]
    # a, valA = 0, 0
    
    # Randomly assigning elements to all the machines 
    for ele in objective.groundset:
        # valE = objective.value([ele])
        
        # if valE >= valA:
        #     a = ele
        #     valA = valE
           
        x = random.randint(0, size-1)
        V[x].append(ele)
    
    # # Obtaining the max singleton 'a'
    # currGains = parallel_margvals_returnvals(objective, S, objective.groundset, comm, rank, size)
    # # queries += len(objective.groundset)
    # a = np.argmax(currGains);

    
    # # Removing the max singleton from the groundset
    # for i in range(size):
    #     V[i] = list(set(V[i]) - set([a]))
    
    p_start_dist = MPI.Wtime()
    S_DistGB_split = Dist_LAS_localMax(objective, k, eps, V, q, comm, rank, size, [], p_root, seed, nthreads)

    S_DistGB = []
    S_DistGB_all = []
    for i in range(len(S_DistGB_split)):
        S_DistGB.extend(list(S_DistGB_split[i]))
        S_tmp = S_DistGB_split[i]
        if(len(S_tmp)>k):
            S_DistGB_all.append(list(S_tmp[len(S_tmp) - k : len(S_tmp)]))
        else:
            S_DistGB_all.append(list(S_tmp))
    S_DistGB = list(np.unique(S_DistGB))
    
    
    p_stop_dist = MPI.Wtime()
    time_dist = (p_stop_dist - p_start_dist)

    p_start_post = MPI.Wtime()

    S_DistGreedy_split_vals = parallel_val_of_sets_thread(objective, S_DistGB_all, objective.nThreads)
    
    S_star = S_DistGB_split[0] # np.argmax(S_DistGreedy_split_vals)
    
    S_star_val = objective.value(S_star) # np.max(S_DistGreedy_split_vals)

    ##################################

    T = LAG_SingleNode(objective, k, eps, S_DistGB, V_all, q, [], seed, stop_if_approx=False, nthreads=objective.nThreads)

    if(objective.value(T)>S_star_val):
        S = T
    else:
        S = list(S_DistGB_all[S_star])
    # print(S)
    p_stop = MPI.Wtime()
    time_post = (p_stop - p_start_post)
    time = (p_stop - p_start)
    valSol = objective.value( S )

    ##################################
    # T = LAS_SingleNode_localMax(S_DistGB, objective, k, eps, q, [], seed, nthreads=objective.nThreads)

    # if(len(T)>k):
    #     T_star = T[len(T) - k : len(T)]
    # else:
    #     T_star = T

    # if(objective.value(T_star)>S_star_val):
    #     S = T_star
    # else:
    #     S = list(S_DistGB_all[S_star])
    
    ##################################
    # print(S)

    # p_stop = MPI.Wtime()
    # time_post = (p_stop - p_start_post)
    # time = (p_stop - p_start)
    # valSol = objective.value( S )
    # Added increment
    queries += 1
    if rank == p_root:
        print ('LDASH-LAG:', valSol, time, 'with k=', k, 'n=', len(objective.groundset), 'eps=', eps, '|S|=', len(S))
    
    return valSol, time, time_dist, time_post, S

#############################################################

# Linear-DASH_LAS_LAG (Algorithm LDASH with LAS in the distributed setting + (LAS + LAG) in post processing)

def LDASH_LAS_LAG(objective, k, eps, comm, rank, size, p_root=0, seed=42, nthreads=16):
    '''
    The parallelizable distributed linear-time greedy algorithm L-DASH. Uses multiple machines to obtain solution (Algorithm 7)
    PARALLEL IMPLEMENTATION (Multithread)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    bool stop_if_aprx: determines whether we exit as soon as we find OPT that reaches approx guarantee or keep searching for better solution
    nthreads -- Number of threads to use on each machine

    OUTPUTS:
    list S -- the solution, where each element in the list is an element in the solution set.
    float f(S) -- the value of the solution
    time -- runtime of the algorithm
    time_dist -- runtime spent on the distributed part of the algorithm
    time_post -- runtime spent on the post processing part of the algorithm
    ''' 

    comm.barrier()
    p_start = MPI.Wtime()
    
    # # Each processor gets the same random state 'q' so they can independently draw IDENTICAL random sequences .
    q = np.random.RandomState(42)

    random.seed(seed)

    S = []
    time_rounds = [0]
    query_rounds = [0]
    queries = 0
    V_all = [ ele for ele in objective.groundset ];
    
    # random.Random(seed).shuffle(V)
    V = [[] for i in range(size)]
    # a, valA = 0, 0
    
    # Randomly assigning elements to all the machines 
    for ele in objective.groundset:
        # valE = objective.value([ele])
        
        # if valE >= valA:
        #     a = ele
        #     valA = valE
           
        x = random.randint(0, size-1)
        V[x].append(ele)
    
    # # Obtaining the max singleton 'a'
    # currGains = parallel_margvals_returnvals(objective, S, objective.groundset, comm, rank, size)
    # # queries += len(objective.groundset)
    # a = np.argmax(currGains);

    
    # # Removing the max singleton from the groundset
    # for i in range(size):
    #     V[i] = list(set(V[i]) - set([a]))
    
    p_start_dist = MPI.Wtime()
    S_DistGB_split = Dist_LAS_localMax(objective, k, eps, V, q, comm, rank, size, [], p_root, seed, nthreads)

    S_DistGB = []
    S_DistGB_all = []
    for i in range(len(S_DistGB_split)):
        S_DistGB.extend(list(S_DistGB_split[i]))
        S_tmp = S_DistGB_split[i]
        if(len(S_tmp)>k):
            S_DistGB_all.append(list(S_tmp[len(S_tmp) - k : len(S_tmp)]))
        else:
            S_DistGB_all.append(list(S_tmp))
    S_DistGB = list(np.unique(S_DistGB))
    
    
    p_stop_dist = MPI.Wtime()
    time_dist = (p_stop_dist - p_start_dist)

    p_start_post = MPI.Wtime()

    S_DistGreedy_split_vals = parallel_val_of_sets_thread(objective, S_DistGB_all, objective.nThreads)
    
    S_star = S_DistGB_split[0] # np.argmax(S_DistGreedy_split_vals)
    
    S_star_val = objective.value(S_star) # np.max(S_DistGreedy_split_vals)

    ##################################
    # LAS on entire returned set from all machines (S_DistGB)
    T_p = LAS_SingleNode_localMax(S_DistGB, objective, k, eps, q, [], seed, nthreads=objective.nThreads)
    if(len(T_p)>k):
        T_star = T_p[len(T_p) - k : len(T_p)]
    else:
        T_star = T_p
    
    ##################################
    # p_start_post = MPI.Wtime()
    # Compute alpha = 1/(4+(...)), gamma = f(T_star) and call LAG on the entire groundset from all machines (S_DistGB)
    alpha = 1/(4*(1+(eps*(2-eps)/((1-eps)*(1-(2*eps))))))
    Gamma = objective.value(T_star)
    T = LAG_SingleNode(objective, k, eps, S_DistGB, V_all, q, [], seed, stop_if_approx=False, nthreads=objective.nThreads, alpha=alpha, Gamma=Gamma)
    
    ##################################
    # S = argmax (S_star, T_star, T)
    if(objective.value(T) >= objective.value(T_star) and objective.value(T) >= S_star_val):
        S = T
    elif(objective.value(T_star) >= objective.value(T) and objective.value(T_star) >= S_star_val):
        S = T_star
    else:
        S = list(S_DistGB_all[S_star])
    
    # print(S)
    p_stop = MPI.Wtime()
    time_post = (p_stop - p_start_post)
    time = (p_stop - p_start)
    valSol = objective.value( S )

    ##################################
    # T = LAS_SingleNode_localMax(S_DistGB, objective, k, eps, q, [], seed, nthreads=objective.nThreads)

    # if(len(T)>k):
    #     T_star = T[len(T) - k : len(T)]
    # else:
    #     T_star = T

    # if(objective.value(T_star)>S_star_val):
    #     S = T_star
    # else:
    #     S = list(S_DistGB_all[S_star])
    
    ##################################
    # print(S)

    # p_stop = MPI.Wtime()
    # time_post = (p_stop - p_start_post)
    # time = (p_stop - p_start)
    # valSol = objective.value( S )
    # Added increment
    queries += 1
    if rank == p_root:
        print ('LDASH-LAS-LAG:', valSol, time, 'with k=', k, 'n=', len(objective.groundset), 'eps=', eps, '|S|=', len(S))
    
    return valSol, time, time_dist, time_post, S

#############################################################


# MED (Algorithm 4)
# Utility functions for MED runinng DASH (S_prev is passed to every routine for computation)
def MED_margvals_of_sets_returnvals(objective, sets, S):
    '''
    Compute the marginal value f(element)-f(L) of each element in set N. Version for stochasticgreedy_parallel.
    Returns the ordered list of marginal values corresponding to the list of remaining elements N
    '''
    set_vals_local_vals = [ objective.marginalval( myset, S ) for myset in sets ]
    
    return set_vals_local_vals

def MED_parallel_margvals_of_sets_threads(objective, list_of_sets, S, nthreads=16):
    ''' Parallel-compute the value f(S) of each set (sublist) in list_of_sets, return ordered list of corresponding values f(S) '''
    nthreads_rank = objective.nThreads
    list_split_local = np.array_split(list_of_sets, nthreads_rank)
    
    #with concurrent.futures.ProcessPoolExecutor(max_workers=nthreads) as executor:
    with concurrent.futures.ThreadPoolExecutor(max_workers=nthreads_rank) as executor:
        futures = [executor.submit(MED_margvals_of_sets_returnvals, objective, split, S) for split in list_split_local]
        return_value = [f.result() for f in futures]
    A = []
    for i in range(len(return_value)):
        A.extend(return_value[i])
    # return return_value
    return A
    # return return_value

def MED_adaptiveAdd_thread(lmbda, idcs, V, S, S_prev, objective, eps, k, tau):
    B = []
    if ( len( idcs ) > 0 ):
        pos=lmbda[ idcs[ 0 ] ];
        ppos=0
        if pos > 1:
            ppos=lmbda[ idcs[0] - 1 ];

        tmpS = list( set().union( V[0 : ppos], S) );
        valTmpS = objective.marginalval(tmpS, S_prev)
        Ti=list(set(V[ppos:pos]))

        for idx in range(1,len(idcs) + 1):
            tmpS = list( set(tmpS) | set( Ti) );
            
            gain= objective.marginalval(tmpS, S_prev) - valTmpS;
            if (tau == 0):
                thresh = len(Ti)*(1-eps)*valTmpS / np.float(k);
            else:
                thresh = len(Ti)*(1-eps)*tau;
                
            if (gain >= thresh):
                B.append(True)
            else:
                B.append(False)

            valTmpS = valTmpS + gain;
            if (idx >= len(idcs)):
                posEnd = lmbda[ idcs[ -1 ] ];
            else:
                posEnd = lmbda[ idcs[ idx ] ];
            Ti=V[ lmbda[ idcs[ idx - 1] ]: posEnd ];

    
    # Gather the partial results to all processes
    

    return B

def MED_parallel_adaptiveAdd_thread(lmbda, V, S, S_prev, objective, eps, k, tau = 0):
    
    '''
    Parallel-compute the marginal value of block  of elements and set the corresponding flag to True based on condition in LINE 13 of ALG. 1
    '''
    #Dynamically obtaining the number of threads fr parallel computation
    # nthreads_rank = multiprocessing.cpu_count()
    nthreads = objective.nThreads
    idcs = np.array_split( range( len( lmbda ) ), nthreads )
    #lmbda_split_local = np.array_split(lmbda, size)[rank]
    #with concurrent.futures.ProcessPoolExecutor(max_workers=nthreads) as executor:
    with concurrent.futures.ThreadPoolExecutor(max_workers=nthreads) as executor:
        futures = [executor.submit(MED_adaptiveAdd_thread, lmbda, split, V, S, S_prev, objective, eps, k, tau) for split in idcs]
        return_value = [f.result() for f in futures]

    B = []
    for i in range(len(return_value)):
        B.extend(return_value[i])
    return B

def MED_LAT_SingleNode(V, S, V_all, V_ground, q, objective, tau, eps, delta, k, pastGains, S_prev):
    
    '''
    The parallelizable greedy algorithm LAT using Single Node execution for Fixed Threshold 'tau'
    PARALLEL IMPLEMENTATION
    
    INPUTS:
    list V - contains the elements currently in the groundset
    list S - contains the elements currently in the solution set
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    float tau -- the fixed threshold
    float eps -- the error tolerance between 0 and 1
    float delta -- the probability tolerance between 0 and 1
    int k -- the cardinality constraint (must be k>0)
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())
    randstate -- random seed to use when drawing samples 
    pastGains -- each element is the current marginal gain of the corresponding element in groundset

    OUTPUTS:
    list pastGains -- each element is the current marginal gain of the corresponding element in groundset
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    list S -- the solution, where each element in the list is an element with marginal values > tau.

    '''  
    
    queries = 0    
    if(len(S)==k):
        return [pastGains, S, queries]
    ell = int(np.ceil( (4.0 + 8.0/ eps ) * np.log( len(V) / delta ) ) ) + 1;
    V = list( np.sort( list( set(V)-set(S).union(S_prev) ) ) );
    for itr in range( ell+1 ): # The additional iteration for ParallelAlgo_PGB
        s = np.min( [k - len(S), len(V) ] );

        q.shuffle(V_ground)
        tmp = [ele for ele in V_ground if ele in V]
        V = tmp

        # seq = sample_seq( V, s, randstate );
        seq = V[0:s]

        lmbda = make_lmbda( eps, s, s );
        
        B = MED_parallel_adaptiveAdd_thread(lmbda, seq, S, S_prev, objective, eps, k, tau)
        #Added query increment
        queries += len(lmbda)
        lmbda_star = lmbda[0]
        if len(B) > 1:
            for i in range(1,len(B)):
                if(B[i]):
                    if (i == len(B) - 1):
                        lmbda_star = lmbda[-1];
                else:
                    lmbda_star = lmbda[i]
                    break;

        T = set(seq[0:lmbda_star]);
        
        for i in range(lmbda_star, len(B)):
            if (B[i]):
                T = set().union(T, seq[ lmbda[ i - 1 ] : lmbda[ i ] ]);
        T = list( set(T)-set(S) )
        # S= list(set().union(S, T))
        S.extend(T)
        V = list( np.sort( list( set(V)-set(S) ) ) );

        #Filter
        gains = parallel_margvals_returnvals_thread(objective, list(set(S).union(S_prev)), V, 1)
        #Added query increment
        queries += len(V)
        V_ids = [V_all.index(elem) for elem in V]

        for ps in range( len(gains )):
            pastGains[ V_ids[ps] ] = gains[ ps ];
        
        V_above_thresh = np.where(gains >= tau)[0]
        V = [ V[idx] for idx in V_above_thresh ];
        if (len(V) == 0):
            break;
        if (len(S) == k):
            break;

    if (len(V) > 0 and len(S) < k and (itr==ell)): 
        # print( "LAT has failed. This should be an extremely rare event. Terminating program..." );
        # print( "V: ", V );
        # print( "S: ", S );
        exit(1);
            
    return [pastGains, S, queries];

def MED_LAG_SingleNode(objective, k, eps, V, V_all, q, C, S_prev, seed=42, stop_if_approx=False, nthreads=16):

    '''
    The algorithm LAG using Single Node execution for Submodular Mazimization.
    PARALLEL IMPLEMENTATION (Multithread)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    bool stop_if_approx: determines whether we exit as soon as we find OPT that reaches approx guarantee or keep searching for better solution
    nthreads -- Number of threads to use on each machine
    
    OUTPUTS:
    list S -- the solution, where each element in the list is an element in the solution set
    ''' 
    
    # eps_FLS = 0.21
    if(len(C) > 0):
        V = list( set().union( C, V)); 
    if(k >= len(V)):
        return V
    # Gamma, sol, pastGains = ParallelLinearSeq_SingleNode_ParallelAlgo(objective, k, eps_FLS, V, seed, True, nthreads=16, return_all=True );
    pastGains = parallel_margvals_returnvals_thread(objective, S_prev, V, nthreads=16)
    
    alpha = 1.0 / k
    valtop = np.max( pastGains);
    Gamma = valtop
    
    S = []
    #I1 = make_I(eps, k)
    
    tau = Gamma / (alpha * np.float(k));
    taumin = Gamma / (3.0 * np.float(k));
    
    # if (tau > valtop / np.float(k)):
    #     tau = valtop / np.float(k);
    
    #pastGains = np.inf*np.ones(len(V));
    while (tau > taumin):
        tau = np.min( [tau, np.max( pastGains ) * (1.0 - eps)] );
        if (tau <= 0.0):
            tau = taumin;
        V_above_thresh_ids = np.where(pastGains >= tau)[0]
        V_above_thresh = [ V[ele] for ele in V_above_thresh_ids ];
        V_above_thresh = list( set(V_above_thresh) - set(S).union(S_prev));
        currGains = parallel_margvals_returnvals_thread(objective, list( set(S).union(S_prev)), V_above_thresh, nthreads=16)
        
        for ps in range( len(V_above_thresh )):
            pastGains[ V_above_thresh_ids[ps]] = currGains[ ps ];
        V_above_thresh_id = np.where(currGains >= tau)[0]
        V_above_thresh = [ V_above_thresh[ele] for ele in V_above_thresh_id ];
        if (len(V_above_thresh) > 0):
            [pastGains, S, queries_tmp] = MED_LAT_SingleNode( V_above_thresh, S, V, V_all, q, objective, tau, eps / np.float(3), 1.0 / ((np.log( alpha / 3 ) / np.log( 1 - eps ) ) + 1) , k, pastGains, S_prev);
            #Added query increment
            S_ids = [V.index(elem) for elem in S]   
            for ele in S_ids:
                pastGains[ele] = 0;

        if (len(S) >= k):
            break;

    if(len(S)>k):
        Ap = S[len(S) - k : len(S)]
    else:
        Ap = S;
    
    return Ap

def MED_Dist_LAG(objective, k, eps, V, V_all, q, comm, rank, size, S_prev, C=[],  p_root=0, seed=42, stop_if_apx=False, nthreads=16):

    check_inputs(objective, k)
    comm.barrier()
    V_split_local = V[rank] #np.array_split(V, size)[rank]
    
    # # Compute the marginal addition for each elem in N, then add the best one to solution L; remove it from remaining elements N
    # ele_A_local_vals = [ LAG_SingleNode(objective, k, eps, list(V_split_local.flatten()), C, seed, False, nthreads)]
    ele_A_local_vals = [ MED_LAG_SingleNode(objective, k, eps, V_split_local, V_all, q, C, S_prev, seed, False, nthreads)]
    ## return AlgRel()

    # # Gather the partial results to all processes
    ele_vals = comm.allgather(ele_A_local_vals)
    
    return [val for sublist in ele_vals for val in sublist]

def MED_DASH(objective, k, S_prev, eps, comm, rank, size, p_root=0, seed=42, stop_if_aprx=False, nthreads=16):
    '''
    The parallelizable distributed greedy algorithm DASH. Uses multiple machines to obtain solution (Algorithm 1)
    PARALLEL IMPLEMENTATION (Multithread)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    list S_prev -- the solution S from the previous iteration
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    bool stop_if_aprx: determines whether we exit as soon as we find OPT that reaches approx guarantee or keep searching for better solution
    nthreads -- Number of threads to use on each machine

    OUTPUTS:
    list S -- the solution, where each element in the list is an element in the solution set.
    float f(S) -- the value of the solution
    time -- runtime of the algorithm
    time_dist -- runtime spent on the distributed part of the algorithm
    time_post -- runtime spent on the post processing part of the algorithm
    ''' 

    comm.barrier()
    p_start = MPI.Wtime()
    
    # # Each processor gets the same random state 'q' so they can independently draw IDENTICAL random sequences .
    q = np.random.RandomState(42)

    random.seed(seed)
    S = []
    time_rounds = [0]
    query_rounds = [0]
    queries = 0
    V_all = [ ele for ele in objective.groundset ];
    # random.Random(seed).shuffle(V)
    V = [[] for i in range(size)]
    for ele in objective.groundset:
        x = random.randint(0, size-1)
        V[x].append(ele)
    p_start_dist = MPI.Wtime()
    S_DistGB_split = MED_Dist_LAG(objective, k, eps, V, V_all, q, comm, rank, size, S_prev, [], p_root, seed, False, nthreads)

    S_DistGB = []
    S_DistGB_all = []
    for i in range(len(S_DistGB_split)):
        S_DistGB.extend(list(S_DistGB_split[i]))
        S_tmp = S_DistGB_split[i]
        if(len(S_tmp)>k):
            S_DistGB_all.append(list(S_tmp[len(S_tmp) - k : len(S_tmp)]))
        else:
            S_DistGB_all.append(list(S_tmp))
    S_DistGB = list(np.unique(S_DistGB))
    
    
    p_stop_dist = MPI.Wtime()
    time_dist = (p_stop_dist - p_start_dist)

    p_start_post = MPI.Wtime()

    S_DistGreedy_split_vals = MED_parallel_margvals_of_sets_threads(objective, S_DistGB_all, S_prev, nthreads=16)
    # S_DistGreedy_split_vals = parallel_val_of_sets_thread(objective, S_DistGB_all, objective.nThreads)
    
    S_p = np.argmax(S_DistGreedy_split_vals)
    
    S_p_val = np.max(S_DistGreedy_split_vals)

    T = MED_LAG_SingleNode(objective, k, eps, S_DistGB, V_all, q, [], S_prev, seed, stop_if_aprx, nthreads=objective.nThreads)
    
    # if(objective.value(T)>S_p_val):
    if(objective.marginalval( T, S_prev )>S_p_val):
        S = T
    else:
        S = list(S_DistGB_all[S_p])
    # print(S)
    p_stop = MPI.Wtime()
    time_post = (p_stop - p_start_post)
    time = (p_stop - p_start)
    # valSol = objective.value( S )
    valSol = objective.marginalval( S, S_prev )
    # Added increment
    queries += 1
    if rank == p_root:
        print ('MED_DASH:', valSol, time, 'with k=', k, 'n=', len(objective.groundset), 'eps=', eps, '|S|=', len(S))
    
    return S

def MEDDASH(objective, k, eps, comm, rank, size, p_root, seed=42, nthreads=16):
    '''
    The parallelizable distributed greedy algorithm MED running DASH over multiple rounds. Uses multiple machines to obtain solution (Algorithm 4)
    PARALLEL IMPLEMENTATION (Multithread)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    list S_prev -- the solution S from the previous iteration
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    bool stop_if_aprx: determines whether we exit as soon as we find OPT that reaches approx guarantee or keep searching for better solution
    nthreads -- Number of threads to use on each machine

    OUTPUTS:
    list S -- the solution, where each element in the list is an element in the solution set.
    float f(S) -- the value of the solution
    time -- runtime of the algorithm
    time_dist -- runtime spent on the distributed part of the algorithm
    time_post -- runtime spent on the post processing part of the algorithm
    '''

    comm.barrier()
    p_start = MPI.Wtime()
    
    S = []
    S_rounds = [[]]
    time_rounds = [0]
    query_rounds = [0]
    queries = 0
    V = [ ele for ele in objective.groundset ];
    np.random.shuffle(V)
    #k_max = np.max(k)iter
    beta = 1
    k_p = int(np.ceil(beta * len(objective.groundset)/ (size ** 2)))
    iters = int(np.ceil(k/k_p))
    k_p = min(k, k_p)
    if(rank==0):
        print("iters: ", iters, "  k_p:", k_p, "  k:",k)
    time_post = 0
    time_dist = 0
    for i in range(0,iters):
        p_start_dist = MPI.Wtime()
        if(k - len(S) < k_p): #Last iteration
            k_p = k - len(S)
        if(rank==0):
            print("(MEDDASH) iter: ", i, "  k_p:", k_p, "  k:",k,"  objective.value( S ):", objective.value( S ) ,"  len(S):", len(S),"  S:", S)
        T = MED_DASH(objective, k_p, S, eps, comm, rank, size, p_root, seed, False, nthreads)
        # valSolT, queriesT, timeT, T, T_DistGB_split, time_roundsT, query_roundsT = RandGreedI_PGB(objective, k_split, eps, comm, rank, size, p_root=0, seed=42, stop_if_apx=False, nthreads=16)
        comm.barrier()
        p_stop_dist = MPI.Wtime()
        time_dist += (p_stop_dist - p_start_dist)
        T_DistGB = list(np.unique(T))
        T_DistGB = list(set(T_DistGB) - set(S))
        comm.barrier()
        
        S.extend(T_DistGB)
        S = list(np.unique(S))
        S_rounds.append(S)
        if(rank==0):
            print("(MEDDASH) Iteration: ", i, "  Len(S):", len(S), "  Val(S): ", objective.value( S ), "  MargeVal(T): ", objective.marginalval(T_DistGB, S))
        if (len(S) >= k):
            # S = S[-k:]
            break;

    comm.barrier()
    # print("OUTSIDE FOR LOOP")
    # if len(S) > K, select last k elements added
    S = np.unique(S)
    if len(S) > k:
        S = S[-k:]
    p_stop = MPI.Wtime()
    time = (p_stop - p_start)
    valSol = objective.value( S )
    # Added increment
    queries += 1
    if rank == p_root:
        print ('MED+DASH:', valSol, queries, time, 'with k=', k, 'n=', len(objective.groundset), 'eps=', eps, '|S|=', len(S))

    return valSol, time, time_dist, time_post, S


'''
Journal Version Additional Algorithms
'''
#############################################################

# Utility functions for MED runinng LDASH (S_prev is passed to every routine for computation)
def MED_adaptiveAdd_thread_LAS(idcs, V, S, S_prev, objective, eps, k):
    B = []
    if ( len( idcs ) > 0 ):
        
        S_union_T = list( set().union( V[0 : idcs[0]], S) ); # T_{i-1} = V[0 : idcs[i]]
        valS_union_T = objective.marginalval( S_union_T, S_prev );

        for v_i in idcs:
            S_union_T = list( set().union( [V[v_i]], S_union_T) );
            gain = objective.marginalval( S_union_T, S_prev ) - valS_union_T;
            thresh = valS_union_T / np.float(k);
            
            if (gain >= thresh):
                B.append(True)
            else:
                B.append(False)
            
            valS_union_T = valS_union_T + gain;
    
    # Gather the partial results to all processes
    
    return B

def MED_parallel_adaptiveAdd_thread_LAS(V, S, S_prev, objective, eps, k):
    
    '''
    Parallel-compute the marginal value of block  of elements and set the corresponding flag to True based on condition in LINE 13 of ALG. 1
    '''
    #Dynamically obtaining the number of threads fr parallel computation
    # nthreads_rank = multiprocessing.cpu_count()
    nthreads = objective.nThreads
    idcs = np.array_split( range( len( V ) ), nthreads )
    #lmbda_split_local = np.array_split(lmbda, size)[rank]
    #with concurrent.futures.ProcessPoolExecutor(max_workers=nthreads) as executor:
    with concurrent.futures.ThreadPoolExecutor(max_workers=nthreads) as executor:
        futures = [executor.submit(MED_adaptiveAdd_thread_LAS, split, V, S, S_prev, objective, eps, k) for split in idcs]
        return_value = [f.result() for f in futures]

    B = []
    for i in range(len(return_value)):
        B.extend(return_value[i])
    return B

def MED_LAS_SingleNode(V_N, objective, k, eps, q, a, S_prev, C, seed=42, nthreads=16):
    '''
    The parallelizable greedy algorithm LAS (Low-Adapive-Sequencing) using Single Node execution (OPTIMIZED IMPLEMENTATION)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k           -- the cardinality constraint (must be k>0)
    float eps       -- the error tolerance between 0 and 1
    int a           -- the max singleton assigned to every machine
    comm            -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank        -- the processor's rank (comm.Get_rank())
    int size        -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed            -- random seed to use when drawing samples
    bool stop_if_approx -- determines whether we exit as soon as we find OPT that reaches approx guarantee or keep searching for better solution

    OUTPUTS:
    float f(S)                  -- the value of the solution
    int queries                 -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time                  -- the processing time to optimize the function.
    list S                      -- the solution, where each element in the list is an element in the solution set.
    list of lists S_rounds      -- each element is a list containing the solution set S at the corresponding round.
    list of lists time_rounds   -- each element is a list containing the time at the corresponding round
    list of lists query_rounds  -- each element is a list containing the number of queries at the corresponding round
    list singletonVals          -- each element is the current marginal gain of the corresponding element in groundset
    '''  

    if(len(C) > 0):
        V_N = list( set().union( C, V_N)); 
    if(k >= len(V_N)):
        return V_N

    n = len(V_N)    

    # Each processor gets its own random state so they can independently draw IDENTICAL random sequences a_seq.
    #proc_random_states = [np.random.RandomState(seed) for processor in range(size)]
    randstate = np.random.RandomState(seed)
    queries = 0
    #initialize S to max singleton
    S = [a]
    
    # q.shuffle(V_ground)
    currGains = parallel_margvals_returnvals_thread(objective, list( set(S).union(S_prev)), V_N, nthreads)
    
    queries += len(V_N)
        
    # S = [np.argmax(currGains)];
    singletonIdcs = np.argsort(currGains);
    singletonVals = currGains;
    currGains = np.sort(currGains);
    
    #run first considering the top 3K singletons as the universe
    V = np.array(V_N)[ singletonIdcs[-3*k:] ]
    
    currGains = currGains[-3*k:];
    valtopk = np.sum( currGains[-k:] );

    while len(V) > 0:
        # Filtering
        t = objective.marginalval(S, S_prev)/np.float(k)
        queries += 1
        
        
        #do lazy discard first, based on last round's computations
        V_above_thresh = np.where(currGains >= t)[0]
        V = list( set([ V[idx] for idx in V_above_thresh ] ));

        #now recompute requisite gains
        # print( len(V) );
        # print( "starting pmr..." );
        
        # currGains = parallel_margvals_returnvals(objective, S, [ele for ele in V], comm, rank, size)
        currGains = parallel_margvals_returnvals_thread(objective, list( set(S).union(S_prev)), V, nthreads)
    
        queries += len(V)
        
        
        V_above_thresh = np.where(currGains >= t)[0]
        V = [ V[idx] for idx in V_above_thresh ];
        currGains = [ currGains[idx] for idx in V_above_thresh ];
        ###End Filtering
        
        #     print( len(V) );

        if (len(V) == 0):
            break;
        
        # Random Permutation
        q.shuffle(V)
        
        # print("starting adaptiveAdd...");
        B = MED_parallel_adaptiveAdd_thread_LAS(V, S, S_prev, objective, eps, k)
        # print("done.");
        
        queries += len( V );
        
        i_star = 0
        
        for i in range(1, len(B)+1):
            if(i <= k):
                if(np.sum(B[0:i]) >= (1-eps)*i):
                    i_star = i
            else:
                if((np.sum(B[i-k:i]) >= (1-eps)*k) and (np.sum(B[0:i-k]) >= (1-eps)*(i-k))):
                    i_star = i
        
        # print( "i_Star: " , i_Star);

        # S = list( set().union( V[0:i_star], S) ); # T_{i-star} = V[0 : i-star]
        T = list(set(V[0:i_star]) - set(S))
        S.extend(T)
        
    #run second on the entire universe

    t = objective.marginalval(S, S_prev) / np.float( k );
    queries += 1
    
    V_above_thresh = np.where(singletonVals >= t)[0]
    V = [ V_N[idx] for idx in V_above_thresh ]; 
    currGains = [ singletonVals[idx] for idx in V_above_thresh ];
    
    while len(V) > 0:
        # Filtering
        t = objective.marginalval(S, S_prev)/np.float(k)
        queries += 1
        
        #do lazy discard first, based on last round's computations
        V_above_thresh = np.where(currGains >= t)[0]
        V = list( set([ V[idx] for idx in V_above_thresh ] ));

        #now recompute requisite gains
        # print( "starting pmr..." );
        currGains = parallel_margvals_returnvals_thread(objective, list( set(S).union(S_prev)), V, nthreads)
    
        #Added query increment
        queries += len(V)
        
        #currGains = parallel_margvals_returnvals(objective, S, [ele for ele in V], comm, rank, size)
        # if ( rank == 0):
        #     print( "done.");
        
        V_above_thresh = np.where(currGains >= t)[0]
        V = [ V[idx] for idx in V_above_thresh ];
        currGains = [ currGains[idx] for idx in V_above_thresh ];

        if (len(V) == 0):
            break;
        
        # if (rank == p_root):
        #     print( len(V) );

        # Random Permutation
        q.shuffle(V)
        
        # print("starting adaptiveAdd...");
        B = MED_parallel_adaptiveAdd_thread_LAS(V, S, S_prev, objective, eps, k)
        # print("done.");
        
        queries += len( V );
        
        i_star = 0
        
        for i in range(1, len(B)+1):
            if(i <= k):
                if(np.sum(B[0:i]) >= (1-eps)*i):
                    i_star = i
            else:
                if((np.sum(B[i-k:i]) >= (1-eps)*k) and (np.sum(B[0:i-k]) >= (1-eps)*(i-k))):
                    i_star = i

        # print( "i_Star: " , i_Star);

        # S = list( set().union( V[0:i_star], S) ); # T_{i-star} = V[0 : i-star]
        T = list(set(V[0:i_star]) - set(S))
        S.extend(T)
    
    if (len(V) > 0 and len(S) < k): 
        print( "LAS has failed. This should be an extremely rare event. Terminating program..." );
        print( "V: ", V );
        # print( "S: ", S );
        exit(1);
    
    return S

def MED_Dist_LAS(objective, k, eps, V, q, a, comm, rank, size, S_prev, C=[], p_root=0, seed=42, nthreads=16):

    check_inputs(objective, k)
    comm.barrier()
    V_split_local = V[rank] #np.array_split(V, size)[rank]
    
    # # Compute the marginal addition for each elem in N, then add the best one to solution L; remove it from remaining elements N
    # ele_A_local_vals = [ LAG_SingleNode(objective, k, eps, list(V_split_local.flatten()), C, seed, False, nthreads)]
    ele_A_local_vals = [ MED_LAS_SingleNode(V_split_local, objective, k, eps, q, a, S_prev, C, seed, nthreads)]
    ## return AlgRel()

    # # Gather the partial results to all processes
    ele_vals = comm.allgather(ele_A_local_vals)
    
    return [val for sublist in ele_vals for val in sublist]

def MED_LDASH(objective, k, eps, comm, rank, size, S_prev, p_root=0, seed=42, nthreads=16):
    '''
    The parallelizable distributed linear-time greedy algorithm L-DASH. Uses multiple machines to obtain solution (Algorithm 1)
    PARALLEL IMPLEMENTATION (Multithread)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())
    list S_prev -- the solution S from the previous iteration

    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    bool stop_if_aprx: determines whether we exit as soon as we find OPT that reaches approx guarantee or keep searching for better solution
    nthreads -- Number of threads to use on each machine

    OUTPUTS:
    list S -- the solution, where each element in the list is an element in the solution set.
    float f(S) -- the value of the solution
    time -- runtime of the algorithm
    time_dist -- runtime spent on the distributed part of the algorithm
    time_post -- runtime spent on the post processing part of the algorithm
    ''' 

    comm.barrier()
    p_start = MPI.Wtime()
    
    # # Each processor gets the same random state 'q' so they can independently draw IDENTICAL random sequences .
    q = np.random.RandomState(42)

    random.seed(seed)

    S = []
    time_rounds = [0]
    query_rounds = [0]
    queries = 0
    V_all = [ ele for ele in objective.groundset ];
    
    # random.Random(seed).shuffle(V)
    V = [[] for i in range(size)]
    a, valA = 0, 0
    
    # Randomly assigning elements to all the machines 
    for ele in objective.groundset:
        # valE = objective.value([ele])
        
        # if valE >= valA:
        #     a = ele
        #     valA = valE
           
        x = random.randint(0, size-1)
        V[x].append(ele)
    
    # Obtaining the max singleton
    currGains = parallel_margvals_returnvals(objective, S, objective.groundset, comm, rank, size)
    # queries += len(objective.groundset)
    a = np.argmax(currGains);

    # Removing the max singleton from the groundset
    for i in range(size):
        V[i] = list(set(V[i]) - set([a]))

    p_start_dist = MPI.Wtime()
    S_DistGB_split = MED_Dist_LAS(objective, k, eps, V, q, a, comm, rank, size, S_prev, [], p_root, seed, nthreads)

    S_DistGB = []
    S_DistGB_all = []
    for i in range(len(S_DistGB_split)):
        S_DistGB.extend(list(S_DistGB_split[i]))
        S_tmp = S_DistGB_split[i]
        if(len(S_tmp)>k):
            S_DistGB_all.append(list(S_tmp[len(S_tmp) - k : len(S_tmp)]))
        else:
            S_DistGB_all.append(list(S_tmp))
    S_DistGB = list(np.unique(S_DistGB))
    
    
    p_stop_dist = MPI.Wtime()
    time_dist = (p_stop_dist - p_start_dist)

    p_start_post = MPI.Wtime()

    S_DistGreedy_split_vals = MED_parallel_margvals_of_sets_threads(objective, S_DistGB_all, S_prev, objective.nThreads)
    # S_DistGreedy_split_vals = parallel_val_of_sets_thread(objective, S_DistGB_all, objective.nThreads)
    
    S_star = np.argmax(S_DistGreedy_split_vals)
    
    S_star_val = np.max(S_DistGreedy_split_vals)

    T = MED_LAS_SingleNode(S_DistGB, objective, k, eps, q, a, S_prev, [], seed, nthreads=objective.nThreads)

    if(len(T)>k):
        T_star = T[len(T) - k : len(T)]
    else:
        T_star = T

    if(objective.value(T_star)>S_star_val):
        S = T_star
    else:
        S = list(S_DistGB_all[S_star])
    
    print(S)

    p_stop = MPI.Wtime()
    time_post = (p_stop - p_start_post)
    time = (p_stop - p_start)
    valSol = objective.value( S )
    # Added increment
    queries += 1
    if rank == p_root:
        print ('MED_LDASH:', valSol, time, 'with k=', k, 'n=', len(objective.groundset), 'eps=', eps, '|S|=', len(S))
    
    return S

def MEDLDASH(objective, k, eps, comm, rank, size, p_root, seed=42, nthreads=16):
    '''
    The parallelizable distributed greedy algorithm MED running LDASH over multiple rounds. Uses multiple machines to obtain solution (Algorithm )
    PARALLEL IMPLEMENTATION (Multithread)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    list S_prev -- the solution S from the previous iteration
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    bool stop_if_aprx: determines whether we exit as soon as we find OPT that reaches approx guarantee or keep searching for better solution
    nthreads -- Number of threads to use on each machine

    OUTPUTS:
    list S -- the solution, where each element in the list is an element in the solution set.
    float f(S) -- the value of the solution
    time -- runtime of the algorithm
    time_dist -- runtime spent on the distributed part of the algorithm
    time_post -- runtime spent on the post processing part of the algorithm
    '''

    comm.barrier()
    p_start = MPI.Wtime()
    
    S = []
    S_rounds = [[]]
    time_rounds = [0]
    query_rounds = [0]
    queries = 0
    V = [ ele for ele in objective.groundset ];
    np.random.shuffle(V)
    #k_max = np.max(k)iter
    beta = 1
    k_p = int(np.ceil(beta * len(objective.groundset)/ (size ** 2)))
    iters = int(np.ceil(k/k_p))
    k_p = min(k, k_p)
    if(rank==0):
        print("iters: ", iters, "  k_p:", k_p, "  k:",k)
    time_post = 0
    time_dist = 0
    for i in range(0,iters):
        p_start_dist = MPI.Wtime()
        if(k - len(S) < k_p): #Last iteration
            k_p = k - len(S)
        T = MED_LDASH(objective, k_p, eps, comm, rank, size, S, p_root, seed, nthreads)
        # valSolT, queriesT, timeT, T, T_DistGB_split, time_roundsT, query_roundsT = RandGreedI_PGB(objective, k_split, eps, comm, rank, size, p_root=0, seed=42, stop_if_apx=False, nthreads=16)
        comm.barrier()
        p_stop_dist = MPI.Wtime()
        time_dist += (p_stop_dist - p_start_dist)
        # print("T:", T)
        T_DistGB = list(set(T))
        
        comm.barrier()
        T_DistGB = list(set(T_DistGB) - set(S))
        S.extend(T_DistGB)
        S = list(np.unique(S))
        S_rounds.append(S)
        # if(rank==0):
        #     print("Iteration: ", i, "  Len(S):", len(S), "  Val(S): ", objective.value( S ), "  MargeVal(T): ", objective.marginalval(T_DistGB, S))
        if (len(S) >= k):
            # S = S[-k:]
            break;

    comm.barrier()
    # print("OUTSIDE FOR LOOP")
    # if len(S) > K, select last k elements added
    S = np.unique(S)
    if len(S) > k:
        S = S[-k:]
    p_stop = MPI.Wtime()
    time = (p_stop - p_start)
    valSol = objective.value( S )
    # Added increment
    queries += 1
    if rank == p_root:
        print ('MED+LDASH:', valSol, queries, time, 'with k=', k, 'n=', len(objective.groundset), 'eps=', eps, '|S|=', len(S))

    return valSol, time, time_dist, time_post, S

#############################################################

def MED_LAS_SingleNode_localMax(V_N, objective, k, eps, q, S_prev, C, seed=42, nthreads=16):
    '''
    The parallelizable greedy algorithm LAS (Low-Adapive-Sequencing) using Single Node execution (OPTIMIZED IMPLEMENTATION)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k           -- the cardinality constraint (must be k>0)
    float eps       -- the error tolerance between 0 and 1
    int a           -- the max singleton assigned to every machine
    comm            -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank        -- the processor's rank (comm.Get_rank())
    int size        -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed            -- random seed to use when drawing samples
    bool stop_if_approx -- determines whether we exit as soon as we find OPT that reaches approx guarantee or keep searching for better solution

    OUTPUTS:
    float f(S)                  -- the value of the solution
    int queries                 -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time                  -- the processing time to optimize the function.
    list S                      -- the solution, where each element in the list is an element in the solution set.
    list of lists S_rounds      -- each element is a list containing the solution set S at the corresponding round.
    list of lists time_rounds   -- each element is a list containing the time at the corresponding round
    list of lists query_rounds  -- each element is a list containing the number of queries at the corresponding round
    list singletonVals          -- each element is the current marginal gain of the corresponding element in groundset
    '''  

    if(len(C) > 0):
        V_N = list( set().union( C, V_N)); 
    if(k >= len(V_N)):
        return V_N

    n = len(V_N)    

    # Each processor gets its own random state so they can independently draw IDENTICAL random sequences a_seq.
    #proc_random_states = [np.random.RandomState(seed) for processor in range(size)]
    randstate = np.random.RandomState(seed)
    queries = 0
    
    
    # q.shuffle(V_ground)
    currGains = parallel_margvals_returnvals_thread(objective, list( S_prev), V_N, nthreads)
    
    queries += len(V_N)

    #initialize S to max singleton
    # S = [a]    
    S = [V_N[np.argmax(currGains)]];

    singletonIdcs = np.argsort(currGains);
    singletonVals = currGains;
    currGains = np.sort(currGains);
    
    #run first considering the top 3K singletons as the universe
    V = np.array(V_N)[ singletonIdcs[-3*k:] ]
    
    currGains = currGains[-3*k:];
    valtopk = np.sum( currGains[-k:] );

    while len(V) > 0:
        # Filtering
        t = objective.marginalval(S, S_prev)/np.float(k)
        queries += 1
        
        
        #do lazy discard first, based on last round's computations
        V_above_thresh = np.where(currGains >= t)[0]
        V = list( set([ V[idx] for idx in V_above_thresh ] ));

        #now recompute requisite gains
        # print( len(V) );
        # print( "starting pmr..." );
        
        # currGains = parallel_margvals_returnvals(objective, S, [ele for ele in V], comm, rank, size)
        currGains = parallel_margvals_returnvals_thread(objective, list( set(S).union(S_prev)), V, nthreads)
    
        queries += len(V)
        
        
        V_above_thresh = np.where(currGains >= t)[0]
        V = [ V[idx] for idx in V_above_thresh ];
        currGains = [ currGains[idx] for idx in V_above_thresh ];
        ###End Filtering
        
        #     print( len(V) );

        if (len(V) == 0):
            break;
        
        # Random Permutation
        q.shuffle(V)
        
        # print("starting adaptiveAdd...");
        B = MED_parallel_adaptiveAdd_thread_LAS(V, S, S_prev, objective, eps, k)
        # print("done.");
        
        queries += len( V );
        
        i_star = 0
        
        for i in range(1, len(B)+1):
            if(i <= k):
                if(np.sum(B[0:i]) >= (1-eps)*i):
                    i_star = i
            else:
                if((np.sum(B[i-k:i]) >= (1-eps)*k) and (np.sum(B[0:i-k]) >= (1-eps)*(i-k))):
                    i_star = i
        
        # print( "i_Star: " , i_Star);

        # S = list( set().union( V[0:i_star], S) ); # T_{i-star} = V[0 : i-star]
        T = list(set(V[0:i_star]) - set(S))
        S.extend(T)
        
    #run second on the entire universe

    t = objective.marginalval(S, S_prev) / np.float( k );
    queries += 1
    
    V_above_thresh = np.where(singletonVals >= t)[0]
    V = [ V_N[idx] for idx in V_above_thresh ]; 
    currGains = [ singletonVals[idx] for idx in V_above_thresh ];
    
    while len(V) > 0:
        # Filtering
        t = objective.marginalval(S, S_prev)/np.float(k)
        queries += 1
        
        #do lazy discard first, based on last round's computations
        V_above_thresh = np.where(currGains >= t)[0]
        V = list( set([ V[idx] for idx in V_above_thresh ] ));

        #now recompute requisite gains
        # print( "starting pmr..." );
        currGains = parallel_margvals_returnvals_thread(objective, list( set(S).union(S_prev)), V, nthreads)
    
        #Added query increment
        queries += len(V)
        
        #currGains = parallel_margvals_returnvals(objective, S, [ele for ele in V], comm, rank, size)
        # if ( rank == 0):
        #     print( "done.");
        
        V_above_thresh = np.where(currGains >= t)[0]
        V = [ V[idx] for idx in V_above_thresh ];
        currGains = [ currGains[idx] for idx in V_above_thresh ];

        if (len(V) == 0):
            break;
        
        # if (rank == p_root):
        #     print( len(V) );

        # Random Permutation
        q.shuffle(V)
        
        # print("starting adaptiveAdd...");
        B = MED_parallel_adaptiveAdd_thread_LAS(V, S, S_prev, objective, eps, k)
        # print("done.");
        
        queries += len( V );
        
        i_star = 0
        
        for i in range(1, len(B)+1):
            if(i <= k):
                if(np.sum(B[0:i]) >= (1-eps)*i):
                    i_star = i
            else:
                if((np.sum(B[i-k:i]) >= (1-eps)*k) and (np.sum(B[0:i-k]) >= (1-eps)*(i-k))):
                    i_star = i

        # print( "i_Star: " , i_Star);

        # S = list( set().union( V[0:i_star], S) ); # T_{i-star} = V[0 : i-star]
        T = list(set(V[0:i_star]) - set(S))
        S.extend(T)
    
    if (len(V) > 0 and len(S) < k): 
        print( "LAS has failed. This should be an extremely rare event. Terminating program..." );
        print( "V: ", V );
        # print( "S: ", S );
        exit(1);
    
    return S

def MED_Dist_LAS_localMax(objective, k, eps, V, q, comm, rank, size, S_prev, C=[], p_root=0, seed=42, nthreads=16):

    check_inputs(objective, k)
    comm.barrier()
    V_split_local = V[rank] #np.array_split(V, size)[rank]
    
    # # Compute the marginal addition for each elem in N, then add the best one to solution L; remove it from remaining elements N
    # ele_A_local_vals = [ LAG_SingleNode(objective, k, eps, list(V_split_local.flatten()), C, seed, False, nthreads)]
    ele_A_local_vals = [ MED_LAS_SingleNode_localMax(V_split_local, objective, k, eps, q, S_prev, C, seed, nthreads)]
    ## return AlgRel()

    # # Gather the partial results to all processes
    ele_vals = comm.allgather(ele_A_local_vals)
    
    return [val for sublist in ele_vals for val in sublist]

def MED_LDASH_localMax(objective, k, eps, comm, rank, size, S_prev, p_root=0, seed=42, nthreads=16):
    '''
    The parallelizable distributed linear-time greedy algorithm L-DASH. Uses multiple machines to obtain solution (Algorithm 1)
    PARALLEL IMPLEMENTATION (Multithread)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())
    list S_prev -- the solution S from the previous iteration

    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    bool stop_if_aprx: determines whether we exit as soon as we find OPT that reaches approx guarantee or keep searching for better solution
    nthreads -- Number of threads to use on each machine

    OUTPUTS:
    list S -- the solution, where each element in the list is an element in the solution set.
    float f(S) -- the value of the solution
    time -- runtime of the algorithm
    time_dist -- runtime spent on the distributed part of the algorithm
    time_post -- runtime spent on the post processing part of the algorithm
    ''' 

    comm.barrier()
    p_start = MPI.Wtime()
    
    # # Each processor gets the same random state 'q' so they can independently draw IDENTICAL random sequences .
    q = np.random.RandomState(42)

    random.seed(seed)

    S = []
    time_rounds = [0]
    query_rounds = [0]
    queries = 0
    V_all = [ ele for ele in objective.groundset ];
    
    # random.Random(seed).shuffle(V)
    V = [[] for i in range(size)]
    a, valA = 0, 0
    
    # Randomly assigning elements to all the machines 
    for ele in objective.groundset:
        # valE = objective.value([ele])
        
        # if valE >= valA:
        #     a = ele
        #     valA = valE
           
        x = random.randint(0, size-1)
        V[x].append(ele)
    
    # # Obtaining the max singleton
    # currGains = parallel_margvals_returnvals(objective, S, objective.groundset, comm, rank, size)
    # # queries += len(objective.groundset)
    # a = np.argmax(currGains);

    # # Removing the max singleton from the groundset
    # for i in range(size):
    #     V[i] = list(set(V[i]) - set([a]))

    p_start_dist = MPI.Wtime()
    S_DistGB_split = MED_Dist_LAS_localMax(objective, k, eps, V, q, comm, rank, size, S_prev, [], p_root, seed, nthreads)

    S_DistGB = []
    S_DistGB_all = []
    for i in range(len(S_DistGB_split)):
        S_DistGB.extend(list(S_DistGB_split[i]))
        S_tmp = S_DistGB_split[i]
        if(len(S_tmp)>k):
            S_DistGB_all.append(list(S_tmp[len(S_tmp) - k : len(S_tmp)]))
        else:
            S_DistGB_all.append(list(S_tmp))
    S_DistGB = list(np.unique(S_DistGB))
    
    
    p_stop_dist = MPI.Wtime()
    time_dist = (p_stop_dist - p_start_dist)

    p_start_post = MPI.Wtime()

    S_DistGreedy_split_vals = MED_parallel_margvals_of_sets_threads(objective, S_DistGB_all, S_prev, objective.nThreads)
    # S_DistGreedy_split_vals = parallel_val_of_sets_thread(objective, S_DistGB_all, objective.nThreads)
    
    S_star = np.argmax(S_DistGreedy_split_vals)
    
    S_star_val = np.max(S_DistGreedy_split_vals)

    T = MED_LAS_SingleNode_localMax(S_DistGB, objective, k, eps, q, S_prev, [], seed, nthreads=objective.nThreads)

    if(len(T)>k):
        T_star = T[len(T) - k : len(T)]
    else:
        T_star = T

    if(objective.value(T_star)>S_star_val):
        S = T_star
    else:
        S = list(S_DistGB_all[S_star])
    
    print(S)

    p_stop = MPI.Wtime()
    time_post = (p_stop - p_start_post)
    time = (p_stop - p_start)
    valSol = objective.value( S )
    # Added increment
    queries += 1
    if rank == p_root:
        print ('MED_LDASH:', valSol, time, 'with k=', k, 'n=', len(objective.groundset), 'eps=', eps, '|S|=', len(S))
    
    return S

def MEDLDASH_localMax(objective, k, eps, comm, rank, size, p_root, seed=42, nthreads=16):
    '''
    The parallelizable distributed greedy algorithm MED running LDASH over multiple rounds. Uses multiple machines to obtain solution (Algorithm )
    PARALLEL IMPLEMENTATION (Multithread)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    list S_prev -- the solution S from the previous iteration
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    bool stop_if_aprx: determines whether we exit as soon as we find OPT that reaches approx guarantee or keep searching for better solution
    nthreads -- Number of threads to use on each machine

    OUTPUTS:
    list S -- the solution, where each element in the list is an element in the solution set.
    float f(S) -- the value of the solution
    time -- runtime of the algorithm
    time_dist -- runtime spent on the distributed part of the algorithm
    time_post -- runtime spent on the post processing part of the algorithm
    '''

    comm.barrier()
    p_start = MPI.Wtime()
    
    S = []
    S_rounds = [[]]
    time_rounds = [0]
    query_rounds = [0]
    queries = 0
    V = [ ele for ele in objective.groundset ];
    np.random.shuffle(V)
    #k_max = np.max(k)iter
    beta = 1
    k_p = int(np.ceil(beta * len(objective.groundset)/ (size ** 2)))
    iters = int(np.ceil(k/k_p))
    k_p = min(k, k_p)
    if(rank==0):
        print("iters: ", iters, "  k_p:", k_p, "  k:",k)
    time_post = 0
    time_dist = 0
    for i in range(0,iters):
        p_start_dist = MPI.Wtime()
        if(k - len(S) < k_p): #Last iteration
            k_p = k - len(S)
        T = MED_LDASH_localMax(objective, k_p, eps, comm, rank, size, S, p_root, seed, nthreads)
        # valSolT, queriesT, timeT, T, T_DistGB_split, time_roundsT, query_roundsT = RandGreedI_PGB(objective, k_split, eps, comm, rank, size, p_root=0, seed=42, stop_if_apx=False, nthreads=16)
        comm.barrier()
        p_stop_dist = MPI.Wtime()
        time_dist += (p_stop_dist - p_start_dist)
        # print("T:", T)
        T_DistGB = list(set(T))
        
        comm.barrier()
        T_DistGB = list(set(T_DistGB) - set(S))
        S.extend(T_DistGB)
        S = list(np.unique(S))
        S_rounds.append(S)
        # if(rank==0):
        #     print("Iteration: ", i, "  Len(S):", len(S), "  Val(S): ", objective.value( S ), "  MargeVal(T): ", objective.marginalval(T_DistGB, S))
        if (len(S) >= k):
            # S = S[-k:]
            break;

    comm.barrier()
    # print("OUTSIDE FOR LOOP")
    # if len(S) > K, select last k elements added
    S = np.unique(S)
    if len(S) > k:
        S = S[-k:]
    p_stop = MPI.Wtime()
    time = (p_stop - p_start)
    valSol = objective.value( S )
    # Added increment
    queries += 1
    if rank == p_root:
        print ('MED+LDASH-localMax:', valSol, queries, time, 'with k=', k, 'n=', len(objective.groundset), 'eps=', eps, '|S|=', len(S))

    return valSol, time, time_dist, time_post, S

#############################################################

def MED_LDASH_LAG(objective, k, eps, comm, rank, size, S_prev, p_root=0, seed=42, nthreads=16):
    '''
    The parallelizable distributed linear-time greedy algorithm L-DASH. Uses multiple machines to obtain solution (Algorithm 1)
    PARALLEL IMPLEMENTATION (Multithread)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())
    list S_prev -- the solution S from the previous iteration

    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    bool stop_if_aprx: determines whether we exit as soon as we find OPT that reaches approx guarantee or keep searching for better solution
    nthreads -- Number of threads to use on each machine

    OUTPUTS:
    list S -- the solution, where each element in the list is an element in the solution set.
    float f(S) -- the value of the solution
    time -- runtime of the algorithm
    time_dist -- runtime spent on the distributed part of the algorithm
    time_post -- runtime spent on the post processing part of the algorithm
    ''' 

    comm.barrier()
    p_start = MPI.Wtime()
    
    # # Each processor gets the same random state 'q' so they can independently draw IDENTICAL random sequences .
    q = np.random.RandomState(42)

    random.seed(seed)

    S = []
    time_rounds = [0]
    query_rounds = [0]
    queries = 0
    V_all = [ ele for ele in objective.groundset ];
    
    # random.Random(seed).shuffle(V)
    V = [[] for i in range(size)]
    a, valA = 0, 0
    
    # Randomly assigning elements to all the machines 
    for ele in objective.groundset:
        # valE = objective.value([ele])
        
        # if valE >= valA:
        #     a = ele
        #     valA = valE
           
        x = random.randint(0, size-1)
        V[x].append(ele)
    
    # # Obtaining the max singleton
    # currGains = parallel_margvals_returnvals(objective, S, objective.groundset, comm, rank, size)
    # # queries += len(objective.groundset)
    # a = np.argmax(currGains);

    # # Removing the max singleton from the groundset
    # for i in range(size):
    #     V[i] = list(set(V[i]) - set([a]))

    p_start_dist = MPI.Wtime()
    S_DistGB_split = MED_Dist_LAS_localMax(objective, k, eps, V, q, comm, rank, size, S_prev, [], p_root, seed, nthreads)

    S_DistGB = []
    S_DistGB_all = []
    for i in range(len(S_DistGB_split)):
        S_DistGB.extend(list(S_DistGB_split[i]))
        S_tmp = S_DistGB_split[i]
        if(len(S_tmp)>k):
            S_DistGB_all.append(list(S_tmp[len(S_tmp) - k : len(S_tmp)]))
        else:
            S_DistGB_all.append(list(S_tmp))
    S_DistGB = list(np.unique(S_DistGB))
    
    
    p_stop_dist = MPI.Wtime()
    time_dist = (p_stop_dist - p_start_dist)

    p_start_post = MPI.Wtime()

    S_DistGreedy_split_vals = MED_parallel_margvals_of_sets_threads(objective, S_DistGB_all, S_prev, objective.nThreads)
    # S_DistGreedy_split_vals = parallel_val_of_sets_thread(objective, S_DistGB_all, objective.nThreads)
    
    S_star = np.argmax(S_DistGreedy_split_vals)
    
    S_star_val = np.max(S_DistGreedy_split_vals)

    ##############################################################

    T = MED_LAG_SingleNode(objective, k, eps, S_DistGB, V_all, q, [], S_prev, seed, stop_if_approx=False, nthreads=objective.nThreads)
    
    # if(objective.value(T)>S_p_val):
    if(objective.marginalval( T, S_prev )>S_star_val):
        S = T
    else:
        S = list(S_DistGB_all[S_star])
    # print(S)
    p_stop = MPI.Wtime()
    time_post = (p_stop - p_start_post)
    time = (p_stop - p_start)
    # valSol = objective.value( S )
    valSol = objective.marginalval( S, S_prev )

    ################################################################

    # T = MED_LAS_SingleNode_localMax(S_DistGB, objective, k, eps, q, S_prev, [], seed, nthreads=objective.nThreads)

    # if(len(T)>k):
    #     T_star = T[len(T) - k : len(T)]
    # else:
    #     T_star = T

    # if(objective.value(T_star)>S_star_val):
    #     S = T_star
    # else:
    #     S = list(S_DistGB_all[S_star])
    ######################################################################
    # print(S)

    p_stop = MPI.Wtime()
    time_post = (p_stop - p_start_post)
    time = (p_stop - p_start)
    valSol = objective.value( S )
    # Added increment
    queries += 1
    if rank == p_root:
        print ('MED_LDASHLAG:', valSol, time, 'with k=', k, 'n=', len(objective.groundset), 'eps=', eps, '|S|=', len(S))
    
    return S

def MEDLDASH_LAG(objective, k, eps, comm, rank, size, p_root, seed=42, nthreads=16):
    '''
    The parallelizable distributed greedy algorithm MED running LDASH over multiple rounds. Uses multiple machines to obtain solution (Algorithm )
    PARALLEL IMPLEMENTATION (Multithread)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    list S_prev -- the solution S from the previous iteration
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    bool stop_if_aprx: determines whether we exit as soon as we find OPT that reaches approx guarantee or keep searching for better solution
    nthreads -- Number of threads to use on each machine

    OUTPUTS:
    list S -- the solution, where each element in the list is an element in the solution set.
    float f(S) -- the value of the solution
    time -- runtime of the algorithm
    time_dist -- runtime spent on the distributed part of the algorithm
    time_post -- runtime spent on the post processing part of the algorithm
    '''

    comm.barrier()
    p_start = MPI.Wtime()
    
    S = []
    S_rounds = [[]]
    time_rounds = [0]
    query_rounds = [0]
    queries = 0
    V = [ ele for ele in objective.groundset ];
    np.random.shuffle(V)
    #k_max = np.max(k)iter
    beta = 1
    k_p = int(np.ceil(beta * len(objective.groundset)/ (size ** 2)))
    iters = int(np.ceil(k/k_p))
    k_p = min(k, k_p)
    if(rank==0):
        print("iters: ", iters, "  k_p:", k_p, "  k:",k)
    time_post = 0
    time_dist = 0
    for i in range(0,iters):
        p_start_dist = MPI.Wtime()
        if(k - len(S) < k_p): #Last iteration
            k_p = k - len(S)
        T = MED_LDASH_LAG(objective, k_p, eps, comm, rank, size, S, p_root, seed, nthreads)
        # valSolT, queriesT, timeT, T, T_DistGB_split, time_roundsT, query_roundsT = RandGreedI_PGB(objective, k_split, eps, comm, rank, size, p_root=0, seed=42, stop_if_apx=False, nthreads=16)
        comm.barrier()
        p_stop_dist = MPI.Wtime()
        time_dist += (p_stop_dist - p_start_dist)
        # print("T:", T)
        T_DistGB = list(set(T))
        
        comm.barrier()
        T_DistGB = list(set(T_DistGB) - set(S))
        S.extend(T_DistGB)
        S = list(np.unique(S))
        S_rounds.append(S)
        # if(rank==0):
        #     print("Iteration: ", i, "  Len(S):", len(S), "  Val(S): ", objective.value( S ), "  MargeVal(T): ", objective.marginalval(T_DistGB, S))
        if (len(S) >= k):
            # S = S[-k:]
            break;

    comm.barrier()
    # print("OUTSIDE FOR LOOP")
    # if len(S) > K, select last k elements added
    S = np.unique(S)
    if len(S) > k:
        S = S[-k:]
    p_stop = MPI.Wtime()
    time = (p_stop - p_start)
    valSol = objective.value( S )
    # Added increment
    queries += 1
    if rank == p_root:
        print ('MED+LDASH-LAG:', valSol, queries, time, 'with k=', k, 'n=', len(objective.groundset), 'eps=', eps, '|S|=', len(S))

    return valSol, time, time_dist, time_post, S

#############################################################

# Utility functions for MED runinng RandGreedI (S_prev is passed to every routine for computation)

def MED_lazygreedy_MultiNode(objective, k, S_prev, N, comm, rank, size, nthreads=16):
    ''' 
    Accelerated lazy greedy algorithm: for k steps (Minoux 1978).
    **NOTE** solution sets and values may be different than those found by our Greedy implementation, 
    as the two implementations break ties differently (i.e. when two elements 
    have the same marginal value,  the two implementations may not pick the same element to add)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint
    
    OUTPUTS:
    float f(L) -- the value of the solution
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time -- the processing time to optimize the function.
    list L -- the solution, where each element in the list is an element in the solution set.
    '''
    check_inputs(objective, k)
    queries = 0
    time0 = datetime.now()
    L = []
    # Cov = []
    L_rounds = [[]] # to hold L's evolution over rounds. Sometimes the new round does not add an element. 
    time_rounds = [0]
    query_rounds = [0]
    # N = [ele for ele in objective.groundset]
    if(k>=len(N)):
        return N
    # On the first iteration, LazyGreedy behaves exactly like regular Greedy:
    # ele_vals = [ objective.marginalval( [elem], L ) for elem in N ]
    ele_vals = parallel_margvals_returnvals(objective, L, N, comm, rank, size)

    ele_vals_sortidx = np.argsort(ele_vals)[::-1]
    bestVal_idx = ele_vals_sortidx[0]
    L.append( N[bestVal_idx] )
    # Cov = objective.updateCov(L)
    queries += len(N)
    lazy_idx = 1

    # On remaining iterations, we update values lazily
    for i in range(1,k):
        if i%25==0:
            print('LazyGreedy round', i, 'of', k)

        # If new marginal value of best remaining ele after prev iter exceeds the *prev* marg value of 2nd best, add it
        queries += 1
        if objective.marginalval( [ N[ ele_vals_sortidx[ lazy_idx ] ] ], L ) >= ele_vals[ ele_vals_sortidx[1+lazy_idx] ]:
            #print('Found a lazy update')
            L.append( N[ele_vals_sortidx[lazy_idx]] )
            L_rounds.append([ele for ele in L])
            time_rounds.append( (datetime.now() - time0).total_seconds() )
            query_rounds.append(queries)

            lazy_idx += 1

        else:
            # If we did any lazy update iterations, we need to update bookkeeping for ground set
            N = list(set(N) - set(L)) 

            # Compute the marginal addition for each elem in N, then add the best one to solution L; 
            # Then remove it from remaning elements N
            # ele_vals = [ objective.marginalval( [elem], L ) for elem in N ]
            ele_vals = parallel_margvals_returnvals(objective, L, N, comm, rank, size)
            ele_vals_sortidx = np.argsort(ele_vals)[::-1]
            bestVal_idx = ele_vals_sortidx[0]
            L.append( N[bestVal_idx] )
            #print('LG adding this value to solution:', ele_vals[bestVal_idx] )

            queries += len(N)
            L_rounds.append([ele for ele in L])
            time_rounds.append( (datetime.now() - time0).total_seconds() )
            query_rounds.append(queries)

            lazy_idx = 1

    return L

def MED_lazygreedy(objective, k, S_prev, N, nthreads=16):
    ''' 
    Accelerated lazy greedy algorithm: for k steps (Minoux 1978).
    **NOTE** solution sets and values may be different than those found by our Greedy implementation, 
    as the two implementations break ties differently (i.e. when two elements 
    have the same marginal value,  the two implementations may not pick the same element to add)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint
    nthreads -- Number of threads to use on each machine
    
    OUTPUTS:
    list L -- the solution, where each element in the list is an element in the solution set.
    '''
    check_inputs(objective, k)
    queries = 0
    time0 = datetime.now()
    L = []
    # Cov = []
    L_rounds = [[]] # to hold L's evolution over rounds. Sometimes the new round does not add an element. 
    time_rounds = [0]
    query_rounds = [0]
    # N = [ele for ele in objective.groundset]
    if(k>=len(N)):
        return N
    # On the first iteration, LazyGreedy behaves exactly like regular Greedy:
    # ele_vals = [ objective.marginalval( [elem], L ) for elem in N ]
    ele_vals = parallel_margvals_returnvals_thread(objective, list( set(L).union(S_prev)), N, nthreads)

    ele_vals_sortidx = np.argsort(ele_vals)[::-1]
    bestVal_idx = ele_vals_sortidx[0]
    L.append( N[bestVal_idx] )
    # Cov = objective.updateCov(L)
    queries += len(N)
    lazy_idx = 1

    
    # print("(MED_lazygreedy)  k:",k, "  len(S_prev):", len(S_prev), "  S_prev:", S_prev)
    # On remaining iterations, we update values lazily
    for i in range(1,k):
        # if i%25==0:
        #     print('LazyGreedy round', i, 'of', k)

        # If new marginal value of best remaining ele after prev iter exceeds the *prev* marg value of 2nd best, add it
        queries += 1
        if objective.marginalval( [ N[ ele_vals_sortidx[ lazy_idx ] ] ], list( set(L).union(S_prev)) ) >= ele_vals[ ele_vals_sortidx[1+lazy_idx] ]:
            #print('Found a lazy update')
            L.append( N[ele_vals_sortidx[lazy_idx]] )
            L_rounds.append([ele for ele in L])
            time_rounds.append( (datetime.now() - time0).total_seconds() )
            query_rounds.append(queries)

            lazy_idx += 1

        else:
            # If we did any lazy update iterations, we need to update bookkeeping for ground set
            N = list(set(N) - set(list( set(L).union(S_prev)))) 

            # Compute the marginal addition for each elem in N, then add the best one to solution L; 
            # Then remove it from remaning elements N
            ele_vals = [ objective.marginalval( [elem], list( set(L).union(S_prev)) ) for elem in N ]
            ele_vals_sortidx = np.argsort(ele_vals)[::-1]
            bestVal_idx = ele_vals_sortidx[0]
            L.append( N[bestVal_idx] )
            #print('LG adding this value to solution:', ele_vals[bestVal_idx] )

            queries += len(N)
            L_rounds.append([ele for ele in L])
            time_rounds.append( (datetime.now() - time0).total_seconds() )
            query_rounds.append(queries)

            lazy_idx = 1

    return L

def MED_Dist_Greedy(objective, k, S_prev, V, comm, rank, size, p_root=0, seed=42, nthreads=16):

    check_inputs(objective, k)
    comm.barrier()
    V_split_local = V[rank] # np.array_split(V, size)[rank]
    
    # # Compute the marginal addition for each elem in N, then add the best one to solution L; remove it from remaining elements N
    # ele_A_local_vals = [ lazygreedy(objective, k, list(V_split_local.flatten()), nthreads)]
    ele_A_local_vals = [ MED_lazygreedy(objective, k, S_prev, V_split_local, nthreads)]
    # # Gather the partial results to all processes
    ele_vals = comm.allgather(ele_A_local_vals)
    
    return [val for sublist in ele_vals for val in sublist]

def MED_RandGreedI(objective, k, S_prev, eps, comm, rank, size, p_root=0, seed=42, nthreads=16):
    '''
    The parallelizable distributed greedy algorithm RandGreedI. Uses multiple machines to obtain solution
    PARALLEL IMPLEMENTATION (Multithread)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    nthreads -- Number of threads to use on each machine

    OUTPUTS:
    list S -- the solution, where each element in the list is an element in the solution set.
    float f(S) -- the value of the solution
    time -- runtime of the algorithm
    time_dist -- runtime spent on the distributed part of the algorithm
    time_post -- runtime spent on the post processing part of the algorithm
    ''' 

    comm.barrier()
    p_start = MPI.Wtime()
    
    S = []
    time_rounds = [0]
    query_rounds = [0]
    queries = 0
    random.seed(seed)
    # V = [ ele for ele in objective.groundset ];
    # random.Random(seed).shuffle(V)
    
    q = np.random.RandomState(42)
    V_all = [ ele for ele in objective.groundset ];
    
    V = [[] for i in range(size)]
    for ele in objective.groundset:
        x = random.randint(0, size-1)
        V[x].append(ele)
    # Get the solution of parallel QS
    p_start_dist = MPI.Wtime()
    if(rank==0):
        print("(MED_RandGreedI)  k:",k, "  S_prev:", S_prev)
    S_DistGreedy_split = MED_Dist_Greedy(objective, k, S_prev, V, comm, rank, size, p_root, seed, nthreads)

    S_DistGreedy = []
    S_DistGreedy_all = [] 
    for i in range(len(S_DistGreedy_split)):
        if(rank==0):
            print("(MED_RandGreedI)  len(S_DistGreedy_split[", i,"]):",len(S_DistGreedy_split[i]))
        S_DistGreedy.extend(list(S_DistGreedy_split[i]))
        S_DistGreedy_all.append(list(S_DistGreedy_split[i]))
    S_DistGreedy = list(np.unique(S_DistGreedy))
    p_stop_dist = MPI.Wtime()
    
    time_dist = (p_stop_dist - p_start_dist)
    
    p_start_post = MPI.Wtime()
    # S_DistGreedy_split_vals = parallel_val_of_sets_thread(objective, S_DistGreedy_split, nthreads)
    S_DistGreedy_split_vals = parallel_val_of_sets_thread(objective, S_DistGreedy_split, nthreads=objective.nThreads)
    

    S_p = np.argmax(S_DistGreedy_split_vals)
    
    S_p_val = np.max(S_DistGreedy_split_vals)

    if(rank==0):
        print("(MED_RandGreedI)  k:",k, "  len(S_DistGreedy):", len(S_DistGreedy), "  S_prev:", S_prev)
    T = MED_lazygreedy(objective, k, S_prev, S_DistGreedy, nthreads=objective.nThreads)

    if(rank==0):
        print("(MED_RandGreedI)  len(T):",len(T), "  objective.value(T):", objective.value(T), "  S_prev:", S_prev, "  len(S_DistGreedy_split[S_p]):", len(S_DistGreedy_split[S_p]))

    if(objective.value(T)>S_p_val):
        S = T
    else:
        S = list(S_DistGreedy_split[S_p])
    # print(S)
    p_stop = MPI.Wtime()
    time_post = (p_stop - p_start_post)
    time = (p_stop - p_start)
    valSol = objective.value( S )
    # Added increment
    queries += 1
    if rank == p_root:
        print ('MED_RandGreedI:', valSol, time, 'with k=', k, 'n=', len(objective.groundset), 'eps=', eps, '|S|=', len(S))
    
    return S

def MEDRG(objective, k, eps, comm, rank, size, p_root, seed=42, nthreads=16):
    '''
    The parallelizable distributed greedy algorithm MED running RandGreedI over multiple rounds. Uses multiple machines to obtain solution (Algorithm 2)
    PARALLEL IMPLEMENTATION (Multithread)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    list S_prev -- the solution S from the previous iteration
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    bool stop_if_aprx: determines whether we exit as soon as we find OPT that reaches approx guarantee or keep searching for better solution
    nthreads -- Number of threads to use on each machine

    OUTPUTS:
    list S -- the solution, where each element in the list is an element in the solution set.
    float f(S) -- the value of the solution
    time -- runtime of the algorithm
    time_dist -- runtime spent on the distributed part of the algorithm
    time_post -- runtime spent on the post processing part of the algorithm
    '''

    comm.barrier()
    p_start = MPI.Wtime()
    
    S = []
    S_rounds = [[]]
    time_rounds = [0]
    query_rounds = [0]
    queries = 0
    V = [ ele for ele in objective.groundset ];
    np.random.shuffle(V)
    #k_max = np.max(k)iter
    beta = 1
    k_p = int(np.ceil(beta * len(objective.groundset)/ (size ** 2)))
    iters = int(np.ceil(k/k_p))
    k_p = min(k, k_p)
    
    time_post = 0
    time_dist = 0
    if(rank==0):
        print("(MEDRG) iters: ", iters, "  k_p:", k_p, "  k:",k, "  S:", S)
    for i in range(iters):
        p_start_dist = MPI.Wtime()
        if(k - len(S) < k_p): #Last iteration
            k_p = k - len(S)

        
        # T = MED_RandGreedI(objective, k, S_prev, eps, comm, rank, size, p_root, seed, nthreads)
        if(rank==0):
            print("(MEDRG) iter: ", i, "  k_p:", k_p, "  k:",k, "  S:", S)
        T = MED_RandGreedI(objective, k_p, S, eps, comm, rank, size, p_root, seed, nthreads)
        if(rank==0):
            print("(MEDRG) iter: ", i, "   T:", T)
        # valSolT, queriesT, timeT, T, T_DistGB_split, time_roundsT, query_roundsT = RandGreedI_PGB(objective, k_split, eps, comm, rank, size, p_root=0, seed=42, stop_if_apx=False, nthreads=16)
        comm.barrier()
        p_stop_dist = MPI.Wtime()
        time_dist += (p_stop_dist - p_start_dist)
        T = list(np.unique(T))
        T = list( set(T)-set(S) )
        comm.barrier()
        
        S.extend(T)
        S = list(np.unique(S))
        S_rounds.append(S)
        # if(rank==0):
        #     print("Iteration: ", i, "  Len(S):", len(S), "  Val(S): ", objective.value( S ), "  MargeVal(T): ", objective.marginalval(T_DistGB, S))
        if (len(S) >= k):
            # S = S[-k:]
            break;

    comm.barrier()
    # print("OUTSIDE FOR LOOP")
    # if len(S) > K, select last k elements added
    S = np.unique(S)
    if len(S) > k:
        S = S[-k:]
    p_stop = MPI.Wtime()
    time = (p_stop - p_start)
    valSol = objective.value( S )
    # Added increment
    queries += 1
    if rank == p_root:
        print ('MED+RandGreedI:', valSol, queries, time, 'with k=', k, 'n=', len(objective.groundset), 'eps=', eps, '|S|=', len(S))

    return valSol, time, time_dist, time_post, S





# LS+PGB [Chen et al. 2021]
def parallel_threshold_sample(V, S, objective, tau, eps, delta, k, comm, rank, size, randstate, pastGains):
    
    '''
    The parallelizable greedy algorithm THRESHOLDSEQ for Fixed Threshold 'tau'
    PARALLEL IMPLEMENTATION
    
    INPUTS:
    list V - contains the elements currently in the groundset
    list S - contains the elements currently in the solution set
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    float tau -- the fixed threshold
    float eps -- the error tolerance between 0 and 1
    float delta -- the probability tolerance between 0 and 1
    int k -- the cardinality constraint (must be k>0)
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())
    randstate -- random seed to use when drawing samples 
    pastGains -- each element is the current marginal gain of the corresponding element in groundset

    OUTPUTS:
    list pastGains -- each element is the current marginal gain of the corresponding element in groundset
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    list S -- the solution, where each element in the list is an element with marginal values > tau.

    '''  
    
    comm.barrier()

    ell = int(np.ceil( (4.0 + 8.0/ eps ) * np.log( len(V) / delta ) ) ) + 1;
    V = list( np.sort( list( set(V)-set(S) ) ) );
    queries = 0
    for itr in range( ell ):
        s = np.min( [k - len(S), len(V) ] );
        
        seq = sample_seq( V, s, randstate );

        lmbda = make_lmbda( eps, s, s );
        
        B = parallel_adaptiveAdd(lmbda, seq, S, objective, eps, k, comm, rank, size, tau)
        #Added query increment
        queries += len(lmbda)
        lmbda_star = lmbda[0]
        if len(B) > 1:
            for i in range(1,len(B)):
                if(B[i]):
                    if (i == len(B) - 1):
                        lmbda_star = lmbda[-1];
                else:
                    lmbda_star = lmbda[i]
                    break;

        #Add elements
        #T= parallel_pessimistically_add_x_seq(objective, S, seq, tau, comm, rank , size );
        T = set(seq[0:lmbda_star]);
        
        for i in range(lmbda_star, len(B)):
            if (B[i]):
                T = set().union(T, seq[ lmbda[ i - 1 ] : lmbda[ i ] ]);
        S= list(set().union(S, T))
        V = list( np.sort( list( set(V)-set(S) ) ) );
        
        gains = parallel_margvals_returnvals(objective, S, V, comm, rank, size)
        #Added query increment
        queries += len(V)
        
        for ps in range( len(gains )):
            pastGains[ V[ps] ] = gains[ ps ];
        
        V_above_thresh = np.where(gains >= tau)[0]
        V = [ V[idx] for idx in V_above_thresh ];
        if (len(V) == 0):
            break;
        if (len(S) == k):
            break;

    if (len(V) > 0 and len(S) < k):
        # if (rank == 0):
        print( "ThresholdSample has failed. This should be an extremely rare event. Terminating program..." );
        exit(1);
            
    return [pastGains, S, queries];

def LinearSeq(objective, k, eps, comm, rank, size, p_root=0, seed=42, stop_if_approx=True):
    '''
    The preprocessing algorithm LINEARSEQ for Submodular Mazimization
    PARALLEL IMPLEMENTATION
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    bool stop_if_approx: determines whether we exit as soon as we find OPT that reaches approx guarantee or keep searching for better solution

    OUTPUTS:
    float f(S) -- the value of the solution
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time -- the processing time to optimize the function.
    list S -- the solution, where each element in the list is an element in the solution set.
    list of lists S_rounds -- each element is a list containing the solution set S at the corresponding round.
    list of lists time_rounds -- each element is a list containing the time at the corresponding round
    list of lists query_rounds -- each element is a list containing the number of queries at the corresponding round
    list singletonVals -- each element is the current marginal gain of the corresponding element in groundset
    '''    
    comm.barrier()
    p_start = MPI.Wtime()    
    n = len(objective.groundset)    

    # Each processor gets its own random state so they can independently draw IDENTICAL random sequences a_seq.
    #proc_random_states = [np.random.RandomState(seed) for processor in range(size)]
    randstate = np.random.RandomState(seed)
    queries = 0
    
    S = []
    S_rounds = [[]]
    time_rounds = [0]
    query_rounds = [0]
    #I1 = make_I(eps, k)

    currGains = parallel_margvals_returnvals(objective, [], [ele for ele in objective.groundset], comm, rank, size)
    #Added query increment
    queries += len(objective.groundset)
    #currGains = parallel_margvals_returnvals(objective, [], [ele for ele in objective.groundset], comm, rank, size)

    #initialize S to top singleton
    S = [np.argmax(currGains)];
    singletonIdcs = np.argsort(currGains);
    singletonVals = currGains;
    currGains = np.sort(currGains);
    #run first considering the top singletons as the universe
    V = np.array(objective.groundset)[ singletonIdcs[-3*k:] ]
    
    currGains = currGains[-3*k:];
    valtopk = np.sum( currGains[-k:] );

    while len(V) > 0:
        # Filtering
        t = objective.value(S)/np.float(k)
        # Added increment
        queries += 1
        if stop_if_approx:
            # if (rank == 0):
            #     print( "checking ratio: ", 0.5*valtopk, t * k );
            if (t >= 0.5 * valtopk / np.float(k)):
                # if (rank == 0):
                #     print( "FLS stopping early, approx reached." );
                if(len(S)>k):
                    Ap = S[len(S) - k : len(S)]
                else:
                    Ap = S;
                valAp = objective.value(Ap)
                queries += 1
                comm.barrier()
                p_stop = MPI.Wtime()
                time = (p_stop - p_start)
                # if rank == p_root:
                #     print ('FLS:', valAp, queries, time, 'with k=', k, 'n=', n, 'eps=', eps, '|S|=', len(Ap))
    
                return valAp, queries, time, S, S_rounds, time_rounds, query_rounds, singletonVals
        
        #do lazy discard first, based on last round's computations
        V_above_thresh = np.where(currGains >= t)[0]
        V = list( set([ V[idx] for idx in V_above_thresh ] ));

        #now recompute requisite gains
        # if (rank == 0):
        #     print( len(V) );
        #     print( "starting pmr..." );
        currGains = parallel_margvals_returnvals(objective, S, [ele for ele in V], comm, rank, size)
        #Added query increment
        queries += len(V)
        #currGains = parallel_margvals_returnvals(objective, S, [ele for ele in V], comm, rank, size)
        # if ( rank == 0):
        #     print( "done.");
        V_above_thresh = np.where(currGains >= t)[0]
        V = [ V[idx] for idx in V_above_thresh ];
        currGains = [ currGains[idx] for idx in V_above_thresh ];
        ###End Filtering
        
        # if (rank == p_root):
        #     print( len(V) );

        if (len(V) == 0):
            break;
        
        # Radnom Permutation
        randstate.shuffle(V);
        lmbda = make_lmbda( eps, k, len(V) );

        # if (rank == p_root):
        #     print("starting adaptiveAdd...");
        B = parallel_adaptiveAdd(lmbda, V, S, objective, eps, k, comm, rank, size)
        
        # if (rank == p_root):
        #     print("done.");
        #Added query increment
        queries += len( lmbda );
        query_rounds.append(queries)
        # time_rounds.append( MPI.Wtime() - p_start ) 
        lmbda_star = lmbda[0]
        
        if len(B) > 1:
            n_LAG = 1
            unionT = 1
            for i in range(1,len(B)):
                if(B[i]):
                    n_LAG += 1
                    unionT += lmbda[i] - lmbda[i-1]
                    if (i == len(B) - 1):
                        lmbda_star = lmbda[-1];
                else:
                    if(lmbda[i - 1]<=k):
                        if(n_LAG == i):
                            lmbda_star = lmbda[i - 1]
                    else:
                        if(unionT >= k):
                            lmbda_star = lmbda[i - 1]
                    n_LAG = 0
                    unionT = 0


        T = set(V[0:lmbda_star])
        for i in range(lmbda_star, len(B)):
            if (B[i]):
                T = set().union(T, V[ lmbda[ i - 1 ] : lmbda[ i ] ]);
        # if (rank == p_root):
        #     print("done.");
        # S= list(set().union(S, T))
        S.extend(list(T))
        indexes = np.unique(S, return_index=True)[1]
        S = [S[index] for index in sorted(indexes)]
        S_rounds.append([ele for ele in S])
        query_rounds.append(queries)
        # time_rounds.append( MPI.Wtime() - p_start ) 

        # if (rank == p_root):
        #     print( "Lambda_Star: " , len(T) );

        
        
    t = objective.value(S) / np.float( k );
    queries += 1
    # if(lazy):
    V_above_thresh = np.where(singletonVals >= t)[0]
    V = [ objective.groundset[idx] for idx in V_above_thresh ]; 
    currGains = [ singletonVals[idx] for idx in V_above_thresh ];
    
    while len(V) > 0:
        # Filtering
        t = objective.value(S)/np.float(k)
        queries += 1
        if stop_if_approx:
            # if (rank == 0):
            #     print( "checking ratio: ", 0.5*valtopk, t * k );
            if (t >= 0.5 * valtopk / np.float(k)):
                # if (rank == 0):
                #     print( "FLS stopping early, approx reached." );
                if(len(S)>k):
                    Ap = S[len(S) - k : len(S)]
                else:
                    Ap = S;
                valAp = objective.value(Ap)
                queries += 1
                comm.barrier()
                p_stop = MPI.Wtime()
                time = (p_stop - p_start)
                if rank == p_root:
                    print ('FLS:', valAp, queries, time, 'with k=', k, 'n=', n, 'eps=', eps, '|S|=', len(Ap))
    
                return valAp, queries, time, S, S_rounds, time_rounds, query_rounds, singletonVals
       
        # if(lazy):
        
        #do lazy discard first, based on last round's computations
        V_above_thresh = np.where(currGains >= t)[0]
        V = list( set([ V[idx] for idx in V_above_thresh ] ));

        #now recompute requisite gains
        # if (rank == 0):
        #     print( len(V) );
        #     print( "starting pmr..." );
        currGains = parallel_margvals_returnvals(objective, S, [ele for ele in V], comm, rank, size)
        #Added query increment
        queries += len(V)
        #currGains = parallel_margvals_returnvals(objective, S, [ele for ele in V], comm, rank, size)
        # if ( rank == 0):
        #     print( "done.");
        V_above_thresh = np.where(currGains >= t)[0]
        V = [ V[idx] for idx in V_above_thresh ];
        currGains = [ currGains[idx] for idx in V_above_thresh ];

        if (len(V) == 0):
            break;
        
        # if (rank == p_root):
        #     print( len(V) );
        # Radnom Permutation
        randstate.shuffle(V);
        lmbda = make_lmbda( eps, k, len(V) );

        # if (rank == p_root):
        #     print("starting adaptiveAdd...");
        B = parallel_adaptiveAdd(lmbda, V, S, objective, eps, k, comm, rank, size)
        
        # if (rank == p_root):
        #     print("done.");
        
        queries += len( lmbda );
        query_rounds.append(queries)
        # time_rounds.append( MPI.Wtime() - p_start ) 
        lmbda_star = lmbda[0]
        
        if len(B) > 1:
            n_LAG = 1
            unionT = 1
            for i in range(1,len(B)):
                if(B[i]):
                    n_LAG += 1
                    unionT += lmbda[i] - lmbda[i-1]
                    if (i == len(B) - 1):
                        lmbda_star = lmbda[-1];
                else:
                    if(lmbda[i - 1]<=k):
                        if(n_LAG == i):
                            lmbda_star = lmbda[i - 1]
                    else:
                        if(unionT >= k):
                            lmbda_star = lmbda[i - 1]
                    n_LAG = 0
                    unionT = 0


        T = list(set(V[0:lmbda_star]))
        #T = parallel_pessimistically_add_x_seqVar( objective, S, V,k, comm, rank, size );
        # if (rank == p_root):
        #    print("done.");
        # S= list(set().union(S, T))
        S.extend(list(T))
        indexes = np.unique(S, return_index=True)[1]
        S = [S[index] for index in sorted(indexes)]
        S_rounds.append([ele for ele in S])
        query_rounds.append(queries)
        # time_rounds.append( MPI.Wtime() - p_start ) 

        if (rank == p_root):
            print( "Lambda_Star: " , len(T) );

        
    
    if(len(S)>k):
        Ap = S[len(S) - k : len(S)]
    else:
        Ap = S;
    valAp = objective.value(Ap)
    queries += 1

    comm.barrier()
    p_stop = MPI.Wtime()
    time = (p_stop - p_start)
    if rank == p_root:
        print ('FLS:', valAp, queries, time, 'with k=', k, 'n=', n, 'eps=', eps, '|S|=', len(Ap))
    
    return valAp, queries, time, S, S_rounds, time_rounds, query_rounds, singletonVals

def ParallelGreedyBoost_Original_MultiNode(objective, k, eps, comm, rank, size, p_root=0, seed=42, stop_if_approx=True):

    '''
    The parallelizable greedy algorithm PARALLELGREEDYBOOST to Boost to the Optimal Ratio.
    PARALLEL IMPLEMENTATION
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    bool stop_if_approx: determines whether we exit as soon as we find OPT that reaches approx guarantee or keep searching for better solution

    OUTPUTS:
    float f(S) -- the value of the solution
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time -- the processing time to optimize the function.
    ''' 
    
    comm.barrier()
    p_start = MPI.Wtime()
    eps_FLS = 0.21

    Gamma, queries, time, sol, sol_r, time_r, queries_r, pastGains = LinearSeq(objective, k, eps_FLS, comm, rank, size, p_root, seed, True );
    
    #alpha = 1.0 / (4.0 + (2*(3+(2*eps_FLS)))/(1 - (2*eps_FLS) - (2*(eps_FLS ** 2)))*eps_FLS);
    alpha = 1.0 / (4.0 + (4*(2-eps_FLS))/((1 - eps_FLS)*(1-(2*eps_FLS))) *eps_FLS);
    valtopk = np.sum( np.sort(pastGains)[-k:] );
    stopRatio = (1.0 - 1.0/np.exp(1) - eps)*valtopk;
    #stopRatio = 0.75*valtopk;
    #stopRatio = 0.85*valtopk;
    if stop_if_approx:
        if Gamma >= stopRatio:
            comm.barrier()
            p_stop = MPI.Wtime()
            time = (p_stop - p_start)
            valSol = Gamma
            if (rank == 0):
                print("ABR stopping early.");

            return Gamma, queries, time, sol, sol_r, time_r, queries_r
    
    # Each processor gets its own random state so they can independently draw IDENTICAL random sequences a_seq.
    #proc_random_states = [np.random.RandomState(seed) for processor in range(size)]
    randstate = np.random.RandomState(seed)
    
    S = []
    S_rounds = [[]]
    time_rounds = [0]
    query_rounds = [0]
    #I1 = make_I(eps, k)
    
    tau = Gamma / (alpha * np.float(k));
    taumin = Gamma / (3.0 * np.float(k));
        
    V = [ ele for ele in objective.groundset ];

    if (tau > valtopk / np.float(k)):
        tau = valtopk / np.float(k);
    
    #pastGains = np.inf*np.ones(len(V));
    while (tau > taumin):
        tau = np.min( [tau, np.max( pastGains ) * (1.0 - eps)] );
        if (tau <= 0.0):
            tau = taumin;
        V_above_thresh = np.where(pastGains >= tau)[0]
        V_above_thresh = [ V[ele] for ele in V_above_thresh ];
        V_above_thresh = list( set(V_above_thresh) - set(S) );
        
        currGains = parallel_margvals_returnvals(objective, S, V_above_thresh, comm, rank, size)
        #Added query increment
        queries += len(V_above_thresh)
        
        for ps in range( len(V_above_thresh )):
            pastGains[ V_above_thresh[ps]] = currGains[ ps ];
        V_above_thresh_id = np.where(currGains >= tau)[0]
        V_above_thresh = [ V_above_thresh[ele] for ele in V_above_thresh_id ];
        
        if (len(V_above_thresh) > 0):
            [pastGains, S, queries_tmp] = parallel_threshold_sample( V_above_thresh, S, objective, tau, eps / np.float(3), 1.0 / ((np.log( alpha / 3 ) / np.log( 1 - eps ) ) + 1) , k, comm, rank, size, randstate, pastGains );
            #Added query increment
            queries += queries_tmp
            for ele in S:
                pastGains[ele] = 0;

        if (len(S) >= k):
            break;
        if stop_if_approx:
            if objective.value(S) >= stopRatio:
                comm.barrier()
                p_stop = MPI.Wtime()
                time = (p_stop - p_start)
                valSol = objective.value( S )
                #Added increment
                queries += 1
                if (rank == 0):
                    print("ABR stopping early.");
                return valSol, queries, time, sol, sol_r, time_r, queries_r
    
    comm.barrier()
    p_stop = MPI.Wtime()
    time = (p_stop - p_start)
    valSol = objective.value( S )
    # Added increment
    queries += 1
    if rank == p_root:
        print ('PGB:', valSol, queries, time, 'with k=', k, 'n=', len(objective.groundset), 'eps=', eps, '|S|=', len(S))
    

    return valSol, time, S, S_rounds, time_rounds, query_rounds





# Lazy Greedy Algorithm (Utility functions for RandGreedI and BiCriteriaGreedy)
def lazygreedy_MultiNode(objective, k, N, comm, rank, size, nthreads=16):
    ''' 
    Accelerated lazy greedy algorithm: for k steps (Minoux 1978).
    **NOTE** solution sets and values may be different than those found by our Greedy implementation, 
    as the two implementations break ties differently (i.e. when two elements 
    have the same marginal value,  the two implementations may not pick the same element to add)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint
    
    OUTPUTS:
    float f(L) -- the value of the solution
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time -- the processing time to optimize the function.
    list L -- the solution, where each element in the list is an element in the solution set.
    '''
    check_inputs(objective, k)
    queries = 0
    time0 = datetime.now()
    L = []
    # Cov = []
    L_rounds = [[]] # to hold L's evolution over rounds. Sometimes the new round does not add an element. 
    time_rounds = [0]
    query_rounds = [0]
    # N = [ele for ele in objective.groundset]
    if(k>=len(N)):
        return N
    # On the first iteration, LazyGreedy behaves exactly like regular Greedy:
    # ele_vals = [ objective.marginalval( [elem], L ) for elem in N ]
    ele_vals = parallel_margvals_returnvals(objective, L, N, comm, rank, size)

    ele_vals_sortidx = np.argsort(ele_vals)[::-1]
    bestVal_idx = ele_vals_sortidx[0]
    L.append( N[bestVal_idx] )
    # Cov = objective.updateCov(L)
    queries += len(N)
    lazy_idx = 1

    # On remaining iterations, we update values lazily
    for i in range(1,k):
        if i%25==0:
            print('LazyGreedy round', i, 'of', k)

        # If new marginal value of best remaining ele after prev iter exceeds the *prev* marg value of 2nd best, add it
        queries += 1
        if objective.marginalval( [ N[ ele_vals_sortidx[ lazy_idx ] ] ], L ) >= ele_vals[ ele_vals_sortidx[1+lazy_idx] ]:
            #print('Found a lazy update')
            L.append( N[ele_vals_sortidx[lazy_idx]] )
            L_rounds.append([ele for ele in L])
            time_rounds.append( (datetime.now() - time0).total_seconds() )
            query_rounds.append(queries)

            lazy_idx += 1

        else:
            # If we did any lazy update iterations, we need to update bookkeeping for ground set
            N = list(set(N) - set(L)) 

            # Compute the marginal addition for each elem in N, then add the best one to solution L; 
            # Then remove it from remaning elements N
            # ele_vals = [ objective.marginalval( [elem], L ) for elem in N ]
            ele_vals = parallel_margvals_returnvals(objective, L, N, comm, rank, size)
            ele_vals_sortidx = np.argsort(ele_vals)[::-1]
            bestVal_idx = ele_vals_sortidx[0]
            L.append( N[bestVal_idx] )
            #print('LG adding this value to solution:', ele_vals[bestVal_idx] )

            queries += len(N)
            L_rounds.append([ele for ele in L])
            time_rounds.append( (datetime.now() - time0).total_seconds() )
            query_rounds.append(queries)

            lazy_idx = 1

    return L

def lazygreedy(objective, k, N, nthreads=16):
    ''' 
    Accelerated lazy greedy algorithm: for k steps (Minoux 1978).
    **NOTE** solution sets and values may be different than those found by our Greedy implementation, 
    as the two implementations break ties differently (i.e. when two elements 
    have the same marginal value,  the two implementations may not pick the same element to add)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint
    nthreads -- Number of threads to use on each machine
    
    OUTPUTS:
    list L -- the solution, where each element in the list is an element in the solution set.
    '''
    check_inputs(objective, k)
    queries = 0
    time0 = datetime.now()
    L = []
    # Cov = []
    L_rounds = [[]] # to hold L's evolution over rounds. Sometimes the new round does not add an element. 
    time_rounds = [0]
    query_rounds = [0]
    # N = [ele for ele in objective.groundset]
    if(k>=len(N)):
        return N
    # On the first iteration, LazyGreedy behaves exactly like regular Greedy:
    # ele_vals = [ objective.marginalval( [elem], L ) for elem in N ]
    ele_vals = parallel_margvals_returnvals_thread(objective, L, N, nthreads)

    ele_vals_sortidx = np.argsort(ele_vals)[::-1]
    bestVal_idx = ele_vals_sortidx[0]
    L.append( N[bestVal_idx] )
    # Cov = objective.updateCov(L)
    queries += len(N)
    lazy_idx = 1

    # On remaining iterations, we update values lazily
    for i in range(1,k):
        if i%25==0:
            print('LazyGreedy round', i, 'of', k)

        # If new marginal value of best remaining ele after prev iter exceeds the *prev* marg value of 2nd best, add it
        queries += 1
        if objective.marginalval( [ N[ ele_vals_sortidx[ lazy_idx ] ] ], L ) >= ele_vals[ ele_vals_sortidx[1+lazy_idx] ]:
            #print('Found a lazy update')
            L.append( N[ele_vals_sortidx[lazy_idx]] )
            L_rounds.append([ele for ele in L])
            time_rounds.append( (datetime.now() - time0).total_seconds() )
            query_rounds.append(queries)

            lazy_idx += 1

        else:
            # If we did any lazy update iterations, we need to update bookkeeping for ground set
            N = list(set(N) - set(L)) 

            # Compute the marginal addition for each elem in N, then add the best one to solution L; 
            # Then remove it from remaning elements N
            ele_vals = [ objective.marginalval( [elem], L ) for elem in N ]
            ele_vals_sortidx = np.argsort(ele_vals)[::-1]
            bestVal_idx = ele_vals_sortidx[0]
            L.append( N[bestVal_idx] )
            #print('LG adding this value to solution:', ele_vals[bestVal_idx] )

            queries += len(N)
            L_rounds.append([ele for ele in L])
            time_rounds.append( (datetime.now() - time0).total_seconds() )
            query_rounds.append(queries)

            lazy_idx = 1

    return L

def Dist_Greedy(objective, k, V, comm, rank, size, p_root=0, seed=42, nthreads=16):

    check_inputs(objective, k)
    comm.barrier()
    V_split_local = V[rank] # np.array_split(V, size)[rank]
    
    # # Compute the marginal addition for each elem in N, then add the best one to solution L; remove it from remaining elements N
    # ele_A_local_vals = [ lazygreedy(objective, k, list(V_split_local.flatten()), nthreads)]
    ele_A_local_vals = [ lazygreedy(objective, k, V_split_local, nthreads)]
    # # Gather the partial results to all processes
    ele_vals = comm.allgather(ele_A_local_vals)
    
    return [val for sublist in ele_vals for val in sublist]

# RandGreedI [Barbosa et al. 2015]
def RandGreedI(objective, k, eps, comm, rank, size, p_root=0, seed=42, nthreads=16):
    '''
    The parallelizable distributed greedy algorithm RandGreedI. Uses multiple machines to obtain solution
    PARALLEL IMPLEMENTATION (Multithread)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    nthreads -- Number of threads to use on each machine

    OUTPUTS:
    list S -- the solution, where each element in the list is an element in the solution set.
    float f(S) -- the value of the solution
    time -- runtime of the algorithm
    time_dist -- runtime spent on the distributed part of the algorithm
    time_post -- runtime spent on the post processing part of the algorithm
    ''' 

    comm.barrier()
    p_start = MPI.Wtime()
    
    S = []
    time_rounds = [0]
    query_rounds = [0]
    queries = 0
    random.seed(seed)
    # V = [ ele for ele in objective.groundset ];
    # random.Random(seed).shuffle(V)
    
    q = np.random.RandomState(42)
    V_all = [ ele for ele in objective.groundset ];
    
    V = [[] for i in range(size)]
    for ele in objective.groundset:
        x = random.randint(0, size-1)
        V[x].append(ele)
    # Get the solution of parallel QS
    p_start_dist = MPI.Wtime()
    
    S_DistGreedy_split = Dist_Greedy(objective, k, V, comm, rank, size, p_root, seed, nthreads)

    S_DistGreedy = []
    S_DistGreedy_all = [] 
    for i in range(len(S_DistGreedy_split)):
        S_DistGreedy.extend(list(S_DistGreedy_split[i]))
        S_DistGreedy_all.append(list(S_DistGreedy_split[i]))
    S_DistGreedy = list(np.unique(S_DistGreedy))
    p_stop_dist = MPI.Wtime()
    
    time_dist = (p_stop_dist - p_start_dist)
    
    p_start_post = MPI.Wtime()
    # S_DistGreedy_split_vals = parallel_val_of_sets_thread(objective, S_DistGreedy_split, nthreads)
    S_DistGreedy_split_vals = parallel_val_of_sets_thread(objective, S_DistGreedy_split, nthreads=objective.nThreads)
    

    S_p = np.argmax(S_DistGreedy_split_vals)
    
    S_p_val = np.max(S_DistGreedy_split_vals)

    T = lazygreedy(objective, k, S_DistGreedy, nthreads=objective.nThreads)

    if(objective.value(T)>S_p_val):
        S = T
    else:
        S = list(S_DistGreedy_split[S_p])
    # print(S)
    p_stop = MPI.Wtime()
    time_post = (p_stop - p_start_post)
    time = (p_stop - p_start)
    valSol = objective.value( S )
    # Added increment
    queries += 1
    if rank == p_root:
        print ('RandGreedI:', valSol, time, 'with k=', k, 'n=', len(objective.groundset), 'eps=', eps, '|S|=', len(S))
    
    return valSol, time, time_dist, time_post, S, S_DistGreedy_split, time_rounds, query_rounds

# RandGreedI using LAG as ALG on the primary machine (Appendix Results)
def RandGreedI_LAG(objective, k, eps, comm, rank, size, p_root=0, seed=42, nthreads=16):
    '''
    The parallelizable distributed greedy algorithm RandGreedI. Uses multiple machines to obtain solution
    PARALLEL IMPLEMENTATION (Multithread)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    nthreads -- Number of threads to use on each machine

    OUTPUTS:
    list S -- the solution, where each element in the list is an element in the solution set.
    float f(S) -- the value of the solution
    time -- runtime of the algorithm
    time_dist -- runtime spent on the distributed part of the algorithm
    time_post -- runtime spent on the post processing part of the algorithm
    ''' 

    comm.barrier()
    p_start = MPI.Wtime()
    
    S = []
    time_rounds = [0]
    query_rounds = [0]
    queries = 0
    random.seed(seed)
    # V = [ ele for ele in objective.groundset ];
    # random.Random(seed).shuffle(V)
    
    q = np.random.RandomState(42)
    V_all = [ ele for ele in objective.groundset ];
    
    V = [[] for i in range(size)]
    for ele in objective.groundset:
        x = random.randint(0, size-1)
        V[x].append(ele)
    # Get the solution of parallel QS
    p_start_dist = MPI.Wtime()
    
    S_DistGreedy_split = Dist_Greedy(objective, k, V, comm, rank, size, p_root, seed, nthreads)

    S_DistGreedy = []
    S_DistGreedy_all = [] 
    for i in range(len(S_DistGreedy_split)):
        S_DistGreedy.extend(list(S_DistGreedy_split[i]))
        S_DistGreedy_all.append(list(S_DistGreedy_split[i]))
    S_DistGreedy = list(np.unique(S_DistGreedy))
    p_stop_dist = MPI.Wtime()
    
    time_dist = (p_stop_dist - p_start_dist)
    
    p_start_post = MPI.Wtime()
    # S_DistGreedy_split_vals = parallel_val_of_sets_thread(objective, S_DistGreedy_split, nthreads)
    S_DistGreedy_split_vals = parallel_val_of_sets_thread(objective, S_DistGreedy_split, nthreads=objective.nThreads)
    

    S_p = np.argmax(S_DistGreedy_split_vals)
    
    S_p_val = np.max(S_DistGreedy_split_vals)

    T = LAG_SingleNode(objective, k, eps, S_DistGreedy, V_all, q, [], seed, stop_if_approx=False, nthreads=objective.nThreads)

    if(objective.value(T)>S_p_val):
        S = T
    else:
        S = list(S_DistGreedy_split[S_p])
    # print(S)
    p_stop = MPI.Wtime()
    time_post = (p_stop - p_start_post)
    time = (p_stop - p_start)
    valSol = objective.value( S )
    # Added increment
    queries += 1
    if rank == p_root:
        print ('RandGreedI:', valSol, time, 'with k=', k, 'n=', len(objective.groundset), 'eps=', eps, '|S|=', len(S))
    
    return valSol, time, time_dist, time_post, S, S_DistGreedy_split, time_rounds, query_rounds

# BicriteriaGreedy [Epasto et al. 2017]
def BiCriteriaGreedy(objective, k, eps, comm, rank, size, p_root=0, seed=42, nthreads=16, run_Full=False):
    '''
    The parallelizable distributed greedy algorithm BiCriteriaGreedy. Uses multiple machines to obtain solution
    PARALLEL IMPLEMENTATION (Multithread)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    int rounds -- the number of MapReduce rounds the algorithm will be executed
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())
    
    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    nthreads -- Number of threads to use on each machine

    OUTPUTS:
    list S -- the solution, where each element in the list is an element in the solution set.
    float f(S) -- the value of the solution
    time -- runtime of the algorithm
    time_dist -- runtime spent on the distributed part of the algorithm
    time_post -- runtime spent on the post processing part of the algorithm
    ''' 

    comm.barrier()
    p_start = MPI.Wtime()
    
    # # Each processor gets its own random state so they can independently draw IDENTICAL random sequences a_seq.
    # #proc_random_states = [np.random.RandomState(seed) for processor in range(size)]
    # randstate = np.random.RandomState(seed)
    # time0 = datetime.now()

    rounds = 1 # As used in experiments by Epasto et. al. 
    alpha = 3/(pow(eps,(1/rounds)))
    jloop_iter = (pow(alpha,2)*pow(np.log(alpha),2) + np.log(alpha)) * k 
    
    S = []
    queries = 0
    random.seed(seed)
    # V = [ ele for ele in objective.groundset ];
    # random.Random(seed).shuffle(V)
    V = [[] for i in range(size)]
    for ele in objective.groundset:
        x = random.randint(0, size-1)
        V[x].append(ele)
    time_dist = 0
    # Get the solution of parallel QS
    p_start_dist = MPI.Wtime()
    rounds = 1
    
    # for l in range(rounds): # Since rounds = 1 for experiments as stated in the paper by Epasto et. al.
        
    tmpk = min(len(objective.groundset), int(alpha*k))

    S_DistGreedy_split = Dist_Greedy(objective, tmpk, V, comm, rank, size, p_root, seed, nthreads)
    S_DistGreedy = []
    # S_DistGreedy_all = []
         
    for i in range(len(S_DistGreedy_split)):
        S_DistGreedy.extend(list(S_DistGreedy_split[i]))
        # S_DistGreedy_all.append(list(S_DistGreedy_split[i]))
    S_DistGreedy = list(np.unique(S_DistGreedy))
    p_stop_dist = MPI.Wtime()
    
    time_dist += (p_stop_dist - p_start_dist)
    
    p_start_post = MPI.Wtime()

    for j in range(int(jloop_iter)):
        currGains = parallel_margvals_returnvals_thread(objective, S, S_DistGreedy)
        EleToAdd = np.argmax(currGains)
        S = list( set().union( S, [S_DistGreedy[EleToAdd]]) );
        if(not run_Full):
            if(len(S)>=k):
                break

    p_stop = MPI.Wtime()
    time_post = (p_stop - p_start_post)
    time = (p_stop - p_start)
    valSol = objective.value( S )
    # Added increment
    queries += 1
    if rank == p_root:
        print ('BiCriteriaGreedy:', valSol, time, 'with k=', k, 'n=', len(objective.groundset), 'eps=', eps, '|S|=', len(S))
    
    return valSol, time, time_dist, time_post, S

    # print(S)





# Lazy Greedy Algorithm for ParallelAlgoGreedy (Cumuluative set C passed every run)
def lazygreedyBarbosa(objective, k, N, C, nthreads=16):
    ''' 
    Accelerated lazy greedy algorithm: for k steps (Minoux 1978).
    **NOTE** solution sets and values may be different than those found by our Greedy implementation, 
    as the two implementations break ties differently (i.e. when two elements 
    have the same marginal value,  the two implementations may not pick the same element to add)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint
    
    OUTPUTS:
    float f(L) -- the value of the solution
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time -- the processing time to optimize the function.
    list L -- the solution, where each element in the list is an element in the solution set.
    '''
    check_inputs(objective, k)
    queries = 0
    time0 = datetime.now()
    L = []
    # Cov = []
    L_rounds = [[]] # to hold L's evolution over rounds. Sometimes the new round does not add an element. 
    time_rounds = [0]
    query_rounds = [0]
    N = list( set().union( C, N));
    if(k>=len(N)):
        return N
    # On the first iteration, LazyGreedy behaves exactly like regular Greedy:
    # ele_vals = [ objective.marginalval( [elem], L ) for elem in N ]
    ele_vals = parallel_margvals_returnvals_thread(objective, L, N, nthreads)

    ele_vals_sortidx = np.argsort(ele_vals)[::-1]
    bestVal_idx = ele_vals_sortidx[0]
    L.append( N[bestVal_idx] )
    # Cov = objective.updateCov(L)
    queries += len(N)
    lazy_idx = 1

    # On remaining iterations, we update values lazily
    for i in range(1,k):
        if i%25==0:
            print('LazyGreedy round', i, 'of', k)

        # If new marginal value of best remaining ele after prev iter exceeds the *prev* marg value of 2nd best, add it
        queries += 1
        if objective.marginalval( [ N[ ele_vals_sortidx[ lazy_idx ] ] ], L ) >= ele_vals[ ele_vals_sortidx[1+lazy_idx] ]:
            #print('Found a lazy update')
            L.append( N[ele_vals_sortidx[lazy_idx]] )
            L_rounds.append([ele for ele in L])
            time_rounds.append( (datetime.now() - time0).total_seconds() )
            query_rounds.append(queries)

            lazy_idx += 1

        else:
            # If we did any lazy update iterations, we need to update bookkeeping for ground set
            N = list(set(N) - set(L)) 

            # Compute the marginal addition for each elem in N, then add the best one to solution L; 
            # Then remove it from remaning elements N
            ele_vals = [ objective.marginalval( [elem], L ) for elem in N ]
            ele_vals_sortidx = np.argsort(ele_vals)[::-1]
            bestVal_idx = ele_vals_sortidx[0]
            L.append( N[bestVal_idx] )
            #print('LG adding this value to solution:', ele_vals[bestVal_idx] )

            queries += len(N)
            L_rounds.append([ele for ele in L])
            time_rounds.append( (datetime.now() - time0).total_seconds() )
            query_rounds.append(queries)

            lazy_idx = 1

    # L_final = [objective.realset[i] for i in L]
    return L #objective.A[L,:]

def Dist_GreedyBarbosa(objective, k, V, C, comm, rank, size, p_root=0, seed=42, nthreads=16):

    check_inputs(objective, k)
    comm.barrier()
    V_split_local = V[rank] #np.array_split(V, size)[rank]
    
    # # Compute the marginal addition for each elem in N, then add the best one to solution L; remove it from remaining elements N
    # ele_A_local_vals = [ lazygreedyBarbosa(objective, k, list(V_split_local.flatten()), C, nthreads)]
    ele_A_local_vals = [ lazygreedyBarbosa(objective, k, V_split_local, C, nthreads)]
    # # Gather the partial results to all processes
    ele_vals = comm.allgather(ele_A_local_vals)
    
    return [val for sublist in ele_vals for val in sublist]

# ParallelGreedyAlg [Barbosa et al. 2016]
def ParallelAlgoGreedy(objective, k, eps, comm, rank, size, p_root=0, seed=42, nthreads=16):
    

    '''
    The parallelizable distributed greedy algorithm ParallelAlgoGreedy. Uses multiple machines to obtain solution (Algorithm 3)
    PARALLEL IMPLEMENTATION (Multithread)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    nthreads -- Number of threads to use on each machine
    
    OUTPUTS:
    list S -- the solution, where each element in the list is an element in the solution set.
    float f(S) -- the value of the solution
    time -- runtime of the algorithm
    ''' 
    comm.barrier()

    p_start = MPI.Wtime()
    
    S = []
    C = []
    C_r = []
    # random.shuffle(V)
    
    check_inputs(objective, k)
    comm.barrier()
    # V = [ ele for ele in objective.groundset ]; 
    random.seed(seed)
    for g in range(int(1/eps)): # The number of groups is stated as (1/eps) in Barbosa et al.
        
        # random.Random(seed).shuffle(V)
        for r in range(int(1/eps)): # The number of run is stated as (1/eps) in Barbosa et al.
            # random.Random(seed).shuffle(V)
            V = [[] for i in range(size)]
            for ele in objective.groundset:
                x = random.randint(0, size-1)
                V[x].append(ele)
            S_DistGreedy_split = Dist_GreedyBarbosa(objective, k, V, C, comm, rank, size, p_root=0, seed=42, nthreads=objective.nThreads)

            S_AlgSol = []
            S_AlgRel = []
            for i in range(len(S_DistGreedy_split)):
                S_AlgRel.extend(list(S_DistGreedy_split[i]))
                S_AlgSol.append(list(S_DistGreedy_split[i]))
                
            S_AlgRel = list(np.unique(S_AlgRel))
            
            S_AlgSol_split_vals = parallel_val_of_sets_thread(objective, S_AlgSol)
            
            S_p = np.argmax(S_AlgSol_split_vals)
            S_p_val = np.max(S_AlgSol_split_vals)

            if(S_p_val > objective.value(S)):
                S = list(S_AlgSol[S_p])
            
            ##added a variable to update C
            C_r = list( set().union( C_r, list(S_AlgRel)));
        
        C = list( set().union( C, list(C_r)));
        
    p_stop = MPI.Wtime()
    time = (p_stop - p_start)
    valSol =  objective.value( S )
    print ('ParallelAlgoGreedy:', valSol, time, 'with k=', k, 'n=', objective.A.shape[1], 'eps=', eps, '|S|=', len(S))
        
    return valSol, time, S #time_dist, time_post, 

# DistortedDistributed [Kazemi et al. 2021]
def DistortedDistributed(objective, k, eps, comm, rank, size, p_root=0, seed=42, nthreads=16):
    

    '''
    The parallelizable distributed greedy algorithm DistortedDistributed for submodular objectives. Uses multiple machines to obtain solution (Algorithm 3)
    PARALLEL IMPLEMENTATION (Multithread)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    nthreads -- Number of threads to use on each machine
    
    OUTPUTS:
    list S -- the solution, where each element in the list is an element in the solution set.
    float f(S) -- the value of the solution
    time -- runtime of the algorithm
    ''' 
    comm.barrier()

    p_start = MPI.Wtime()
    
    S = []
    C_r = []
    # random.shuffle(V)
    
    check_inputs(objective, k)
    comm.barrier()
    # V = [ ele for ele in objective.groundset ]; 
    random.seed(seed)
        
    # random.Random(seed).shuffle(V)
    for r in range(int(1/eps)): # The number of run is stated as (1/eps) in Barbosa et al.
        # random.Random(seed).shuffle(V)
        V = [[] for i in range(size)]
        for ele in objective.groundset:
            x = random.randint(0, size-1)
            V[x].append(ele)
        S_DistGreedy_split = Dist_GreedyBarbosa(objective, k, V, C_r, comm, rank, size, p_root=0, seed=42, nthreads=objective.nThreads)

        S_AlgSol = []
        S_AlgRel = []
        for i in range(len(S_DistGreedy_split)):
            S_AlgRel.extend(list(S_DistGreedy_split[i]))
            S_AlgSol.append(list(S_DistGreedy_split[i]))
            
        S_AlgRel = list(np.unique(S_AlgRel))
        
        S_AlgSol_split_vals = parallel_val_of_sets_thread(objective, S_AlgSol)
        
        S_p = np.argmax(S_AlgSol_split_vals)
        S_p_val = np.max(S_AlgSol_split_vals)

        if(S_p_val > objective.value(S)):
            S = list(S_AlgSol[S_p])
        
        ##added a variable to update C
        C_r = list( set().union( C_r, list(S_AlgRel)));
        
    p_stop = MPI.Wtime()
    time = (p_stop - p_start)
    valSol =  objective.value( S )
    print ('ParallelAlgoGreedy:', valSol, time, 'with k=', k, 'n=', objective.A.shape[1], 'eps=', eps, '|S|=', len(S))
        
    return valSol, time, S #time_dist, time_post, 


# Utility functions for RandGreedI_FR
def Dist_Greedy_FR_thread(objective, k, N, nthreads=16):
    '''
    Parallel-compute the marginal value f(element)-f(L) of each element in set N. Version for stochasticgreedy_parallel.
    Returns the ordered list of marginal values corresponding to the list of remaining elements N
    '''
    #Dynamically obtaining the number of threads fr parallel computation
    # nthreads_rank = multiprocessing.cpu_count()
    nthreads_obj = objective.nThreads
    N_split_local = np.array_split(N, nthreads_obj)
    #with concurrent.futures.ProcessPoolExecutor(max_workers=nthreads) as executor:
    with concurrent.futures.ThreadPoolExecutor(max_workers=nthreads_obj) as executor:
        futures = [executor.submit(lazygreedy, objective, k, split) for split in N_split_local]
        return_value = [f.result() for f in futures]
    A = []
    for i in range(len(return_value)):
        A.extend(return_value[i])
    # return return_value
    return A

    # return return_value

def Dist_Greedy_FR(objective, k, V, comm, rank, size, p_root=0, seed=42, nthreads=16):

    check_inputs(objective, k)
    comm.barrier()
    V_split_local = V[rank] # np.array_split(V, size)[rank]
    
    # # Compute the marginal addition for each elem in N, then add the best one to solution L; remove it from remaining elements N
    # ele_A_local_vals = [ lazygreedy(objective, k, list(V_split_local.flatten()), nthreads)]
    ele_A_local_vals = [ lazygreedy(objective, k, V_split_local, nthreads)]
    # # Gather the partial results to all processes
    ele_vals = comm.allgather(ele_A_local_vals)
    
    return [val for sublist in ele_vals for val in sublist]

# RandGreedI [Barbosa et al. 2015] variant that utilizes the full resources (FR) (i.e \ell = Number of Machines * Number of threads per machine)
def RandGreedI_FR(objective, k, eps, comm, rank, size, p_root=0, seed=42, nthreads=16):
    '''
    The parallelizable distributed greedy algorithm RandGreedI. Uses multiple machines to obtain solution
    PARALLEL IMPLEMENTATION (Multithread)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    nthreads -- Number of threads to use on each machine

    OUTPUTS:
    list S -- the solution, where each element in the list is an element in the solution set.
    float f(S) -- the value of the solution
    time -- runtime of the algorithm
    time_dist -- runtime spent on the distributed part of the algorithm
    time_post -- runtime spent on the post processing part of the algorithm
    ''' 

    comm.barrier()
    p_start = MPI.Wtime()
    
    S = []
    time_rounds = [0]
    query_rounds = [0]
    queries = 0
    random.seed(seed)
    # V = [ ele for ele in objective.groundset ];
    # random.Random(seed).shuffle(V)
    
    q = np.random.RandomState(42)
    V_all = [ ele for ele in objective.groundset ];
    
    V = [[] for i in range(size)]
    for ele in objective.groundset:
        x = random.randint(0, size-1)
        V[x].append(ele)
    # Get the solution of parallel QS
    p_start_dist = MPI.Wtime()
    
    S_DistGreedy_split = Dist_Greedy_FR(objective, k, V, comm, rank, size, p_root, seed, nthreads)

    S_DistGreedy = []
    S_DistGreedy_all = [] 
    for i in range(len(S_DistGreedy_split)):
        S_DistGreedy.extend(list(S_DistGreedy_split[i]))
        S_DistGreedy_all.append(list(S_DistGreedy_split[i]))
    S_DistGreedy = list(np.unique(S_DistGreedy))
    p_stop_dist = MPI.Wtime()
    
    time_dist = (p_stop_dist - p_start_dist)
    
    p_start_post = MPI.Wtime()
    # S_DistGreedy_split_vals = parallel_val_of_sets_thread(objective, S_DistGreedy_split, nthreads)
    S_DistGreedy_split_vals = parallel_val_of_sets_thread(objective, S_DistGreedy_split, nthreads=objective.nThreads)
    

    S_p = np.argmax(S_DistGreedy_split_vals)
    
    S_p_val = np.max(S_DistGreedy_split_vals)

    T = lazygreedy(objective, k, S_DistGreedy, nthreads=objective.nThreads)

    if(objective.value(T)>S_p_val):
        S = T
    else:
        S = list(S_DistGreedy_split[S_p])
    # print(S)
    p_stop = MPI.Wtime()
    time_post = (p_stop - p_start_post)
    time = (p_stop - p_start)
    valSol = objective.value( S )
    # Added increment
    queries += 1
    if rank == p_root:
        print ('RandGreedI:', valSol, time, 'with k=', k, 'n=', len(objective.groundset), 'eps=', eps, '|S|=', len(S))
    
    return valSol, time, time_dist, time_post, S, S_DistGreedy_split, time_rounds, query_rounds





#############################################################

def TG_SingleNode(objective, k, eps, V, V_all, q, C, seed=42, stop_if_approx=False, nthreads=16, alpha=0, Gamma=0):

    '''
    The algorithm ThresholdGreedy using Single Node execution for Submodular Mazimization.
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    bool stop_if_approx: determines whether we exit as soon as we find OPT that reaches approx guarantee or keep searching for better solution
    nthreads -- Number of threads to use on each machine
    
    OUTPUTS:
    list S -- the solution, where each element in the list is an element in the solution set
    ''' 
    
    if(k >= len(V)):
        return V
    
    if (alpha==0 or Gamma==0):
        alpha = 1.0 / k
        valtop = np.max( pastGains);
        Gamma = valtop
    
    S = []
    #I1 = make_I(eps, k)
    
    tau = Gamma / (alpha * np.float(k));
    # taumin = Gamma / (3.0 * np.float(k));
    taumin = (eps * Gamma) / np.float(k);
    print( "TG-- Gamma:", Gamma, "  alpha:", alpha,   "  tau:", tau, "  taumin:", taumin, "  |V|:", len(V), "  k:", k);
    
    while (tau >= taumin):
        
        for e in V:
            currGain =  objective.marginalval( [e], S )
            # queries += 1
            if currGain >= tau:
                S.append(e)

        print("TG-- For tau:", tau, "  --|S|:", len(S))
        if (len(S) >= k):
            break;
        tau = tau * (1.0 - eps)
    
    print("TG: After WHILE-- |V|:", len(V), "  |S|:", len(S), "  tau:", tau, "  k:", k )
    if(len(S)>k):
        Ap = S[len(S) - k : len(S)]
    else:
        Ap = S;
    
    return Ap

def QS_SingleNode_localMax(V_N, objective, k, eps, q, C, seed=42, nthreads=16):
    '''
    The parallelizable greedy algorithm LAS (Low-Adapive-Sequencing) using Single Node execution (OPTIMIZED IMPLEMENTATION)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k           -- the cardinality constraint (must be k>0)
    float eps       -- the error tolerance between 0 and 1
    int a           -- the max singleton assigned to every machine
    comm            -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank        -- the processor's rank (comm.Get_rank())
    int size        -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed            -- random seed to use when drawing samples
    bool stop_if_approx -- determines whether we exit as soon as we find OPT that reaches approx guarantee or keep searching for better solution

    OUTPUTS:
    float f(S)                  -- the value of the solution
    int queries                 -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time                  -- the processing time to optimize the function.
    list S                      -- the solution, where each element in the list is an element in the solution set.
    list of lists S_rounds      -- each element is a list containing the solution set S at the corresponding round.
    list of lists time_rounds   -- each element is a list containing the time at the corresponding round
    list of lists query_rounds  -- each element is a list containing the number of queries at the corresponding round
    list singletonVals          -- each element is the current marginal gain of the corresponding element in groundset
    '''  

    
    print( "\n --- Running QS on N_i of sizeN=", len(V_N)," ---"); 

    if(k >= len(V_N)):
        return V_N

    n = len(V_N)    

    # Each processor gets its own random state so they can independently draw IDENTICAL random sequences a_seq.
    #proc_random_states = [np.random.RandomState(seed) for processor in range(size)]
    randstate = np.random.RandomState(seed)
    queries = 0
    
    
    # q.shuffle(V_ground)
    currGains = parallel_margvals_returnvals_thread(objective, [], V_N, nthreads)
    
    #initialize S to max singleton
    # S = [a]
    S = [V_N[np.argmax(currGains)]];
    
    queries += len(V_N)
       
    
    V = list( set(V_N) - set(S));
    # Random Permutation
    q.shuffle(V)

    print("--QS: |V|=", len(V))

    for e in V:
        t = (1/np.float(k))*objective.value(S)
        queries += 1
        currGain =  objective.marginalval( [e], S )
        print("QS --Element:", e, "  --currGain:", currGain, "  --t:", t, "  -|S|", len(S))
        if currGain >= t:
            S.append(e)
    
    print( "\n --- Completed QS on N_i with S of sizeS=", len(S)," ---"); 

    return S

def Dist_QS_localMax(objective, k, eps, V, q, comm, rank, size, C=[], p_root=0, seed=42, nthreads=16):

    check_inputs(objective, k)
    comm.barrier()
    V_split_local = V[rank] #np.array_split(V, size)[rank]
    
    # # Compute the marginal addition for each elem in N, then add the best one to solution L; remove it from remaining elements N
    # ele_A_local_vals = [ LAG_SingleNode(objective, k, eps, list(V_split_local.flatten()), C, seed, False, nthreads)]
    ele_A_local_vals = [ QS_SingleNode_localMax(V_split_local, objective, k, eps, q, C, seed, nthreads)]
    ## return AlgRel()

    # # Gather the partial results to all processes
    ele_vals = comm.allgather(ele_A_local_vals)
    
    return [val for sublist in ele_vals for val in sublist]

#############################################################

# DQS_QS_TG (Algorithm DQS with QS in the distributed setting + (QS + TG) in post processing)

def DQS_QS_TG(objective, k, eps, comm, rank, size, p_root=0, seed=42, nthreads=16):
    '''
    The parallelizable distributed linear-time greedy algorithm D-QS. Uses multiple machines to obtain solution (Algorithm 7)
    PARALLEL IMPLEMENTATION (Multithread)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    bool stop_if_aprx: determines whether we exit as soon as we find OPT that reaches approx guarantee or keep searching for better solution
    nthreads -- Number of threads to use on each machine

    OUTPUTS:
    list S -- the solution, where each element in the list is an element in the solution set.
    float f(S) -- the value of the solution
    time -- runtime of the algorithm
    time_dist -- runtime spent on the distributed part of the algorithm
    time_post -- runtime spent on the post processing part of the algorithm
    ''' 

    comm.barrier()
    p_start = MPI.Wtime()
    
    # # Each processor gets the same random state 'q' so they can independently draw IDENTICAL random sequences .
    q = np.random.RandomState(42)

    random.seed(seed)

    S = []
    time_rounds = [0]
    query_rounds = [0]
    queries = 0
    V_all = [ ele for ele in objective.groundset ];
    
    # random.Random(seed).shuffle(V)
    V = [[] for i in range(size)]
    
    # Randomly assigning elements to all the machines 
    for ele in objective.groundset:
        
        x = random.randint(0, size-1)
        V[x].append(ele)
    
    p_start_dist = MPI.Wtime()
    S_DistGB_split = Dist_QS_localMax(objective, k, eps, V, q, comm, rank, size, [], p_root, seed, nthreads)

    S_DistGB = []
    S_DistGB_all = []
    for i in range(len(S_DistGB_split)):
        S_DistGB.extend(list(S_DistGB_split[i]))
        S_tmp = S_DistGB_split[i]
        if(len(S_tmp)>k):
            S_DistGB_all.append(list(S_tmp[len(S_tmp) - k : len(S_tmp)]))
        else:
            S_DistGB_all.append(list(S_tmp))
    S_DistGB = list(np.unique(S_DistGB))
    
    
    p_stop_dist = MPI.Wtime()
    time_dist = (p_stop_dist - p_start_dist)

    p_start_post = MPI.Wtime()

    S_DistGreedy_split_vals = parallel_val_of_sets_thread(objective, S_DistGB_all, objective.nThreads)
    
    S_star = S_DistGB_split[0] # np.argmax(S_DistGreedy_split_vals)
    
    S_star_val = objective.value(S_star) # np.max(S_DistGreedy_split_vals)

    print("DQS: --|S|:", len(S_DistGB), "  --|S_star|:", len(S_star), "  --S_star_val", S_star_val )
    ##################################
    # QS on entire returned set from all machines (S_DistGB)
    T_p = QS_SingleNode_localMax(S_DistGB, objective, k, eps, q, [], seed, nthreads=objective.nThreads)
    if(len(T_p)>k):
        T_star = T_p[len(T_p) - k : len(T_p)]
    else:
        T_star = T_p
    
    
    
    ##################################
    # p_start_post = MPI.Wtime()
    # Compute alpha = 1/(4+(...)), gamma = f(T_star) and call ThresholdGreedy on the T_p returned by QS on (S_DistGB)
    alpha = 1/2
    Gamma = objective.value(T_star)
    print("DQS: Completed QS on S --|T_p|:", len(T_p), "  --|T_star|:", len(T_star), "  --T_star_val", Gamma )

    T = TG_SingleNode(objective, k, eps, T_p, V_all, q, [], seed, stop_if_approx=False, nthreads=objective.nThreads, alpha=alpha, Gamma=Gamma)
    
    print("DQS: Completed TG on T_p --|T|:", len(T))
    ##################################
    # S = argmax (S_star, T_star, T)
    if(objective.value(T) >= objective.value(T_star) and objective.value(T) >= S_star_val):
        S = T
    elif(objective.value(T_star) >= objective.value(T) and objective.value(T_star) >= S_star_val):
        S = T_star
    else:
        S = list(S_DistGB_all[S_star])
    
    # print(S)
    p_stop = MPI.Wtime()
    time_post = (p_stop - p_start_post)
    time = (p_stop - p_start)
    valSol = objective.value( S )

    ##################################
    
    if rank == p_root:
        print ('DQS-QS-TG:', valSol, time, 'with k=', k, 'n=', len(objective.groundset), 'eps=', eps, '|S|=', len(S))
    
    return valSol, time, time_dist, time_post, S

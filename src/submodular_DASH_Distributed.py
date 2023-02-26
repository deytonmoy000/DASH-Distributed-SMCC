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
import multiprocessing
# os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from datetime import datetime
import numpy as np
import random
from queue import PriorityQueue
from scipy import sparse
import pandas as pd

try:
    from mpi4py import MPI
except ImportError:
    MPI = None 
if not MPI:
    print("MPI not loaded from the mpi4py package. Serial implementations will function, \
            but parallel implementations will not function.")

# # Load the Objective function classes 
from objectives import  NetCoverSparse,\
                        RevenueMaxOnNetSparse,\
                        InfluenceMaxSparse,\
                        ImageSummarizationMonotone


def check_inputs(objective, k):
    '''
    Function to run basic tests on the inputs of one of our optimization functions:
    '''
    # objective class contains the ground set and also value, marginalval methods
    assert( hasattr(objective, 'groundset') )
    assert( hasattr(objective, 'value') )
    assert( hasattr(objective, 'marginalval') )
    # k is greater than 0
    assert( k>0 )
    # k is smaller than the number of elements in the ground set
    assert( k<len(objective.groundset) )
    # the ground set contains all integers from 0 to the max integer in the set
    assert( np.array_equal(objective.groundset, list(range(np.max(objective.groundset)+1) )) )

def sample_seq( X, k, randstate ):
    if len(X) <= k:
        randstate.shuffle(X)
        return X
    Y = list(randstate.choice(X, k, replace=False));
    randstate.shuffle(Y);
    return Y;

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

def parallel_margvals_returnvals_thread(objective, L, N, nthreads=1):
    '''
    Parallel-compute the marginal value f(element)-f(L) of each element in set N. Version for stochasticgreedy_parallel.
    Returns the ordered list of marginal values corresponding to the list of remaining elements N
    '''
    #Dynamically obtaining the number of threads fr parallel computation
    # nthreads_rank = multiprocessing.cpu_count()
    nthreads_rank = objective.nThreads
    N_split_local = np.array_split(N, nthreads_rank)
    #with concurrent.futures.ProcessPoolExecutor(max_workers=nthreads) as executor:
    with concurrent.futures.ThreadPoolExecutor(max_workers=nthreads_rank) as executor:
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

    ele_vals_local_vals = parallel_margvals_returnvals_thread(objective, L, list(N_split_local), nthreads=1)
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

def parallel_val_of_sets_thread(objective, list_of_sets, nthreads=1):
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
    ele_vals_local_vals = parallel_val_of_sets_thread(objective, list(N_split_local), nthreads=1)
    # Gather the partial results to all processes
    ele_vals = comm.allgather(ele_vals_local_vals)
    return [val for sublist in ele_vals for val in sublist]

    # return ele_vals

def create_objective_dist(objective, M):
    if(objective.name=="NetCov"):
        A = sparse.vstack(objective.A, M)
        objective_new = NetCoverSparse.NetCoverSparse(A, objective.nThreads)
    elif(objective.name=="InfMax"):
        A = sparse.vstack(objective.A, M)
        objective_new = InfluenceMaxSparse.InfluenceMaxSparse(A, objective.p, objective.nThreads)
    elif(objective.name=="RevMax"):
        A = sparse.vstack(objective.A, M)
        objective_new = RevenueMaxOnNetSparse.RevenueMaxOnNetSparse(A, objective.alpha, objective.nThreads)
    elif(objective.name=="ImgSum"):
        A = sparse.vstack(objective.A, M)
        A = A.todense()
        objective_new = ImageSummarizationMonotone.ImageSummarizationMonotone(A, objective.nThreads)
    elif(objective.name=="TwtSum"):
        A = sparse.vstack(objective.A, M)
        objective_new = TwitterStreamSummarizationMonotone.TwitterStreamSummarizationMonotone(A, objective.nThreads)
    elif(objective.name=="CalTrf"):
        A = sparse.vstack(objective.A, M)
        objective_new = TrafficCoverDirWeightedSparse.TrafficCoverDirWeightedSparse(A, objective.nThreads)
    elif(objective.name=="MovRec"):
        A = sparse.vstack(objective.A, M)
        objective_new = MovieRecommendationFahrbach.MovieRecommendationFahrbach(A, objective.lmbda, objective.nThreads)

    return objective_new

def create_objective(objective, A):
    if(objective.name=="NetCov"):
        objective_new = NetCoverSparse.NetCoverSparse(A, objective.nThreads)
    elif(objective.name=="InfMax"):
        objective_new = InfluenceMaxSparse.InfluenceMaxSparse(A, objective.p, objective.nThreads)
    elif(objective.name=="RevMax"):
        objective_new = RevenueMaxOnNetSparse.RevenueMaxOnNetSparse(A, objective.alpha, objective.nThreads)
    elif(objective.name=="ImgSum"):
        objective_new = ImageSummarizationMonotone.ImageSummarizationMonotone(A, objective.nThreads)
    elif(objective.name=="TwtSum"):
        objective_new = TwitterStreamSummarizationMonotone.TwitterStreamSummarizationMonotone(A, objective.nThreads)
    elif(objective.name=="CalTrf"):
        objective_new = TrafficCoverDirWeightedSparse.TrafficCoverDirWeightedSparse(A, objective.nThreads)
    elif(objective.name=="MovRec"):
        objective_new = MovieRecommendationFahrbach.MovieRecommendationFahrbach(A, objective.lmbda, objective.nThreads)

    return objective_new

def create_objective_new(obj, nthreads, comm, rank, size, exp_string):
    path=exp_string+"_"+str(size)+".npz"
    A = sparse.load_npz(path)
    if(obj=="ER"):
        objective = NetCoverSparse.NetCoverSparse(A, nthreads)

    elif(obj=="BA"):
        objective = NetCoverSparse.NetCoverSparse(A, nthreads)

    elif(obj=="WS"):
        objective = NetCoverSparse.NetCoverSparse(A, nthreads)
    
    elif(obj=="IFM"):
        p = 0.01
        objective = InfluenceMaxSparse.InfluenceMaxSparse(A, p, nthreads)
    
    elif(obj=="RVM"):
        alpha = 0.3
        objective = RevenueMaxOnNetSparse.RevenueMaxOnNetSparse(A, alpha, nthreads)
    elif(obj=="IFM2"):
        p = 0.01
        objective = InfluenceMaxSparse.InfluenceMaxSparse(A, p, nthreads)
    
    elif(obj=="RVM2"):
        alpha = 0.3
        objective = RevenueMaxOnNetSparse.RevenueMaxOnNetSparse(A, alpha, nthreads)

    elif(obj=="IS"):
        Sim = A.todense()
        objective = ImageSummarizationMonotone.ImageSummarizationMonotone(Sim, nthreads)

    elif(obj=="MVR"):
        lmbda = 0.95
        objective = MovieRecommendationFahrbach.MovieRecommendationFahrbach(A, lmbda, nthreads)

    comm.barrier()

    if rank == 0:
        print( obj,' Objective initialized. Beginning tests.' )
    
    return objective




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





# DASH
def LAT_SingleNode(V, S, objective, tau, eps, delta, k, randstate, pastGains):
    
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
    
    # comm.barrier()

    ell = int(np.ceil( (4.0 + 8.0/ eps ) * np.log( len(V) / delta ) ) ) + 1;
    V = list( np.sort( list( set(V)-set(S) ) ) );
    queries = 0
    for itr in range( ell+1):
        s = np.min( [k - len(S), len(V) ] );
        
        seq = sample_seq( V, s, randstate );

        lmbda = make_lmbda( eps, s, s );
        
        # B = parallel_adaptiveAdd(lmbda, seq, S, objective, eps, k, comm, rank, size, tau)
        B = parallel_adaptiveAdd_thread(lmbda, seq, S, objective, eps, k, tau)
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
        
        gains = parallel_margvals_returnvals_thread(objective, S, V)
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
        print( "LAT has failed. This should be an extremely rare event. Terminating program..." );
        exit(1);
            
    return [pastGains, S, queries];

def LAG_SingleNode(objective, k, eps, seed=42, stop_if_approx=False, return_rows=True, nthreads=1, is_LAG=False):

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
    
    # comm.barrier()
    # p_start = MPI.Wtime()
    
    time = 0
    V = [ ele for ele in objective.groundset ];
    pastGains = parallel_margvals_returnvals_thread(objective, [], V, nthreads=1)

    alpha = 1.0 / k
    valtop = np.max( pastGains);
    Gamma = valtop
    tau = Gamma / (alpha * np.float(k));
    taumin = Gamma / (3.0 * np.float(k));
    
    
    # Each processor gets its own random state so they can independently draw IDENTICAL random sequences a_seq.
    #proc_random_states = [np.random.RandomState(seed) for processor in range(size)]
    randstate = np.random.RandomState(seed)
    S = []
    
    while (tau > taumin):
        tau = np.min( [tau, np.max( pastGains ) * (1.0 - eps)] );
        if (tau <= 0.0):
            tau = taumin;
        V_above_thresh = np.where(pastGains >= tau)[0]
        V_above_thresh = [ V[ele] for ele in V_above_thresh ];
        V_above_thresh = list( set(V_above_thresh) - set(S) );
        
        currGains = parallel_margvals_returnvals_thread(objective, S, V_above_thresh)
        #Added query increment
        for ps in range( len(V_above_thresh )):
            pastGains[ V_above_thresh[ps]] = currGains[ ps ];
        V_above_thresh_id = np.where(currGains >= tau)[0]
        V_above_thresh = [ V_above_thresh[ele] for ele in V_above_thresh_id ];
        
        if (len(V_above_thresh) > 0):
            [pastGains, S, queries_tmp] = LAT_SingleNode( V_above_thresh, S, objective, tau, eps / np.float(3), 1.0 / ((np.log( alpha / 3 ) / np.log( 1 - eps ) ) + 1) , k, randstate, pastGains );
            #Added query increment
            for ele in S:
                pastGains[ele] = 0;

        if (len(S) >= k):
            break;
        if stop_if_approx:
            if objective.value(S) >= stopRatio:
                return objective.A[S,:]
    
    valSol = objective.value( S )
    # Added increment
    # if rank == p_root:
    # print ('ABR:', valSol, queries, time, 'with k=', k, 'n=', len(objective.groundset), 'eps=', eps, '|S|=', len(S))
    
    # return valSol, queries, time, S, S_rounds, time_rounds, query_rounds
    if(return_rows):
        if(objective.name=="ImgSum"):
            sA = sparse.csr_matrix(objective.A[S,:])
            return sA
        else:
            return objective.A[S,:]
    else:
        return S

def Dist_LAG(objective, k, eps, comm, rank, size, p_root=0, seed=42, stop_if_apx=False, nthreads=1):

    check_inputs(objective, k)
    comm.barrier()
        
    # # Compute the marginal addition for each elem in N, then add the best one to solution L; remove it from remaining elements N
    ele_A_local_vals = [ LAG_SingleNode(objective, k, eps, seed, False, True, 1, True)]
    
    if(objective.name=="MovRec"):
        if(rank != 0):
            X = ele_A_local_vals[0]
            comm.send(X, dest=0)
        shapeX = int(np.ceil(objective.A.shape[0]))
        shapeY = int(np.ceil(objective.A.shape[1]))
        shape = (shapeX, shapeY)

        ele_A_local_vals_tmp = np.empty(shape, dtype = 'd')
        if (rank == 0):
            ele_vals_tmp = [ele_A_local_vals[0]]
            for i in range(1, size):
                ele_A_local_vals_tmp = comm.recv(source=i)
                ele_vals_tmp.append(ele_A_local_vals_tmp)
                
        if rank != 0:
            ele_vals_tmp = list([0]*size)
        ele_vals = []
        for i in range(size):
            ele_vals_i = comm.bcast(ele_vals_tmp[i], root=0)
            ele_vals.append(ele_vals_i)
    else:
        ele_vals = comm.allgather(ele_A_local_vals)
   
    return ele_vals

def DASH(obj, exp_string, k, eps, comm, rank, size, p_root=0, seed=42, stop_if_aprx=False, nthreads=1):
    '''
    The parallelizable distributed greedy algorithm DASH. Uses multiple machines to obtain solution (Algorithm 1)
    PARALLEL IMPLEMENTATION (Multithread)
    
    INPUTS:
    string obj -- contains the objective abbreviation
    string exp_string -- contains the idstributed filename stored locally after splitting
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
    
    
    S = []
    S_rounds = []
    time_rounds = [0]
    query_rounds = [0]
    queries = 0
    objective = create_objective_new(obj, nthreads, comm, rank, size, exp_string)
    p_start = MPI.Wtime()
    p_start_dist = MPI.Wtime()
    # ele_vals = Dist_LS(objective, k, eps, comm, rank, size, p_root, seed, False, nthreads)
    ele_vals = Dist_LAG(objective, k, eps, comm, rank, size, p_root, seed, False, nthreads)
    
    p_stop_dist = MPI.Wtime()
    time_dist = (p_stop_dist - p_start_dist)

    p_start_post = MPI.Wtime()


    if(rank ==0):
        ele_vals_all = [val for sublist in ele_vals for val in sublist]
        if(objective.name=="ImgSum"):
            S_DistGB_rows = sparse.vstack(ele_vals_all)
            S_DistGB_rows = S_DistGB_rows.todense()
        else:
            S_DistGB_rows = sparse.vstack(ele_vals_all)
        objective_Root = create_objective(objective, S_DistGB_rows)
        objective = []

        Sets_all = []
        ele_min = 0
        ele_max = 0

        for i in range(len(ele_vals_all)):
            ele_max += ele_vals_all[i].shape[0]
            if(ele_vals_all[i].shape[0] > k):
                ele_min += ele_vals_all[i].shape[0] - k
            else:
                ele_min += 0
            Sets_all.append([ele for ele in range(ele_min, ele_max)])
            ele_min = ele_max
        Sets_values = parallel_val_of_sets_thread(objective_Root, Sets_all)
        Sets_argmax = np.argmax(Sets_values)

        T = LAG_SingleNode(objective_Root, k, eps, seed, stop_if_aprx, return_rows=False, nthreads=1, is_LAG=True)
    
        if(objective_Root.value( T ) > objective_Root.value(Sets_all[Sets_argmax])):
            S = T
        else:
            S = Sets_all[Sets_argmax]

        p_stop = MPI.Wtime()
        time_post = (p_stop - p_start_post)
        time = (p_stop - p_start)
        valSol = objective_Root.value( S )
        # Added increment
        queries += 1
        
        output = pd.DataFrame({
                            'f_of_S'          :       [valSol], \
                            'queries'         :       [queries], \
                            'Time'            :       [time], \
                            'TimeDist'        :       [time_dist], \
                            'TimePost'        :       [time_post], \
                            'S'               :       [S]
                            })
    # output = output.transpose()
        return output





# LS+PGB [Chen et al. 2021] Heuristic for Disctributed Data

def parallel_threshold_sample_SingleNode(V, S, objective, tau, eps, delta, k, randstate, pastGains):
    
    '''
    The parallelizable greedy algorithm THRESHOLDSEQ using Single Node execution for Fixed Threshold 'tau'
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
    
    # comm.barrier()

    ell = int(np.ceil( (4.0 + 8.0/ eps ) * np.log( len(V) / delta ) ) ) + 1;
    V = list( np.sort( list( set(V)-set(S) ) ) );
    queries = 0
    for itr in range( ell ):
        s = np.min( [k - len(S), len(V) ] );
        
        seq = sample_seq( V, s, randstate );

        lmbda = make_lmbda( eps, s, s );
        
        # B = parallel_adaptiveAdd(lmbda, seq, S, objective, eps, k, comm, rank, size, tau)
        B = parallel_adaptiveAdd_thread(lmbda, seq, S, objective, eps, k, tau)
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
        
        gains = parallel_margvals_returnvals_thread(objective, S, V)
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

def LinearSeq(objective, k, eps, p_root=0, seed=42, stop_if_approx=True):
    '''
    The algorithm ParallelLinearSeq using Single Node execution for Submodular Mazimization.
    PARALLEL IMPLEMENTATION
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    list N -- the groundset to process
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    bool stop_if_approx: determines whether we exit as soon as we find OPT that reaches approx guarantee or keep searching for better solution
    nthreads -- Number of threads to use on the machine

    OUTPUTS:
    list S -- the solution, where each element in the list is an element in the solution set.
    '''    

    n = len(objective.groundset)

    # Each processor gets its own random state so they can independently draw IDENTICAL random sequences a_seq.
    #proc_random_states = [np.random.RandomState(seed) for processor in range(size)]
    randstate = np.random.RandomState(seed)
    queries = 0
    
    S = []
    S_rounds = [[]]
    time_rounds = [0]
    query_rounds = [0]
    
    currGains = parallel_margvals_returnvals_thread(objective, [], [ele for ele in objective.groundset])
    #Added query increment
    queries += len(objective.groundset)
    #currGains = parallel_margvals_returnvals(objective, [], [ele for ele in objective.groundset], comm, rank, size)
    time = 0
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
        
        #do lazy discard first, based on last round's computations
        V_above_thresh = np.where(currGains >= t)[0]
        V = list( set([ V[idx] for idx in V_above_thresh ] ));

        currGains = parallel_margvals_returnvals_thread(objective, S, [ele for ele in V])
        #Added query increment
        queries += len(V)
        V_above_thresh = np.where(currGains >= t)[0]
        V = [ V[idx] for idx in V_above_thresh ];
        currGains = [ currGains[idx] for idx in V_above_thresh ];
        

        if (len(V) == 0):
            break;
        
        # Radnom Permutation
        randstate.shuffle(V);
        lmbda = make_lmbda( eps, k, len(V) );

        B = parallel_adaptiveAdd_thread(lmbda, V, S, objective, eps, k)
        
        
        queries += len( lmbda );
        query_rounds.append(queries)
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
        # S= list(set().union(S, T))
        S.extend(list(T))
        indexes = np.unique(S, return_index=True)[1]
        S = [S[index] for index in sorted(indexes)]
        S_rounds.append([ele for ele in S])
        query_rounds.append(queries)
        
        
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
            if (t >= 0.5 * valtopk / np.float(k)):
                if(len(S)>k):
                    Ap = S[len(S) - k : len(S)]
                else:
                    Ap = S;
                valAp = objective.value(Ap)
                queries += 1
                print ('FLS:', valAp, queries, time, 'with k=', k, 'n=', n, 'eps=', eps, '|S|=', len(Ap))
    
                return valAp, queries, time, S, S_rounds, time_rounds, query_rounds, singletonVals
       
        V_above_thresh = np.where(currGains >= t)[0]
        V = list( set([ V[idx] for idx in V_above_thresh ] ));

        currGains = parallel_margvals_returnvals_thread(objective, S, [ele for ele in V])
        #Added query increment
        queries += len(V)
        V_above_thresh = np.where(currGains >= t)[0]
        V = [ V[idx] for idx in V_above_thresh ];
        currGains = [ currGains[idx] for idx in V_above_thresh ];

        if (len(V) == 0):
            break;
        
        randstate.shuffle(V);
        lmbda = make_lmbda( eps, k, len(V) );

        B = parallel_adaptiveAdd_thread(lmbda, V, S, objective, eps, k)
        
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
        # S= list(set().union(S, T))
        S.extend(list(T))
        indexes = np.unique(S, return_index=True)[1]
        S = [S[index] for index in sorted(indexes)]
        S_rounds.append([ele for ele in S])
        query_rounds.append(queries)        
    
    if(len(S)>k):
        Ap = S[len(S) - k : len(S)]
    else:
        Ap = S;
    valAp = objective.value(Ap)
    queries += 1

    # if rank == p_root:
    print ('FLS:', valAp, queries, time, 'with k=', k, 'n=', n, 'eps=', eps, '|S|=', len(Ap))
    
    return valAp, queries, time, S, S_rounds, time_rounds, query_rounds, singletonVals

def ParallelGreedyBoost_SingleNode(objective, k, eps, seed=42, stop_if_approx=False, return_rows=True, nthreads=1, is_LAG=False):

    '''
    The algorithm ParallelGreedyBoost using Single Node execution for Submodular Mazimization.
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
    
    # comm.barrier()
    # p_start = MPI.Wtime()
    eps_FLS = 0.21
    time = 0
    V = [ ele for ele in objective.groundset ];
    if(is_LAG):
        pastGains = parallel_margvals_returnvals_thread(objective, [], V, nthreads=1)
    
        alpha = 1.0 / k
        valtop = np.max( pastGains);
        Gamma = valtop
        tau = Gamma / (alpha * np.float(k));
        taumin = Gamma / (3.0 * np.float(k));
    else:
        # Gamma, queries, time, sol, sol_r, time_r, queries_r, pastGains = LinearSeq(objective, k, eps_FLS, p_root, seed, True );
        Gamma, queries, time, sol, sol_r, time_r, queries_r, pastGains = ParallelLinearSeq_SingleNode(objective, k, eps_FLS, seed, False, True, 1);
        
        #alpha = 1.0 / (4.0 + (2*(3+(2*eps_FLS)))/(1 - (2*eps_FLS) - (2*(eps_FLS ** 2)))*eps_FLS);
        alpha = 1.0 / (4.0 + (4*(2-eps_FLS))/((1 - eps_FLS)*(1-(2*eps_FLS))) *eps_FLS);
        valtopk = np.sum( np.sort(pastGains)[-k:] );
        stopRatio = (1.0 - 1.0/np.exp(1) - eps)*valtopk;

        tau = Gamma / (alpha * np.float(k));
        taumin = Gamma / (3.0 * np.float(k));
        if (tau > valtopk / np.float(k)):
            tau = valtopk / np.float(k);
    
    # Each processor gets its own random state so they can independently draw IDENTICAL random sequences a_seq.
    #proc_random_states = [np.random.RandomState(seed) for processor in range(size)]
    randstate = np.random.RandomState(seed)
    
    
    S = []
        
    
    
    while (tau > taumin):
        tau = np.min( [tau, np.max( pastGains ) * (1.0 - eps)] );
        if (tau <= 0.0):
            tau = taumin;
        V_above_thresh = np.where(pastGains >= tau)[0]
        V_above_thresh = [ V[ele] for ele in V_above_thresh ];
        V_above_thresh = list( set(V_above_thresh) - set(S) );
        
        currGains = parallel_margvals_returnvals_thread(objective, S, V_above_thresh)
        #Added query increment
        for ps in range( len(V_above_thresh )):
            pastGains[ V_above_thresh[ps]] = currGains[ ps ];
        V_above_thresh_id = np.where(currGains >= tau)[0]
        V_above_thresh = [ V_above_thresh[ele] for ele in V_above_thresh_id ];
        
        if (len(V_above_thresh) > 0):
            [pastGains, S, queries_tmp] = parallel_threshold_sample_SingleNode( V_above_thresh, S, objective, tau, eps / np.float(3), 1.0 / ((np.log( alpha / 3 ) / np.log( 1 - eps ) ) + 1) , k, randstate, pastGains );
            #Added query increment
            for ele in S:
                pastGains[ele] = 0;

        if (len(S) >= k):
            break;
        if stop_if_approx:
            if objective.value(S) >= stopRatio:
                return objective.A[S,:]
    
    valSol = objective.value( S )
    # Added increment
    # if rank == p_root:
    # print ('ABR:', valSol, queries, time, 'with k=', k, 'n=', len(objective.groundset), 'eps=', eps, '|S|=', len(S))
    
    # return valSol, queries, time, S, S_rounds, time_rounds, query_rounds
    if(return_rows):
        if(objective.name=="ImgSum"):
            sA = sparse.csr_matrix(objective.A[S,:])
            return sA
        else:
            return objective.A[S,:]
    else:
        return S

def Dist_PGB(objective, k, eps, comm, rank, size, p_root=0, seed=42, stop_if_apx=False, nthreads=1):

    check_inputs(objective, k)
    comm.barrier()
    
    # # Compute the marginal addition for each elem in N, then add the best one to solution L; remove it from remaining elements N
    ele_A_local_vals = [ ParallelGreedyBoost_SingleNode(objective, k, eps, seed, False, return_rows=True)]
    # print(rank, ele_A_local_vals[0].shape)
    if(objective.name=="MovRec"):
        if(rank != 0):
            X = ele_A_local_vals
            comm.send(X, dest=0)
        shapeX = int(np.ceil(objective.A.shape[0]))
        shapeY = int(np.ceil(objective.A.shape[1]))
        shape = (shapeX, shapeY)

        ele_A_local_vals_tmp = np.empty(shape, dtype = 'd')
        if (rank == 0):
            ele_vals_tmp = [ele_A_local_vals]
            for i in range(1, size):
                ele_A_local_vals_tmp = comm.recv(source=i)
                ele_vals_tmp.append(ele_A_local_vals_tmp)
                
        if rank != 0:
            ele_vals_tmp = list([0]*size)
        ele_vals = []
        for i in range(size):
            ele_vals_i = comm.bcast(ele_vals_tmp[i], root=0)
            ele_vals.append(ele_vals_i)
    else:
        ele_vals = comm.allgather(ele_A_local_vals)

    return ele_vals

def ParallelGreedyBoostDistributed(obj, exp_string, k, eps, comm, rank, size, p_root=0, seed=42, stop_if_aprx=False, nthreads=1):
    
    '''
    A distributed variant (heuristic) for the parallelizable greedy algorithm PARALLELGREEDYBOOST to Boost to the Optimal Ratio.
    PARALLEL IMPLEMENTATION
    
    INPUTS:
    string obj -- contains the objective abbreviation
    string exp_string -- contains the idstributed filename stored locally after splitting
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

    objective = create_objective_new(obj, nthreads, comm, rank, size, exp_string)
    p_start = MPI.Wtime()
    
    # # Each processor gets its own random state so they can independently draw IDENTICAL random sequences a_seq.
    # #proc_random_states = [np.random.RandomState(seed) for processor in range(size)]
    # randstate = np.random.RandomState(seed)
    # time0 = datetime.now()
    S = []
    time_rounds = [0]
    query_rounds = [0]
    queries = 0

    # Get the solution of parallel QS
    p_start_dist = MPI.Wtime()
    S_rows_split = Dist_PGB(objective, k, eps, comm, rank, size, p_root, seed, stop_if_aprx, nthreads)

    p_stop_dist = MPI.Wtime()
    time_dist = (p_stop_dist - p_start_dist)

    p_start_post = MPI.Wtime()

    if(rank ==0):
        S_rows_split_all = [val for sublist in S_rows_split for val in sublist]
        if(objective.name=="ImgSum"):
            A = sparse.vstack(S_rows_split_all)
            A = A.todense()
        else:
            A = sparse.vstack(S_rows_split_all)
        Sets_all = []
        ele_min = 0
        ele_max = 0

        for i in range(len(S_rows_split_all)):
            ele_max += S_rows_split_all[i].shape[0]
            Sets_all.append([ele for ele in range(ele_min, ele_max)])
            ele_min += S_rows_split_all[i].shape[0]
        objective_Root = create_objective(objective, A)
        objective = []
        Sets_values = parallel_val_of_sets_thread(objective_Root, Sets_all)
        Sets_argmax = np.argmax(Sets_values)

        p_stop = MPI.Wtime()
        time_post = (p_stop - p_start_post)
        time = (p_stop - p_start)
        S = Sets_all[Sets_argmax] 
        valSol = objective_Root.value( S )
        
        queries += 1
        
        output = pd.DataFrame({
                            'f_of_S'          :       [valSol], \
                            'queries'         :       [queries], \
                            'Time'            :       [time], \
                            'TimeDist'        :       [time_dist], \
                            'TimePost'        :       [time_post], \
                            'S'               :       [S]
                            })
        return output





# LazyGreedy Algorthm
def lazygreedy_MultiNode(objective, k, comm, rank, size, nthreads=1):
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
    N = [ele for ele in objective.groundset]
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

def lazygreedy(objective, k, nthreads=1, return_rows=True):
    ''' 
    Accelerated lazy greedy algorithm: for k steps (Minoux 1978).
    **NOTE** solution sets and values may be different than those found by our Greedy implementation, 
    as the two implementations break ties differently (i.e. when two elements 
    have the same marginal value,  the two implementations may not pick the same element to add)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint
    
    OUTPUTS:
    list L -- the solution (matrix rows), where each element in the list is an element in the solution set.
    '''
    check_inputs(objective, k)
    queries = 0
    time0 = datetime.now()
    L = []
    # Cov = []
    L_rounds = [[]] # to hold L's evolution over rounds. Sometimes the new round does not add an element. 
    time_rounds = [0]
    query_rounds = [0]
    N = [ ele for ele in objective.groundset ]    
    n = len(N)
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
        # if i%25==0:
        #     print('LazyGreedy round', i, 'of', k)

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

    # return L
    # return objective.A[L,:]
    if(return_rows):
        if(objective.name=="ImgSum"):
            sA = sparse.csr_matrix(objective.A[L,:])
            return sA
        elif(objective.name=="MovRec"):
            if(len(L)>k):
                Ap = L[len(L) - k : len(L)]
            else:
                Ap = L;
            return objective.A[Ap,:]
        else:
            return objective.A[L,:]
    else:
        return L

def Dist_Greedy(objective, k, comm, rank, size, p_root=0, seed=42, nthreads=1):

    check_inputs(objective, k)
    comm.barrier()
        
    # # Compute the marginal addition for each elem in N, then add the best one to solution L; remove it from remaining elements N
    ele_A_local_vals = [ lazygreedy(objective, k,  nthreads)]
    
    if(objective.name=="MovRec"):
        if(rank != 0):
            X = ele_A_local_vals[0]
            comm.send(X, dest=0)
        shapeX = int(np.ceil(objective.A.shape[0]))
        shapeY = int(np.ceil(objective.A.shape[1]))
        shape = (shapeX, shapeY)

        ele_A_local_vals_tmp = np.empty(shape, dtype = 'd')
        if (rank == 0):
            ele_vals_tmp = [ele_A_local_vals[0]]
            for i in range(1, size):
                ele_A_local_vals_tmp = comm.recv(source=i)
                ele_vals_tmp.append(ele_A_local_vals_tmp)
                
        if rank != 0:
            ele_vals_tmp = list([0]*size)
        ele_vals = []
        for i in range(size):
            ele_vals_i = comm.bcast(ele_vals_tmp[i], root=0)
            ele_vals.append(ele_vals_i)
    else:
        ele_vals = comm.allgather(ele_A_local_vals)
   
    return ele_vals

# RandGreedI [Barbosa et al. 2015]
def RandGreedI(obj, exp_string, k, eps, comm, rank, size, p_root=0, seed=42, nthreads=1):
    '''
    The parallelizable distributed greedy algorithm RandGreedI. Uses multiple machines to obtain solution
    PARALLEL IMPLEMENTATION (Multithread)
    
    INPUTS:
    string obj -- contains the objective abbreviation
    string exp_string -- contains the idstributed filename stored locally after splitting
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

    objective = create_objective_new(obj, nthreads, comm, rank, size, exp_string)
    p_start = MPI.Wtime()
    
    S = []
    queries = 0

    # Get the solution of parallel QS
    p_start_dist = MPI.Wtime()
    
    ele_vals = Dist_Greedy(objective, k, comm, rank, size, p_root, seed, nthreads)

    p_stop_dist = MPI.Wtime()
    time_dist = (p_stop_dist - p_start_dist)

    p_start_post = MPI.Wtime()

    if(rank ==0):
        S_rows_split_all = [val for sublist in ele_vals for val in sublist]
        if(objective.name=="ImgSum"):
            A = sparse.vstack(S_rows_split_all)
            A = A.todense()
        else:
            A = sparse.vstack(S_rows_split_all)
        Sets_all = []
        ele_min = 0
        ele_max = 0

        for i in range(len(S_rows_split_all)):
            ele_max += S_rows_split_all[i].shape[0]
            Sets_all.append([ele for ele in range(ele_min, ele_max)])
            ele_min += S_rows_split_all[i].shape[0]
        objective_Root = create_objective(objective, A)
        objective = []
        Sets_values = parallel_val_of_sets_thread(objective_Root, Sets_all)
        Sets_argmax = np.argmax(Sets_values)

        # T = LAG_SingleNode(objective_Root, k, eps, seed, False, return_rows=False)
        T = lazygreedy(objective_Root, k, objective_Root.nThreads, False)
        if(objective_Root.value( T ) > objective_Root.value(Sets_all[Sets_argmax])):
            S = T
        else:
            S = Sets_all[Sets_argmax] 
        
        p_stop = MPI.Wtime()
        time_post = (p_stop - p_start_post)
        time = (p_stop - p_start)
        valSol = objective_Root.value( S )
        
        queries += 1
        
        output = pd.DataFrame({
                            'f_of_S'          :       [valSol], \
                            'Time'            :       [time], \
                            'TimeDist'        :       [time_dist], \
                            'TimePost'        :       [time_post], \
                            'S'               :       [S]
                            })
        return output

# RandGreedI using LAG as ALG on the primary machine (Appendix Results)
def RandGreedI_LAG(obj, exp_string, k, eps, comm, rank, size, p_root=0, seed=42, nthreads=1):
    '''
    The parallelizable distributed greedy algorithm RandGreedI. Uses multiple machines to obtain solution
    PARALLEL IMPLEMENTATION (Multithread)
    
    INPUTS:
    string obj -- contains the objective abbreviation
    string exp_string -- contains the idstributed filename stored locally after splitting
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

    objective = create_objective_new(obj, nthreads, comm, rank, size, exp_string)
    p_start = MPI.Wtime()
    
    S = []
    queries = 0

    # Get the solution of parallel QS
    p_start_dist = MPI.Wtime()
    
    ele_vals = Dist_Greedy(objective, k, comm, rank, size, p_root, seed, nthreads)

    p_stop_dist = MPI.Wtime()
    time_dist = (p_stop_dist - p_start_dist)

    p_start_post = MPI.Wtime()

    if(rank ==0):
        S_rows_split_all = [val for sublist in ele_vals for val in sublist]
        if(objective.name=="ImgSum"):
            A = sparse.vstack(S_rows_split_all)
            A = A.todense()
        else:
            A = sparse.vstack(S_rows_split_all)
        Sets_all = []
        ele_min = 0
        ele_max = 0

        for i in range(len(S_rows_split_all)):
            ele_max += S_rows_split_all[i].shape[0]
            Sets_all.append([ele for ele in range(ele_min, ele_max)])
            ele_min += S_rows_split_all[i].shape[0]
        objective_Root = create_objective(objective, A)
        objective = []
        Sets_values = parallel_val_of_sets_thread(objective_Root, Sets_all)
        Sets_argmax = np.argmax(Sets_values)

        T = LAG_SingleNode(objective_Root, k, eps, seed, stop_if_approx=False, return_rows=False, nthreads=objective_Root.nThreads, is_LAG=True)
        
        if(objective_Root.value( T ) > objective_Root.value(Sets_all[Sets_argmax])):
            S = T
        else:
            S = Sets_all[Sets_argmax] 
        
        p_stop = MPI.Wtime()
        time_post = (p_stop - p_start_post)
        time = (p_stop - p_start)
        valSol = objective_Root.value( S )
        
        queries += 1
        
        output = pd.DataFrame({
                            'f_of_S'          :       [valSol], \
                            'Time'            :       [time], \
                            'TimeDist'        :       [time_dist], \
                            'TimePost'        :       [time_post], \
                            'S'               :       [S]
                            })
        return output

# BicriteriaGreedy [Epasto et al. 2017]
def BiCriteriaGreedy(obj, exp_string, k, eps, comm, rank, size, p_root=0, seed=42, nthreads=1, run_Full=False):
    '''
    The parallelizable distributed greedy algorithm BiCriteriaGreedy. Uses multiple machines to obtain solution
    PARALLEL IMPLEMENTATION (Multithread)
    
    INPUTS:
    string obj -- contains the objective abbreviation
    string exp_string -- contains the idstributed filename stored locally after splitting
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
    objective = create_objective_new(obj, nthreads, comm, rank, size, exp_string)
    p_start = MPI.Wtime()
    rounds = 1 # As used in experiments by Epasto et. al. 
    alpha = 3/(pow(eps,(1/rounds)))
    jloop_iter = (pow(alpha,2)*pow(np.log(alpha),2) + np.log(alpha)) * k
    
    S = []
    queries = 0

    
    
    # for l in range(rounds): # Since rounds = 1 as stated in the paper 
        
    if(int(alpha*k)>len(objective.groundset)):
        tmpk = len(objective.groundset)
    else:
        tmpk = int(alpha*k)

    p_start_dist = MPI.Wtime()

    ele_vals = Dist_Greedy(objective, tmpk, comm, rank, size, p_root, seed, nthreads)
    
    p_stop_dist = MPI.Wtime()
    time_dist = (p_stop_dist - p_start_dist)

    p_start_post = MPI.Wtime()

    S_DistGreedy = []

    if(rank ==0):
        S_rows_split_all = [val for sublist in ele_vals for val in sublist]
        if(objective.name=="ImgSum"):
            A = sparse.vstack(S_rows_split_all)
            A = A.todense()
        else:
            A = sparse.vstack(S_rows_split_all)
        objective_Root = create_objective(objective, A)
        objective = []
        S_DistGreedy = [ele for ele in range(len(objective_Root.groundset))]
        for j in range(int(jloop_iter)):
            currGains = parallel_margvals_returnvals_thread(objective_Root, S, S_DistGreedy, nthreads=objective_Root.nThreads)
            EleToAdd = np.argmax(currGains)
            S = list( set().union( S, [S_DistGreedy[EleToAdd]]) );
            # S.append(S_DistGreedy[EleToAdd])
            if(not run_Full):
                if(len(S)>=k):
                    break

        p_stop = MPI.Wtime()
        time_post = (p_stop - p_start_post)
        time = (p_stop - p_start)
        valSol = objective_Root.value( S )
        
        queries += 1
        
        output = pd.DataFrame({
                            'f_of_S'          :       [valSol], \
                            'Time'            :       [time], \
                            'TimeDist'        :       [time_dist], \
                            'TimePost'        :       [time_post], \
                            'S'               :       [S]
                            })
        return output

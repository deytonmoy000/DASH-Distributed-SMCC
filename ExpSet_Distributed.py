#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

import numpy as np
from datetime import datetime
import pandas as pd
from scipy import sparse
import networkx as nx
import random
import sys
#import ast
import os

# Disable  
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

from mpi4py import MPI

# # Load the Objective function classes 
from objectives import  NetCoverSparse,\
                        RevenueMaxOnNetSparse,\
                        InfluenceMaxSparse,\
                        ImageSummarizationMonotone

# Load our optimization algorithms and helper functions

from src import submodular_DASH_Distributed

def load_csr_matrix(filename):
    """Load compressed sparse row (csr) matrix from file.

       Based on http://stackoverflow.com/a/8980156/232571

    """
    assert filename.endswith('.npz')
    loader = np.load(filename)
    args = (loader['data'], loader['indices'], loader['indptr'])
    matrix = sparse.csr_matrix(args, shape=loader['shape'])
    return matrix

def run_DASH(objective, exp_string, k_vals_vec, filepath_string, experiment_string, comm, rank, size, size_groundset, nthreads, p_root=0, trials=1, gph = True):
    # size_groundset = objective.A.shape[1]
    nT = nthreads
    comm.barrier()
    algostring = 'DASH'
    algostring += '-'+str(size)+'-'+str(nT)
    if rank == p_root:
        print('Beginning ', algostring, 'for experiment: ', experiment_string)
    # eps = 0.007
    eps = 0.05
    val_vec = []
    queries_vec = []
    time_vec = []
    time_dist_vec = []
    time_post_vec = []
    sol_size_vec = []
    # Save data progressively.
    #STOP IF APPROX is reached
    stop_if_approx = False 
    
    for ii, kk in enumerate(k_vals_vec):

        for trial in range(trials):
            comm.barrier()

            # Free memory between trials
            sol = []
            sol_r = [] 
            time_r = []
            queries_r = []

            # Run the algorithm
            output = submodular_DASH_Distributed.DASH(objective, exp_string, kk, eps, comm, rank, size, p_root=0, seed=42, stop_if_aprx=stop_if_approx, nthreads=nthreads)

            output = comm.bcast(output, root=0)
            val = output.iloc[0]['f_of_S']
            # queries = output.iloc[0]['queries']
            time = output.iloc[0]['Time']
            time_dist = output.iloc[0]['TimeDist']
            time_post = output.iloc[0]['TimePost']
            sol = output.iloc[0]['S']

            if rank == p_root:
                val_vec.append(val)
                # queries_vec.append(queries)
                time_vec.append(time)
                time_dist_vec.append(time_dist)
                time_post_vec.append(time_post)
                sol_size_vec.append(len(sol))

                ## Save data progressively
                dataset = pd.DataFrame({'f_of_S':  val_vec, \
                                        # 'Queries': queries_vec, \
                                        'Time':    time_vec, \
                                        'TimeDist':    time_dist_vec, \
                                        'TimePost':    time_post_vec, \
                                        'SolSize':    sol_size_vec, \
                                        'k':       np.concatenate([np.repeat(k_vals_vec[:ii], trials), [kk]*(trial+1)]), \
                                        'n':       [size_groundset]*(ii*trials+trial+1), \
                                        'nNodes':   [size]*(ii*trials+trial+1), \
                                        'nThreads':   [nT]*(ii*trials+trial+1), \
                                        'trial':   np.concatenate([np.tile(range(1,(trials+1)), ii), range(1, (trial+2))])
                                        })
                dataset.to_csv(path_or_buf = filepath_string + experiment_string +'_exp2_'+ algostring +".csv", index=False)


            if rank == p_root:
                print('f(S)=', val, 'time=', time, algostring, experiment_string, 'k=', kk)
                print('\n')


def run_PGBDistributed(objective, exp_string, k_vals_vec, filepath_string, experiment_string, comm, rank, size, size_groundset, nthreads, p_root=0, trials=1, gph = True):
    procs_vec = [size]*len(k_vals_vec)
    # size_groundset = objective.A.shape[1]
    nT = nthreads
    n_vec = [size_groundset]*len(k_vals_vec)
    
    comm.barrier();
    algostring = 'PGBHeuristic'
    algostring += '-'+str(size)+'-'+str(nT)
    # if(size<=8):
    #     algostring = 'PGBDistributed-'+str(size)+'Proc-SingleNode'
    # else:
    #     algostring = 'PGBDistributed-'+str(size)+'Proc-MultiNode'

    OPT = None
    if rank == p_root:
        print('Beginning ', algostring, 'for experiment: ', experiment_string)
    eps = 0.05
    val_vec = []
    queries_vec = []
    time_vec = []
    time_dist_vec = []
    time_post_vec = []
    sol_size_vec = []
    for ii, kk in enumerate(k_vals_vec):

        for trial in range(trials):
            comm.barrier()

            # Free memory between trials
            sol = []
            sol_r = [] 
            time_r = []
            queries_r = []

            # Run the algorithm
            # val, queries, time, time_dist, time_post, sol, sol_r, time_r, queries_r = submodular_DASH_Distributed.ParallelGreedyBoostDistributed(objective, kk, eps, comm, rank, size, p_root, seed=trial, stop_if_aprx=False, nthreads=nthreads );
            output = submodular_DASH_Distributed.ParallelGreedyBoostDistributed(objective, exp_string, kk, eps, comm, rank, size, p_root, seed=trial, stop_if_aprx=False, nthreads=nthreads );
            
            output = comm.bcast(output, root=0)
            val = output.iloc[0]['f_of_S']
            # queries = output.iloc[0]['queries']
            time = output.iloc[0]['Time']
            time_dist = output.iloc[0]['TimeDist']
            time_post = output.iloc[0]['TimePost']
            sol = output.iloc[0]['S']
            if rank == p_root:
            
                print('f(S)=', val, 'time=', time, algostring, experiment_string, 'with k=', kk)

                val_vec.append(val)
                # queries_vec.append(queries)
                time_vec.append(time)
                time_dist_vec.append(time_dist)
                time_post_vec.append(time_post)
                sol_size_vec.append(len(sol))
                ## Save data progressively
                dataset = pd.DataFrame({'f_of_S':  val_vec, \
                                        # 'Queries': queries_vec, \
                                        'Time':    time_vec, \
                                        'TimeDist':    time_dist_vec, \
                                        'TimePost':    time_post_vec, \
                                        'SolSize':    sol_size_vec, \
                                        'k':       np.concatenate([np.repeat(k_vals_vec[:ii], trials), [kk]*(trial+1)]), \
                                        'n':       [size_groundset]*(ii*trials+trial+1), \
                                        'nNodes':   [size]*(ii*trials+trial+1), \
                                        'nThreads':   [nT]*(ii*trials+trial+1), \
                                        'trial':   np.concatenate([np.tile(range(1,(trials+1)), ii), range(1, (trial+2))])
                                        })
                dataset.to_csv(path_or_buf = filepath_string + experiment_string +'_exp2_'+ algostring +".csv", index=False)    


def run_RandGreedI(objective, exp_string, k_vals_vec, filepath_string, experiment_string, comm, rank, size, size_groundset, nthreads, p_root=0, trials=1, gph = True):
    # size_groundset = objective.A.shape[1]
    nT = nthreads
    comm.barrier()
    algostring = 'RandGreedI'
    algostring += '-'+str(size)+'-'+str(nT)
    if rank == p_root:
        print('Beginning ', algostring, 'for experiment: ', experiment_string)
    # eps = 0.007
    eps = 0.05
    val_vec = []
    queries_vec = []
    time_vec = []
    time_dist_vec = []
    time_post_vec = []
    sol_size_vec = []
    # Save data progressively.
    #STOP IF APPROX is reached
    stop_if_approx = False 
    
    for ii, kk in enumerate(k_vals_vec):

        for trial in range(trials):
            comm.barrier()

            # Free memory between trials
            sol = []
            sol_r = [] 
            time_r = []
            queries_r = []

            # Run the algorithm
            output = submodular_DASH_Distributed.RandGreedI(objective, exp_string, kk, eps, comm, rank, size, p_root=0, seed=42, nthreads=nthreads)

            output = comm.bcast(output, root=0)
            val = output.iloc[0]['f_of_S']
            # queries = output.iloc[0]['queries']
            time = output.iloc[0]['Time']
            time_dist = output.iloc[0]['TimeDist']
            time_post = output.iloc[0]['TimePost']
            sol = output.iloc[0]['S']

            if rank == p_root:
                val_vec.append(val)
                # queries_vec.append(queries)
                time_vec.append(time)
                time_dist_vec.append(time_dist)
                time_post_vec.append(time_post)
                sol_size_vec.append(len(sol))

                ## Save data progressively
                dataset = pd.DataFrame({'f_of_S':  val_vec, \
                                        # 'Queries': queries_vec, \
                                        'Time':    time_vec, \
                                        'TimeDist':    time_dist_vec, \
                                        'TimePost':    time_post_vec, \
                                        'SolSize':    sol_size_vec, \
                                        'k':       np.concatenate([np.repeat(k_vals_vec[:ii], trials), [kk]*(trial+1)]), \
                                        'n':       [size_groundset]*(ii*trials+trial+1), \
                                        'nNodes':   [size]*(ii*trials+trial+1), \
                                        'nThreads':   [nT]*(ii*trials+trial+1), \
                                        'trial':   np.concatenate([np.tile(range(1,(trials+1)), ii), range(1, (trial+2))])
                                        })
                dataset.to_csv(path_or_buf = filepath_string + experiment_string +'_exp2_'+ algostring +".csv", index=False)


            if rank == p_root:
                print('f(S)=', val, 'time=', time, algostring, experiment_string, 'k=', kk)
                print('\n')


def run_RandGreedI_LAG(objective, exp_string, k_vals_vec, filepath_string, experiment_string, comm, rank, size, size_groundset, nthreads, p_root=0, trials=1, gph = True):
    # size_groundset = objective.A.shape[1]
    nT = nthreads
    comm.barrier()
    algostring = 'RandGreedILAG'
    algostring += '-'+str(size)+'-'+str(nT)
    if rank == p_root:
        print('Beginning ', algostring, 'for experiment: ', experiment_string)
    # eps = 0.007
    eps = 0.05
    val_vec = []
    queries_vec = []
    time_vec = []
    time_dist_vec = []
    time_post_vec = []
    sol_size_vec = []
    # Save data progressively.
    #STOP IF APPROX is reached
    stop_if_approx = False 
    
    for ii, kk in enumerate(k_vals_vec):

        for trial in range(trials):
            comm.barrier()

            # Free memory between trials
            sol = []
            sol_r = [] 
            time_r = []
            queries_r = []

            # Run the algorithm
            output = submodular_DASH_Distributed.RandGreedI_LAG(objective, exp_string, kk, eps, comm, rank, size, p_root=0, seed=42, nthreads=nthreads)

            output = comm.bcast(output, root=0)
            val = output.iloc[0]['f_of_S']
            # queries = output.iloc[0]['queries']
            time = output.iloc[0]['Time']
            time_dist = output.iloc[0]['TimeDist']
            time_post = output.iloc[0]['TimePost']
            sol = output.iloc[0]['S']

            if rank == p_root:
                val_vec.append(val)
                # queries_vec.append(queries)
                time_vec.append(time)
                time_dist_vec.append(time_dist)
                time_post_vec.append(time_post)
                sol_size_vec.append(len(sol))

                ## Save data progressively
                dataset = pd.DataFrame({'f_of_S':  val_vec, \
                                        # 'Queries': queries_vec, \
                                        'Time':    time_vec, \
                                        'TimeDist':    time_dist_vec, \
                                        'TimePost':    time_post_vec, \
                                        'SolSize':    sol_size_vec, \
                                        'k':       np.concatenate([np.repeat(k_vals_vec[:ii], trials), [kk]*(trial+1)]), \
                                        'n':       [size_groundset]*(ii*trials+trial+1), \
                                        'nNodes':   [size]*(ii*trials+trial+1), \
                                        'nThreads':   [nT]*(ii*trials+trial+1), \
                                        'trial':   np.concatenate([np.tile(range(1,(trials+1)), ii), range(1, (trial+2))])
                                        })
                dataset.to_csv(path_or_buf = filepath_string + experiment_string +'_exp2_'+ algostring +".csv", index=False)


            if rank == p_root:
                print('f(S)=', val, 'time=', time, algostring, experiment_string, 'k=', kk)
                print('\n')


def run_BCG(objective, exp_string, k_vals_vec, filepath_string, experiment_string, comm, rank, size, size_groundset, nthreads, p_root=0, trials=1, gph = True):
    procs_vec = [size]*len(k_vals_vec)
    # size_groundset = objective.A.shape[1]
    nT = nthreads
    n_vec = [size_groundset]*len(k_vals_vec)
    
    comm.barrier();
    algostring = 'BiCriteriaGreedy'
    algostring += '-'+str(size)+'-'+str(nT)

    OPT = None
    if rank == p_root:
        print('Beginning ', algostring, 'for experiment: ', experiment_string)
    eps = 0.05
    val_vec = []
    queries_vec = []
    time_vec = []
    time_dist_vec = []
    time_post_vec = []
    sol_size_vec = []
    for ii, kk in enumerate(k_vals_vec):

        for trial in range(trials):
            comm.barrier()

            # Free memory between trials
            sol = []
            sol_r = [] 
            time_r = []
            queries_r = []

            # Run the algorithm
            # val, queries, time, time_dist, time_post, sol, sol_r, time_r, queries_r = submodular_DASH_Distributed.ParallelGreedyBoostDistributed(objective, kk, eps, comm, rank, size, p_root, seed=trial, stop_if_aprx=False, nthreads=nthreads );
            output = submodular_DASH_Distributed.BiCriteriaGreedy(objective, exp_string, kk, eps, comm, rank, size, p_root, seed=trial, nthreads=nthreads, run_Full=False );
            
            output = comm.bcast(output, root=0)
            val = output.iloc[0]['f_of_S']
            # queries = output.iloc[0]['queries']
            time = output.iloc[0]['Time']
            time_dist = output.iloc[0]['TimeDist']
            time_post = output.iloc[0]['TimePost']
            sol = output.iloc[0]['S']
            if rank == p_root:
            
                print('f(S)=', val, 'time=', time, algostring, experiment_string, 'with k=', kk)

                val_vec.append(val)
                # queries_vec.append(queries)
                time_vec.append(time)
                time_dist_vec.append(time_dist)
                time_post_vec.append(time_post)
                sol_size_vec.append(len(sol))
                ## Save data progressively
                dataset = pd.DataFrame({'f_of_S':  val_vec, \
                                        # 'Queries': queries_vec, \
                                        'Time':    time_vec, \
                                        'TimeDist':    time_dist_vec, \
                                        'TimePost':    time_post_vec, \
                                        'SolSize':    sol_size_vec, \
                                        'k':       np.concatenate([np.repeat(k_vals_vec[:ii], trials), [kk]*(trial+1)]), \
                                        'n':       [size_groundset]*(ii*trials+trial+1), \
                                        'nNodes':   [size]*(ii*trials+trial+1), \
                                        'nThreads':   [nT]*(ii*trials+trial+1), \
                                        'trial':   np.concatenate([np.tile(range(1,(trials+1)), ii), range(1, (trial+2))])
                                        })
                dataset.to_csv(path_or_buf = filepath_string + experiment_string +'_exp2_'+ algostring +".csv", index=False)    


def run_LTLG_experiments(objective, exp_string, k_vals_vec, filepath_string, experiment_string, comm, rank, size, size_groundset, algo = "ALL", nthreads=1, p_root=0, trials=1):
    """ Parallel MPI function to run all benchmark algorithms over all values of k for a given objective function and 
    save CSV files of data and runtimes """
    blockPrint()
    # run_RS(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, nthreads=nthreads)

    if(algo=="DASH"):
        run_DASH(objective, exp_string, k_vals_vec, filepath_string, experiment_string, comm, rank, size, size_groundset, nthreads=nthreads)
    
    if(algo=="PGBD"):
        run_PGBDistributed(objective, exp_string, k_vals_vec, filepath_string, experiment_string, comm, rank, size, size_groundset, nthreads=nthreads)

    if(algo=="RGGB"):
        run_RandGreedI(objective, exp_string, k_vals_vec, filepath_string, experiment_string, comm, rank, size, size_groundset, nthreads=nthreads)
    
    if(algo=="RGLAG"):
        run_RandGreedI_LAG(objective, exp_string, k_vals_vec, filepath_string, experiment_string, comm, rank, size, size_groundset, nthreads=nthreads)
    
    if(algo=="BCG"):
        run_BCG(objective, exp_string, k_vals_vec, filepath_string, experiment_string, comm, rank, size, size_groundset, nthreads=nthreads)


    enablePrint()
    comm.barrier()
        
    if rank == p_root:
        print('EXP 2 DIST FINISHED\n')
    comm.barrier()

    return

if __name__ == '__main__':

    start_runtime = datetime.now()

    p_root = 0

    filepath_string = "experiment_results_output_data/Exp2/"
    
    obj = str( sys.argv[1] );
    algoIn = str( sys.argv[2] );
    nthreads = int( sys.argv[3])
    # Start MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    k_vals_vec_tmp = [0.01, 0.02, 0.05, 0.10]
    if rank == p_root:
        print('Initializing run')


    directory = "../../data_exp2_split/"

    # ################################################################
    # ## MaxCOVER -  BA Graph Distrbuted         #####################
    # ################################################################
    if(obj=="BA"):
        comm.barrier()
        experiment_string = 'ba1m_sparse'
        exp_string = directory+'ba1m_sparse'
        size_groundset = 1000000
        size_ground_eachMachine = np.ceil(size_groundset/(size*size))
        k_vals_vec = [int(i * size_ground_eachMachine) for i in k_vals_vec_tmp]
        comm.barrier()
        
        run_LTLG_experiments(obj, exp_string, k_vals_vec, filepath_string, experiment_string, comm, rank, size, size_groundset, algo=algoIn, nthreads=nthreads)
    
    
    # ################################################################
    # ###           Image Summarisation - CIFAR 50,000 Images ########
    # ################################################################
    # Load the image data and generate the image pairwise distance matrix Dist
    
    
    if(obj=="IS"):
        comm.barrier()
        experiment_string = 'images_sparse'
        exp_string = directory+'images_sparse'

        size_groundset = 50000
        size_ground_eachMachine = np.ceil(size_groundset/(size*size))
        k_vals_vec = [int(i * size_ground_eachMachine) for i in k_vals_vec_tmp]
        comm.barrier()
        
        run_LTLG_experiments(obj, exp_string, k_vals_vec, filepath_string, experiment_string, comm, rank, size, size_groundset, algo=algoIn, nthreads=nthreads)

    # ################################################################
    # ##           INfluencemax - Youtube 1M            ##############
    # ################################################################
    if(obj=="IFM2"):
        comm.barrier()
        # experiment_string = 'twitter_sparse'
        experiment_string = 'Youtube_sparse'
        exp_string = directory+'youtube1m_split'
        size_groundset = 1134890
        size_ground_eachMachine = np.ceil(size_groundset/(size*size))
        k_vals_vec = [int(i * size_ground_eachMachine) for i in k_vals_vec_tmp]
        comm.barrier()
        
        run_LTLG_experiments(obj, exp_string, k_vals_vec, filepath_string, experiment_string, comm, rank, size, size_groundset, algo=algoIn, nthreads=nthreads)


    # ################################################################
    # ## REVENUE MAXIMIZATION -  Orkut 3M         ####################
    # ################################################################
    if(obj=="RVM2"):
        comm.barrier()
        # experiment_string = 'friendster_sparse'
        experiment_string = 'Orkut_sparse'
        exp_string = directory+'orkut3m_split'
        size_groundset = 3072441
        size_ground_eachMachine = np.ceil(size_groundset/(size*size))
        k_vals_vec = [int(i * size_ground_eachMachine) for i in k_vals_vec_tmp]
        comm.barrier()
        
        run_LTLG_experiments(obj, exp_string, k_vals_vec, filepath_string, experiment_string, comm, rank, size, size_groundset, algo=algoIn, nthreads=nthreads)
        
    if rank == p_root:
        print ('\n\nALL EXPERIMENTS COMPLETE, total minutes elapsed =', (datetime.now()-start_runtime).total_seconds()/60.0,'\n\n')

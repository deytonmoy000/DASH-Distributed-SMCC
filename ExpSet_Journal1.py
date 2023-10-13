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
from src import submodular_DASH_Centralized

'''
Journal Version Additional Algorithms
'''
#############################################################

def run_LDASH(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP, nthreads=1, p_root=0, trials=1, gph = True):
    size_groundset = objective.A.shape[1]
    nT = objective.nThreads
    comm.barrier()
    algostring = 'LDASH'
    algostring += '-'+str(size)+'-'+str(nT)+'_'+str(seedP)
    if rank == p_root:
        print('Beginning ', algostring, 'for experiment: ', experiment_string)
    # eps = 0.07
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
            val, time, time_dist, time_post, sol = submodular_DASH_Centralized.LDASH(objective, kk, eps, comm, rank, size, p_root=0, seed=seedP, nthreads=nthreads)

            if rank == p_root:
                val_vec.append(val)
                time_vec.append(time)
                time_dist_vec.append(time_dist)
                time_post_vec.append(time_post)
                sol_size_vec.append(len(sol))

                ## Save data progressively
                dataset = pd.DataFrame({'app':  [objective.name]*(ii*trials+trial+1), \
                                        'alg':  [algostring]*(ii*trials+trial+1), \
                                        'f_of_S':  val_vec, \
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
                dataset.to_csv(path_or_buf = filepath_string + experiment_string +'_exp1_'+ algostring +".csv", index=False)


            if rank == p_root:
                print('f(S)=', val,  'time=', time, algostring, experiment_string, 'k=', kk)
                print('\n')

def run_LDASH_localMax(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP, nthreads=1, p_root=0, trials=1, gph = True):
    size_groundset = objective.A.shape[1]
    nT = objective.nThreads
    comm.barrier()
    algostring = 'LDASH-localMax'
    algostring += '-'+str(size)+'-'+str(nT)+'_'+str(seedP)
    if rank == p_root:
        print('Beginning ', algostring, 'for experiment: ', experiment_string)
    # eps = 0.07
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
            val, time, time_dist, time_post, sol = submodular_DASH_Centralized.LDASH_localMax(objective, kk, eps, comm, rank, size, p_root=0, seed=seedP, nthreads=nthreads)

            if rank == p_root:
                val_vec.append(val)
                time_vec.append(time)
                time_dist_vec.append(time_dist)
                time_post_vec.append(time_post)
                sol_size_vec.append(len(sol))

                ## Save data progressively
                dataset = pd.DataFrame({'app':  [objective.name]*(ii*trials+trial+1), \
                                        'alg':  [algostring]*(ii*trials+trial+1), \
                                        'f_of_S':  val_vec, \
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
                dataset.to_csv(path_or_buf = filepath_string + experiment_string +'_exp1_'+ algostring +".csv", index=False)


            if rank == p_root:
                print('f(S)=', val,  'time=', time, algostring, experiment_string, 'k=', kk)
                print('\n')

def run_LDASH_LAG(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP, nthreads=1, p_root=0, trials=1, gph = True):
    size_groundset = objective.A.shape[1]
    nT = objective.nThreads
    comm.barrier()
    algostring = 'LDASH-LAG'
    algostring += '-'+str(size)+'-'+str(nT)+'_'+str(seedP)
    if rank == p_root:
        print('Beginning ', algostring, 'for experiment: ', experiment_string)
    # eps = 0.07
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
            val, time, time_dist, time_post, sol = submodular_DASH_Centralized.LDASH_LAG(objective, kk, eps, comm, rank, size, p_root=0, seed=seedP, nthreads=nthreads)

            if rank == p_root:
                val_vec.append(val)
                time_vec.append(time)
                time_dist_vec.append(time_dist)
                time_post_vec.append(time_post)
                sol_size_vec.append(len(sol))

                ## Save data progressively
                dataset = pd.DataFrame({'app':  [objective.name]*(ii*trials+trial+1), \
                                        'alg':  [algostring]*(ii*trials+trial+1), \
                                        'f_of_S':  val_vec, \
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
                dataset.to_csv(path_or_buf = filepath_string + experiment_string +'_exp1_'+ algostring +".csv", index=False)


            if rank == p_root:
                print('f(S)=', val,  'time=', time, algostring, experiment_string, 'k=', kk)
                print('\n')

def run_LDASH_LAS_LAG(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP, nthreads=1, p_root=0, trials=1, gph = True):
    size_groundset = objective.A.shape[1]
    nT = objective.nThreads
    comm.barrier()
    algostring = 'LDASH-LAS-LAG'
    algostring += '-'+str(size)+'-'+str(nT)+'_'+str(seedP)
    if rank == p_root:
        print('Beginning ', algostring, 'for experiment: ', experiment_string)
    # eps = 0.07
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
            val, time, time_dist, time_post, sol = submodular_DASH_Centralized.LDASH_LAS_LAG(objective, kk, eps, comm, rank, size, p_root=0, seed=seedP, nthreads=nthreads)

            if rank == p_root:
                val_vec.append(val)
                time_vec.append(time)
                time_dist_vec.append(time_dist)
                time_post_vec.append(time_post)
                sol_size_vec.append(len(sol))

                ## Save data progressively
                dataset = pd.DataFrame({'app':  [objective.name]*(ii*trials+trial+1), \
                                        'alg':  [algostring]*(ii*trials+trial+1), \
                                        'f_of_S':  val_vec, \
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
                dataset.to_csv(path_or_buf = filepath_string + experiment_string +'_exp1_'+ algostring +".csv", index=False)


            if rank == p_root:
                print('f(S)=', val,  'time=', time, algostring, experiment_string, 'k=', kk)
                print('\n')


def run_MED_LDASH(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP, nthreads=1, p_root=0, trials=1, gph = True):
    size_groundset = objective.A.shape[1]
    nT = objective.nThreads
    comm.barrier()
    algostring = 'MEDLDASH'
    algostring += '-'+str(size)+'-'+str(nT)+'_'+str(seedP)
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
            val, time, time_dist, time_post, sol = submodular_DASH_Centralized.MEDLDASH(objective, kk, eps, comm, rank, size, p_root=0, seed=seedP, nthreads=nthreads)

            if rank == p_root:
                val_vec.append(val)
                time_vec.append(time)
                time_dist_vec.append(time_dist)
                time_post_vec.append(time_post)
                sol_size_vec.append(len(sol))

                ## Save data progressively
                dataset = pd.DataFrame({'app':  [objective.name]*(ii*trials+trial+1), \
                                        'alg':  [algostring]*(ii*trials+trial+1), \
                                        'f_of_S':  val_vec, \
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
                dataset.to_csv(path_or_buf = filepath_string + experiment_string +'_exp1_'+ algostring +".csv", index=False)


            if rank == p_root:
                print('f(S)=', val,  'time=', time, algostring, experiment_string, 'k=', kk)
                print('\n')

def run_MED_LDASH_localMax(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP, nthreads=1, p_root=0, trials=1, gph = True):
    size_groundset = objective.A.shape[1]
    nT = objective.nThreads
    comm.barrier()
    algostring = 'MEDLDASH-localMax'
    algostring += '-'+str(size)+'-'+str(nT)+'_'+str(seedP)
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
            val, time, time_dist, time_post, sol = submodular_DASH_Centralized.MEDLDASH_localMax(objective, kk, eps, comm, rank, size, p_root=0, seed=seedP, nthreads=nthreads)

            if rank == p_root:
                val_vec.append(val)
                time_vec.append(time)
                time_dist_vec.append(time_dist)
                time_post_vec.append(time_post)
                sol_size_vec.append(len(sol))

                ## Save data progressively
                dataset = pd.DataFrame({'app':  [objective.name]*(ii*trials+trial+1), \
                                        'alg':  [algostring]*(ii*trials+trial+1), \
                                        'f_of_S':  val_vec, \
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
                dataset.to_csv(path_or_buf = filepath_string + experiment_string +'_exp1_'+ algostring +".csv", index=False)


            if rank == p_root:
                print('f(S)=', val,  'time=', time, algostring, experiment_string, 'k=', kk)
                print('\n')

def run_MED_LDASH_LAG(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP, nthreads=1, p_root=0, trials=1, gph = True):
    size_groundset = objective.A.shape[1]
    nT = objective.nThreads
    comm.barrier()
    algostring = 'MEDLDASH-LAG'
    algostring += '-'+str(size)+'-'+str(nT)+'_'+str(seedP)
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
            val, time, time_dist, time_post, sol = submodular_DASH_Centralized.MEDLDASH_LAG(objective, kk, eps, comm, rank, size, p_root=0, seed=seedP, nthreads=nthreads)

            if rank == p_root:
                val_vec.append(val)
                time_vec.append(time)
                time_dist_vec.append(time_dist)
                time_post_vec.append(time_post)
                sol_size_vec.append(len(sol))

                ## Save data progressively
                dataset = pd.DataFrame({'app':  [objective.name]*(ii*trials+trial+1), \
                                        'alg':  [algostring]*(ii*trials+trial+1), \
                                        'f_of_S':  val_vec, \
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
                dataset.to_csv(path_or_buf = filepath_string + experiment_string +'_exp1_'+ algostring +".csv", index=False)


            if rank == p_root:
                print('f(S)=', val,  'time=', time, algostring, experiment_string, 'k=', kk)
                print('\n')

#############################################################

def run_DQS_QS_TG(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP, nthreads=1, p_root=0, trials=1, gph = True):
    size_groundset = objective.A.shape[1]
    nT = objective.nThreads
    comm.barrier()
    algostring = 'DQS'
    algostring += '-'+str(size)+'-'+str(nT)+'_'+str(seedP)
    if rank == p_root:
        print('Beginning ', algostring, 'for experiment: ', experiment_string)
    # eps = 0.07
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
            val, time, time_dist, time_post, sol = submodular_DASH_Centralized.DQS_QS_TG(objective, kk, eps, comm, rank, size, p_root=0, seed=seedP, nthreads=nthreads)

            if rank == p_root:
                val_vec.append(val)
                time_vec.append(time)
                time_dist_vec.append(time_dist)
                time_post_vec.append(time_post)
                sol_size_vec.append(len(sol))

                ## Save data progressively
                dataset = pd.DataFrame({'app':  [objective.name]*(ii*trials+trial+1), \
                                        'alg':  [algostring]*(ii*trials+trial+1), \
                                        'f_of_S':  val_vec, \
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
                dataset.to_csv(path_or_buf = filepath_string + experiment_string +'_exp1_'+ algostring +".csv", index=False)


            if rank == p_root:
                print('f(S)=', val,  'time=', time, algostring, experiment_string, 'k=', kk)
                print('\n')


def run_DASH(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP, nthreads=1, p_root=0, trials=1, gph = True):
    size_groundset = objective.A.shape[1]
    nT = objective.nThreads
    comm.barrier()
    algostring = 'RDASHn'
    algostring += '-'+str(size)+'-'+str(nT)+'_'+str(seedP)
    if rank == p_root:
        print('Beginning ', algostring, 'for experiment: ', experiment_string)
    # eps = 0.07
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
            val, time, time_dist, time_post, sol = submodular_DASH_Centralized.RDASH(objective, kk, eps, comm, rank, size, p_root=0, seed=seedP, stop_if_aprx=stop_if_approx, nthreads=nthreads)

            if rank == p_root:
                val_vec.append(val)
                time_vec.append(time)
                time_dist_vec.append(time_dist)
                time_post_vec.append(time_post)
                sol_size_vec.append(len(sol))

                ## Save data progressively
                dataset = pd.DataFrame({'app':  [objective.name]*(ii*trials+trial+1), \
                                        'alg':  [algostring]*(ii*trials+trial+1), \
                                        'f_of_S':  val_vec, \
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
                dataset.to_csv(path_or_buf = filepath_string + experiment_string +'_exp1_'+ algostring +".csv", index=False)


            if rank == p_root:
                print('f(S)=', val,  'time=', time, algostring, experiment_string, 'k=', kk)
                print('\n')


def run_DAT(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP, nthreads=1, p_root=0, trials=1, gph = True):
    size_groundset = objective.A.shape[1]
    nT = objective.nThreads
    comm.barrier()
    algostring = 'DAT'
    algostring += '-'+str(size)+'-'+str(nT)+'_'+str(seedP)
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
            val, time, time_dist, time_post, sol = submodular_DASH_Centralized.DAT_GuessOPT(objective, kk, eps, comm, rank, size, p_root=0, seed=seedP, stop_if_aprx=stop_if_approx, nthreads=nthreads)

            if rank == p_root:
                val_vec.append(val)
                time_vec.append(time)
                time_dist_vec.append(time_dist)
                time_post_vec.append(time_post)
                sol_size_vec.append(len(sol))

                ## Save data progressively
                dataset = pd.DataFrame({'app':  [objective.name]*(ii*trials+trial+1), \
                                        'alg':  [algostring]*(ii*trials+trial+1), \
                                        'f_of_S':  val_vec, \
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
                dataset.to_csv(path_or_buf = filepath_string + experiment_string +'_exp1_'+ algostring +".csv", index=False)


            if rank == p_root:
                print('f(S)=', val,  'time=', time, algostring, experiment_string, 'k=', kk)
                print('\n')


def run_PGB(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP, nthreads=1, p_root=0, trials=1, gph = True):
    
    size_groundset = objective.A.shape[1]
    nT = objective.nThreads
    
    comm.barrier();
    algostring = 'PGB'
    algostring += '-'+str(size)+'-'+str(nT)+'_'+str(seedP)

    
    if rank == p_root:
        print('Beginning ', algostring, 'for experiment: ', experiment_string)
    eps = 0.05
    val_vec = []
    queries_vec = []
    time_vec = []
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
            val, time, sol, sol_r, time_r, queries_r = submodular_DASH_Centralized.ParallelGreedyBoost_Original_MultiNode(objective, kk, eps, comm, rank, size, p_root, seed=seedP, stop_if_approx=False );
            if rank == p_root:
            
                print('f(S)=', val,  'time=', time, algostring, experiment_string, 'with k=', kk)

                val_vec.append(val)
                time_vec.append(time)
                sol_size_vec.append(len(sol))
                ## Save data progressively
                dataset = pd.DataFrame({'app':  [objective.name]*(ii*trials+trial+1), \
                                        'alg':  [algostring]*(ii*trials+trial+1), \
                                        'f_of_S':  val_vec, \
                                        'Time':    time_vec, \
                                        'SolSize':    sol_size_vec, \
                                        'k':       np.concatenate([np.repeat(k_vals_vec[:ii], trials), [kk]*(trial+1)]), \
                                        'n':       [size_groundset]*(ii*trials+trial+1), \
                                        'nNodes':   [size]*(ii*trials+trial+1), \
                                        'nThreads':   [nT]*(ii*trials+trial+1), \
                                        'trial':   np.concatenate([np.tile(range(1,(trials+1)), ii), range(1, (trial+2))])
                                        })
                dataset.to_csv(path_or_buf = filepath_string + experiment_string +'_exp1_'+ algostring +".csv", index=False)    


def run_MED_DASH(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP, nthreads=1, p_root=0, trials=1, gph = True):
    size_groundset = objective.A.shape[1]
    nT = objective.nThreads
    comm.barrier()
    algostring = 'MEDRDASH'
    algostring += '-'+str(size)+'-'+str(nT)+'_'+str(seedP)
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
            val, time, time_dist, time_post, sol = submodular_DASH_Centralized.MEDDASH(objective, kk, eps, comm, rank, size, p_root=0, seed=seedP, nthreads=nthreads)

            if rank == p_root:
                val_vec.append(val)
                time_vec.append(time)
                time_dist_vec.append(time_dist)
                time_post_vec.append(time_post)
                sol_size_vec.append(len(sol))

                ## Save data progressively
                dataset = pd.DataFrame({'app':  [objective.name]*(ii*trials+trial+1), \
                                        'alg':  [algostring]*(ii*trials+trial+1), \
                                        'f_of_S':  val_vec, \
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
                dataset.to_csv(path_or_buf = filepath_string + experiment_string +'_exp1_'+ algostring +".csv", index=False)


            if rank == p_root:
                print('f(S)=', val,  'time=', time, algostring, experiment_string, 'k=', kk)
                print('\n')


def run_MED_RG(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP, nthreads=1, p_root=0, trials=1, gph = True):
    size_groundset = objective.A.shape[1]
    nT = objective.nThreads
    comm.barrier()
    algostring = 'MEDRG'
    algostring += '-'+str(size)+'-'+str(nT)+'_'+str(seedP)
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
            val, time, time_dist, time_post, sol = submodular_DASH_Centralized.MEDRG(objective, kk, eps, comm, rank, size, p_root=0, seed=seedP, nthreads=nthreads)

            if rank == p_root:
                val_vec.append(val)
                time_vec.append(time)
                time_dist_vec.append(time_dist)
                time_post_vec.append(time_post)
                sol_size_vec.append(len(sol))

                ## Save data progressively
                dataset = pd.DataFrame({'app':  [objective.name]*(ii*trials+trial+1), \
                                        'alg':  [algostring]*(ii*trials+trial+1), \
                                        'f_of_S':  val_vec, \
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
                dataset.to_csv(path_or_buf = filepath_string + experiment_string +'_exp1_'+ algostring +".csv", index=False)


            if rank == p_root:
                print('f(S)=', val,  'time=', time, algostring, experiment_string, 'k=', kk)
                print('\n')


def run_RandGreedI(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP, nthreads=1, p_root=0, trials=1, gph = True):
    size_groundset = objective.A.shape[1]
    nT = objective.nThreads
    comm.barrier();
    algostring = 'RandGreedI'
    algostring += '-'+str(size)+'-'+str(nT)+'_'+str(seedP)
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
            val, time, time_dist, time_post, sol, sol_r, time_r, queries_r = submodular_DASH_Centralized.RandGreedI(objective, kk, eps, comm, rank, size, p_root=0, seed=seedP, nthreads=nthreads)
            
            if rank == p_root:
            
                print('f(S)=', val,  'time=', time, algostring, experiment_string, 'with k=', kk)

                val_vec.append(val)
                time_vec.append(time)
                time_dist_vec.append(time_dist)
                time_post_vec.append(time_post)
                sol_size_vec.append(len(sol))

                ## Save data progressively
                dataset = pd.DataFrame({'app':  [objective.name]*(ii*trials+trial+1), \
                                        'alg':  [algostring]*(ii*trials+trial+1), \
                                        'f_of_S':  val_vec, \
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
                dataset.to_csv(path_or_buf = filepath_string + experiment_string +'_exp1_'+ algostring +".csv", index=False)    


def run_RandGreedI_LAG(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP, nthreads=1, p_root=0, trials=1, gph = True):
    size_groundset = objective.A.shape[1]
    nT = objective.nThreads
    comm.barrier();
    algostring = 'RandGreedILAG'
    algostring += '-'+str(size)+'-'+str(nT)+'_'+str(seedP)
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
            val, time, time_dist, time_post, sol, sol_r, time_r, queries_r = submodular_DASH_Centralized.RandGreedI_LAG(objective, kk, eps, comm, rank, size, p_root=0, seed=seedP, nthreads=nthreads)
            
            if rank == p_root:
            
                print('f(S)=', val,  'time=', time, algostring, experiment_string, 'with k=', kk)

                val_vec.append(val)
                time_vec.append(time)
                time_dist_vec.append(time_dist)
                time_post_vec.append(time_post)
                sol_size_vec.append(len(sol))

                ## Save data progressively
                dataset = pd.DataFrame({'app':  [objective.name]*(ii*trials+trial+1), \
                                        'alg':  [algostring]*(ii*trials+trial+1), \
                                        'f_of_S':  val_vec, \
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
                dataset.to_csv(path_or_buf = filepath_string + experiment_string +'_exp1_'+ algostring +".csv", index=False)    


def run_BiCriteriaGreedy(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP, nthreads=1, p_root=0, trials=1, gph = True):
    size_groundset = objective.A.shape[1]
    nT = objective.nThreads
    comm.barrier()
    algostring = 'BiCriteriaGreedy'
    algostring += '-'+str(size)+'-'+str(nT)+'_'+str(seedP)
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
            val, time, time_dist, time_post, sol = submodular_DASH_Centralized.BiCriteriaGreedy(objective, kk, eps, comm, rank, size, p_root=0, seed=seedP, nthreads=nthreads, run_Full=False)

            if rank == p_root:
                val_vec.append(val)
                time_vec.append(time)
                time_dist_vec.append(time_dist)
                time_post_vec.append(time_post)
                sol_size_vec.append(len(sol))

                ## Save data progressively
                dataset = pd.DataFrame({'app':  [objective.name]*(ii*trials+trial+1), \
                                        'alg':  [algostring]*(ii*trials+trial+1), \
                                        'f_of_S':  val_vec, \
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
                dataset.to_csv(path_or_buf = filepath_string + experiment_string +'_exp1_'+ algostring +".csv", index=False)


            if rank == p_root:
                print('f(S)=', val,  'time=', time, algostring, experiment_string, 'k=', kk)
                print('\n')


def run_GDASH(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP, nthreads=1, p_root=0, trials=1, gph = True):
    size_groundset = objective.A.shape[1]
    nT = objective.nThreads
    comm.barrier()
    algostring = 'GDASH'
    algostring += '-'+str(size)+'-'+str(nT)+'_'+str(seedP)
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
    for ii, kk in enumerate(k_vals_vec):

        for trial in range(trials):
            comm.barrier()

            # Free memory between trials
            sol = []
            # Run the algorithm
            val, time, sol = submodular_DASH_Centralized.GDASH(objective, kk, eps, comm, rank, size, p_root=0, seed=seedP, nthreads=nthreads)

            if rank == p_root:
                val_vec.append(val)
                time_vec.append(time)
                time_dist_vec.append(0)
                time_post_vec.append(0)
                sol_size_vec.append(len(sol))

                ## Save data progressively
                dataset = pd.DataFrame({'app':  [objective.name]*(ii*trials+trial+1), \
                                        'alg':  [algostring]*(ii*trials+trial+1), \
                                        'f_of_S':  val_vec, \
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
                dataset.to_csv(path_or_buf = filepath_string + experiment_string +'_exp1_'+ algostring +".csv", index=False)


            if rank == p_root:
                print('f(S)=', val,  'time=', time, algostring, experiment_string, 'k=', kk)
                print('\n')


def run_DD(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP, nthreads=1, p_root=0, trials=1, gph = True):
    size_groundset = objective.A.shape[1]
    nT = objective.nThreads
    comm.barrier()
    algostring = 'DD'
    algostring += '-'+str(size)+'-'+str(nT)+'_'+str(seedP)
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
    for ii, kk in enumerate(k_vals_vec):

        for trial in range(trials):
            comm.barrier()

            # Free memory between trials
            sol = []
            # Run the algorithm
            val, time, sol = submodular_DASH_Centralized.DistortedDistributed(objective, kk, eps, comm, rank, size, p_root=0, seed=seedP, nthreads=nthreads)

            if rank == p_root:
                val_vec.append(val)
                time_vec.append(time)
                time_dist_vec.append(0)
                time_post_vec.append(0)
                sol_size_vec.append(len(sol))

                ## Save data progressively
                dataset = pd.DataFrame({'app':  [objective.name]*(ii*trials+trial+1), \
                                        'alg':  [algostring]*(ii*trials+trial+1), \
                                        'f_of_S':  val_vec, \
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
                dataset.to_csv(path_or_buf = filepath_string + experiment_string +'_exp1_'+ algostring +".csv", index=False)


            if rank == p_root:
                print('f(S)=', val,  'time=', time, algostring, experiment_string, 'k=', kk)
                print('\n')


def run_ParallelAlgoGreedy(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP, nthreads=1, p_root=0, trials=1, gph = True):
    size_groundset = objective.A.shape[1]
    nT = objective.nThreads
    comm.barrier()
    algostring = 'ParallelAlgoGreedy'
    algostring += '-'+str(size)+'-'+str(nT)+'_'+str(seedP)
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
    for ii, kk in enumerate(k_vals_vec):

        for trial in range(trials):
            comm.barrier()

            # Free memory between trials
            sol = []
            # Run the algorithm
            val, time, sol = submodular_DASH_Centralized.ParallelAlgoGreedy(objective, kk, eps, comm, rank, size, p_root=0, seed=seedP, nthreads=nthreads)

            if rank == p_root:
                val_vec.append(val)
                time_vec.append(time)
                time_dist_vec.append(0)
                time_post_vec.append(0)
                sol_size_vec.append(len(sol))

                ## Save data progressively
                dataset = pd.DataFrame({'app':  [objective.name]*(ii*trials+trial+1), \
                                        'alg':  [algostring]*(ii*trials+trial+1), \
                                        'f_of_S':  val_vec, \
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
                dataset.to_csv(path_or_buf = filepath_string + experiment_string +'_exp1_'+ algostring +".csv", index=False)


            if rank == p_root:
                print('f(S)=', val,  'time=', time, algostring, experiment_string, 'k=', kk)
                print('\n')


def run_LTLG_experiments(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedV=42, algo = "ALL", nthreads=1, p_root=0, trials=1):
    """ Parallel MPI function to run all benchmark algorithms over all values of k for a given objective function and 
    save CSV files of data and runtimes """
    blockPrint()
    # run_RS(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, nthreads=nthreads)
    seed = seedV #  Seed values = 8, 14, 25, 35, 42 (5 repetitions for non-Greedy algorithms)
    if(algo=="BCG"):
        run_BiCriteriaGreedy(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP=seed, nthreads=objective.nThreads)


    if(algo=="GDASH"):
        run_GDASH(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP=seed, nthreads=objective.nThreads)


    if(algo=="DD"):
        run_DD(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP=seed, nthreads=objective.nThreads)


    if(algo=="PAG"):
        run_ParallelAlgoGreedy(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP=seed, nthreads=objective.nThreads)

    
    if(algo=="RGGB"):    
        run_RandGreedI(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP=seed, nthreads=objective.nThreads)

    
    if(algo=="RGLAG"):    
        run_RandGreedI_LAG(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP=seed, nthreads=objective.nThreads)

    
    if(algo=="DASH"):
        run_DASH(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP=seed, nthreads=objective.nThreads)


    if(algo=="MEDDASH"):
        run_MED_DASH(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP=seed, nthreads=objective.nThreads)

    
    if(algo=="MEDRG"):
        run_MED_RG(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP=seed, nthreads=objective.nThreads)


    if(algo=="DAT"):
        run_DAT(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP=seed, nthreads=objective.nThreads)


    if(algo=="PGBD"):
        run_PGB(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP=seed)


    if(algo=="LDASH"):
        run_LDASH(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP=seed, nthreads=objective.nThreads)
    

    if(algo=="MEDLDASH"):
        run_MED_LDASH(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP=seed, nthreads=objective.nThreads)

    if(algo=="LDASHLM"):
        run_LDASH_localMax(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP=seed, nthreads=objective.nThreads)
    

    if(algo=="MEDLDASHLM"):
        run_MED_LDASH_localMax(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP=seed, nthreads=objective.nThreads)

    if(algo=="LDASHLAG"):
        run_LDASH_LAG(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP=seed, nthreads=objective.nThreads)
    
    if(algo=="LDASHLASLAG"):
        run_LDASH_LAS_LAG(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP=seed, nthreads=objective.nThreads)

    if(algo=="DQS"):
        run_DQS_QS_TG(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP=seed, nthreads=objective.nThreads)

    if(algo=="MEDLDASHLAG"):
        run_MED_LDASH_LAG(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedP=seed, nthreads=objective.nThreads)


    enablePrint()
    comm.barrier()
        
    if rank == p_root:
        print('EXP 1 FINISHED\n')
    comm.barrier()

    return

if __name__ == '__main__':

    start_runtime = datetime.now()

    p_root = 0

    filepath_string = "results_TAMU/Exp1/"
    
    obj = str( sys.argv[1] );
    algoIn = str( sys.argv[2] );
    nthreads = int( sys.argv[3])
    seedP = int(sys.argv[4])
    directory = 'data_large_TAMU/'
    # Start MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == p_root:
        print('Initializing run')
        print('Arguments: ','--obj=', obj, ' --algoIn=', algoIn, ' --nthreads=', nthreads, ' --size:', size)

    
      
    
    k_vals_vec_tmp = [0.05, 0.10, 0.15, 0.20] # [0.05, 0.10, 0.15, 0.20]

    # ################################################################
    # ## MaxCover - BA Graph #####################
    # ################################################################
    if(obj=="MCV"):
        comm.barrier()
        if rank == p_root:
            print( 'Initializing MaxCov Objective' )
        experiment_string = 'MaxCov-ba5M'
        # k_vals_vec = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10000]
    
        # if rank == p_root:
    
            # G = nx.barabasi_albert_graph(size_of_ground_set, 5, seed=42)
            # try:
            #     G.remove_edges_from(G.selfloop_edges())
            # except:
            #     G.remove_edges_from(nx.selfloop_edges(G)) #Later version of networkx prefers this syntax
    
            # if size_of_ground_set < 100:
            #     A = np.asarray( nx.to_numpy_matrix(G) )
            #     A.fill_diagonal(1)
            #     A.astype('bool_')
            #     objective_rootprocessor = NetCover.NetCover(A)
    
            # else:
            #     A = nx.to_scipy_sparse_matrix(G, format='csr')
            #     A.setdiag(1)
            #     # Generate our NetCover class containing the function
            #     objective_rootprocessor = NetCoverSparse.NetCoverSparse(A, nthreads)
            
        path=directory+"ba5M.npz"
        size_of_ground_set = 5000000
        size_ground_eachMachine = np.ceil(size_of_ground_set/(64*64))
        k_vals_vec = [int(i * size_ground_eachMachine) for i in k_vals_vec_tmp]
        
        A = sparse.load_npz(path)
            # print("A.shape:", A.shape)
        # print("Rank: ", rank, " A.shape: ", A.shape)
        objective = NetCoverSparse.NetCoverSparse(A, nthreads)
        # Send class to all processors
        # if rank != 0:
        #     objective_rootprocessor = None
        # objective = comm.bcast(objective_rootprocessor, root=0)
        
        if rank == p_root:
            print( 'MaxCov Objective initialized(objective.A.shape:', objective.A.shape,'). Beginning tests (--k=[', k_vals_vec, '])' )
        comm.barrier()
        run_LTLG_experiments(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedV=seedP, algo=algoIn, nthreads=nthreads)
    
        comm.barrier()
    
    
    # ################################################################
    # ##           InfluenceMax - ER-5M      ##############
    # ################################################################
    if(obj=="IFM"):
        comm.barrier()
        if rank == p_root:
            print( 'Initializing influence max Objective' )
    
        # experiment_string = 'InfMax-er5M'
        experiment_string = 'InfMax-youtube1M'
    
        # Undirected Facebook Network. Format as an adjacency matrix
        # filename_net = directory+"soc-epinions.csv"
    
    
        # edgelist = pd.read_csv(filename_net)
        # net_nx = nx.from_pandas_edgelist(edgelist, source='source', target='target', edge_attr=None, create_using=None)
        # net_nx = net_nx.to_undirected()
        # try:
        #     net_nx.remove_edges_from(net_nx.selfloop_edges())
        # except:
        #     net_nx.remove_edges_from(nx.selfloop_edges(net_nx)) #Later version of networkx prefers this syntax
    
    
        # #A = np.asarray( nx.adjacency_matrix(net_nx).todense() )
        # if rank == p_root:
        #     print( 'Loaded data. Generating sparse adjacency matrix' )
        # A = nx.to_scipy_sparse_matrix(net_nx, format='csr')
        # A.setdiag(1)
        # path=directory+"er5M.npz"
        # size_of_ground_set = 5000000
        # size_ground_eachMachine = np.ceil(size_of_ground_set/(size*size))
        # k_vals_vec = [int(i * size_ground_eachMachine) for i in k_vals_vec_tmp]
        
        path=directory+"youtube1M.npz"
        size_of_ground_set = 1134890
        size_ground_eachMachine = np.ceil(size_of_ground_set/(64*64))
        k_vals_vec = [int(i * size_ground_eachMachine) for i in k_vals_vec_tmp]
        
        A = sparse.load_npz(path)
        # print("Rank: ", rank, " A.shape: ", A.shape)
        #objective = NetCover.NetCover(A)
        p = 0.01
        objective = InfluenceMaxSparse.InfluenceMaxSparse(A, p, nthreads)
        if rank == p_root:
            print( 'InfMax Objective of', A.shape[0], 'elements initialized. Beginning tests (--k=[', k_vals_vec, '])' )
    
        comm.barrier()
        run_LTLG_experiments(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedV=seedP, algo=algoIn)





    # ################################################################
    # ## WS-5M REVENUE MAXIMIZATION EXAMPLE ####################
    # ################################################################
    if(obj=="RVM"):
        comm.barrier()
        if rank == p_root:
            print( 'Initializing Youtube Objective' )
        # experiment_string = 'RevMax-ws5M'
        experiment_string = 'RevMax-orkut3M'
    
        # edgelist = pd.read_csv(directory+'youtube_2000rand_edgelist.csv', delimiter=',')
        # A = edgelist.pivot(index = "source", columns = "target", values = "weight_draw")
        # A = A.values
        # A[np.isnan(A)] = 0
        # A[A>0] = A[A>0] + 1.0
        # # A.setdiag(0)
        # np.fill_diagonal(A, 0)
        # # Set the power between 0 and 1. More towards 0 means revenue is more subadditive as a node gets more influences
        alpha = 0.3
        # A = sparse.csr_matrix(A)
        # Generate class containing our f(S)


        # path=directory+"ws5M.npz"
        # size_of_ground_set = 5000000
        # size_ground_eachMachine = np.ceil(size_of_ground_set/(size*size))
        # k_vals_vec = [int(i * size_ground_eachMachine) for i in k_vals_vec_tmp]
        
        path=directory+"orkut3M.npz"
        size_of_ground_set = 3072441
        size_ground_eachMachine = np.ceil(size_of_ground_set/(64*64))
        k_vals_vec = [int(i * size_ground_eachMachine) for i in k_vals_vec_tmp]

        A = sparse.load_npz(path)
        # print("Rank: ", rank, " A.shape: ", A.shape)
        objective = RevenueMaxOnNetSparse.RevenueMaxOnNetSparse(A, alpha, nthreads)
        if rank == p_root:
            print( 'RevMax Objective initialized. Adjacency matrix shape is:', A.shape, ' Beginning tests (--k=[', k_vals_vec, '])' )
    
        comm.barrier()
        run_LTLG_experiments(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedV=seedP, algo=algoIn)
    


    # ################################################################
    # ###     Image Summarisation  - CIFAR 10           ##############
    # ################################################################
    # Load the image data and generate the image pairwise distance matrix Dist
    
    #Download the "images_10K_mat.csv" file from "https://file.io/w0PEXw4j5Xcx" and place it in "data/data_exp1" #
    if(obj=="IS"):
        comm.barrier()
        if rank == p_root:
            print( 'Initializing image summ Objective' )
    
        experiment_string = 'IMAGESUMM'
    
        # Undirected Facebook Network. Format as an adjacency matrix
        filename_net = directory+"images_10K_mat.csv"
        
        # Sim = ImageSummarizationMonotone.load_image_similarity_matrix(filename_net)
        Sim = pd.read_csv(filename_net, header=None).values
        
        
        if rank == p_root:
            print( 'Loaded data. Generated adjacency matrix' )
        objective = ImageSummarizationMonotone.ImageSummarizationMonotone(Sim, nthreads)
        
        if rank == p_root:
            print( ' Objective of', Sim.shape[0], 'images initialized. Beginning tests.' )
    
        comm.barrier()
        run_LTLG_experiments(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, seedV=seedP, algo=algoIn, nthreads=nthreads)
    
    if rank == p_root:
        print ('\n\nALL EXPERIMENTS COMPLETE, total minutes elapsed =', (datetime.now()-start_runtime).total_seconds()/60.0,'\n\n')

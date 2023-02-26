#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite



class ImageSummarizationMonotone:  

    
    def __init__(self, image_similarity_matrix, nThreads):
        """
        class ImageSummarization:
            INPUTS and INSTANCE VARIABLES:
                image_similarity_matrix (becomes instance variable A for short): a 2D symmetric np.array of float64s
        """

        self.groundset       = list( range(image_similarity_matrix.shape[0]) )
        self.num_images      = image_similarity_matrix.shape[0]
        self.A               = image_similarity_matrix
        self.name            = "ImgSum"
        self.nThreads        = nThreads

       


    def value(self, S):
        ''' Coverage term + genre diversity term '''
        #return (1.0-self.alpha)*np.sum(self.A[list(set(S)), :]) + self.alpha*len(np.unique( self.movie_genres[list(set(S))] ))
        if(len(S)>0):
            S = np.sort(list(set(S)))
            max_sim_eachI = np.amax(self.A[list(set(S)), :], 0)
            return np.sum(max_sim_eachI)
        else:
            return 0
        # return np.sum(self.A[list(set(S)), :])
    



    def marginalval(self, S, T):
        """ Speed-Optimized Marginal value of adding set S to current set T for value function above (EQ. 2) """
        return self.value(list(set().union(S, T))) - self.value(T)


    def load_image_similarity_matrix(image_mat_fname):
        '''
        INPUTS:
        image_mat_fname: a string filename of the image similarity edgelist
        OUTPUTS: 
        Return a 2d numpy array 
        '''
        A = pd.read_csv(image_mat_fname, header=None).values
        return A
        
    ################################################################################
    ###   Some helper functions to load the image similarity matrix ###
    ################################################################################
    def initialize_matrix(nt, nk):
        final_matrix = []
        tmp_matrix = [0.0] * nk
        for i in range (0, nt):
            final_matrix.append(tmp_matrix)
        return np.array(final_matrix)

    

    def load_image_similarity_matrix2(edgelist_fname):
        '''
        INPUTS:
        image_mat_fname: a string filename of the image similarity edgelist
        OUTPUTS: 
        Return a 2d numpy array 
        '''
        
        lenU = 50000 #len(u_dist)
        print("MATRIX initializing...")
        A = initialize_matrix(lenU, lenU)
        print("MATRIX initialized")
        
        
        print ("Reading from file", edgelist_fname);
        with open(edgelist_fname) as infile:
            for line in infile:
                edge = line.split(',')
                u = int(edge[0])
                v = int(edge[1])
                w = float(edge[2])
                A[u][v] = w
                A[v][u] = w
        print ("Reading from file COMPLETE", edgelist_fname);
        print("A MATRIX generated")       
        
        return A

    

"""
class ImageSummarizationMonotone:  

    
    def __init__(self, image_similarity_matrix):
        
        #class ImageSummarization:
        #    INPUTS and INSTANCE VARIABLES:
         #       image_similarity_matrix (becomes instance variable A for short): a 2D symmetric np.array of float64s

        self.groundset       = list( range(image_similarity_matrix.shape[0]) )
        self.num_images      = image_similarity_matrix.shape[0]
        self.A             = image_similarity_matrix


       


    def value(self, S):
        ''' Coverage term + genre diversity term '''
        #return (1.0-self.alpha)*np.sum(self.A[list(set(S)), :]) + self.alpha*len(np.unique( self.movie_genres[list(set(S))] ))
        max_sim_eachI = (self.A[:,S]).max(axis=1).toarray()
        
        return np.sum(max_sim_eachI)
        # return np.sum(self.A[list(set(S)), :])
    



    def marginalval(self, S, T):
         
        #Speed-Optimized Marginal value of adding set S to current set T for value function above (EQ. 2) 
        # Fast approach
        # Coverage term: only change is that we add each of S's rows
        # Diversity term: only change is the if we increase the number of unique genres
        return self.value(list(set().union(S, T))) - self.value(S)


    def printS(self, S):
        #Print the id of the images in the set (of indices) S
        for idx in S:
            print(idx)
"""
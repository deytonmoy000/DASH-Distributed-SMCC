nNodes=8   
nThreads=4

################################################################################################

# ExpSet Small - CentralizedData - Fig 1,4 and Fig 6 (upper row)
bash ./bash_scripts/run_ExpSet1.bash ExpSet_Centralized.py ${nNodes} ${nThreads}


################################################################################################

# ExpSet Large - DistributedData - Fig 2,5 and Fig 6 (bottom row)
bash ./bash_scripts/run_ExpSet2.bash ExpSet_Distributed.py ${nNodes} ${nThreads}


################################################################################################

# ExpSet Large Scalability - DistributedData - Figure 3
bash ./bash_scripts/run_ExpSet3.bash ExpSet3_Distributed_NPZ.py DASH ${nThreads} 1 ${nNodes}


################################################################################################
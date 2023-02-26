nNodes=8   
nThreads=4

##################################################################################################

# ExpSet Small - CentralizedData - Fig 6 (upper row)
bash ./bash_scripts/run_ExpSet_LSPGB.bash ExpSet_Centralized.py ${nNodes} ${nThreads}


###################################################################################################

# ExpSet Large - DistributedData - Fig 6 (bottom row)
bash ./bash_scripts/run_ExpSet_LSPGB2.bash ExpSet_Distributed.py ${nNodes} ${nThreads}


####################################################################################################

# ExpSet MED - CentralizedData - Fig 7 (Appendix)
bash ./bash_scripts/run_ExpSet_MetaDASH.bash ExpSet_ExtraMetaDASH_Results.py ${nNodes} ${nThreads}


####################################################################################################

# ExpSet Small - CentralizedData - Fig 8
bash ./bash_scripts/run_ExpSet_RGLAG.bash ExpSet_Centralized.py ${nNodes} ${nThreads}


####################################################################################################

# ExpSet Large - DistributedData - Fig 9
bash ./bash_scripts/run_ExpSet_RGLAG2.bash ExpSet_Distributed.py ${nNodes} ${nThreads}


####################################################################################################

# ExpSet Large - DistributedData - Fig 10 (bottom row)
bash ./bash_scripts/run_ExpSet_RG_8v32.bash ExpSet_Centralized.py ${nNodes} ${nThreads}


####################################################################################################








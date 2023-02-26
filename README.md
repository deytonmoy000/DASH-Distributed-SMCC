### DASH-Distributed_SMCC-python ###
*DASH-Distributed_SMCC-python* is a python library of high-performance MPI-parallelized implementation of state-of-the-art  distributed algorithms for submodular maximization described in the paper ***"DASH: Distributed Adaptive Sequencing Heuristic for Submodular Maximization"***. 



### Prerequisites for replicating our Experiments: ###
Please ensure the following steps are completed before running the experiments:

- Install **MPICH** version **3.3a2** (DO NOT Install OpenMPI; and ensure *mpirun* utlizes *MPICH* using the command *mpirun --version* (Ubuntu))   

- Install **pandas**

- Install **mpi4py** 

- Install **scipy**
      
- Install **networkx**


### Replicating our Experiments: ###

Our experiments can be replicated by running the following scripts:
-  Set up an MPI cluster using the tutorial in *https://mpitutorial.com/tutorials/running-an-mpi-cluster-within-a-lan/*
- Clone the *"DASH-Distributed_SMCC-python"* repository inside the MPI shared repository  (*"/cloud"* in this case using the given tutorial)
-  Create and update the host file */cloud/nodesFileIPnew* to store the ip addresses of all the connected MPI machines before running any experiments (First machine being the primary machine)
     - NOTE: Please place *nodesFileIPnew* inside the MPI shared repository, *"cloud/"* in this case (at the same level as the code base directory).  DO NOT place it inside the code base *"DASH-Distributed_SMCC-python/"* directory.

- **Additional Datasets For Experiment 1**: Please download the Image Similarity Matrix file **"images_10K_mat.csv"** ([download](https://drive.google.com/file/d/1s9PzUhV-C5dW8iL4tZPVjSRX4PBhrsiJ/view?usp=sharing)) and place it in the *"data/data_exp1/"*  directory. 

To generate the distributed data for **Experiment 1** and **2** : Please follow the below steps:
- **NOTE**: Please clone the *"DASH-Distributed_SMCC-python"* repository and execute the following commands on a **machine with sufficient memory (RAM)** ; capable of generating the large datasets. This repository **NEED NOT** be the primary repository (*"/cloud/DASH-Distributed_SMCC-python/"*) on the shared memory of the cluster; that will be used for the experiments.
   - Run *bash GenerateDistributedData.bash **nThreads** **nNodes*** 
   - The previous command should generate **nNodes** directories in *"loading_data/"* directory (with names *machine\<nodeNo\>Data*)
   - Copy the *"data_exp2_split/"* and *"data_exp3_split/"* directories within each *"machine\<i\>Data"* directory to the corresponding machine **M<sub>i</sub>** and place the directories outside *"/cloud"* (Shared directory created after setting up an MPI cluster using the ([tutorial](https://mpitutorial.com/tutorials/running-an-mpi-cluster-within-a-lan/))).


 **To run all main section experiments in sequential manner**
   -  Run *bash run.bash* from the base directory
      -  Replace **nThreads** in the script *./bash_scripts/runExpSet.sh* by the number of threads you would like the experiments to use.
      -  Replace **nNodes** in the script *./bash_scripts/runExpSet.sh* by the number of machines (distributed) you would like the experiments to use.
   -  Run 
      - *bash  ./plots_scripts/plot_Fig1.bash* (Generates **Figure 1** (and 4) and store in ***plots/Fig1/***)), 
      - *bash  ./plots_scripts/plot_Fig1.bash* (Generates **Figure 2** (and 5) and store in ***plots/Fig2/***)), 
      - *bash  ./plots_scripts/plot_Fig4.sh* (Generates **Figure 3** plots and store in ***plots/Fig3/***))

 **To generate results only for the centralized dataset experiments:** 
   -  Run *bash  ./bash_scripts/run_ExpSet1.bash ExpSet_Centralized.py **nNodes** **nThreads** "*
      -  Replace **nNodes** and **nThreads** by the number of number of machines and threads you would like the experiments to use.  
   -  Run *bash  ./plots_scripts/plot_Fig1.sh* (Generate **Figure 1, 4** plots and store in ***"plots/Fig1/"***)
   -  Run *bash  ./plots_scripts/plot_Fig6Small.sh"* (Generate **Figure 6** (upper row) plots and store in ***"plots/Fig3Small/"***)
 
 **To generate data for the distributed dataset experiments:**
   -  Run *bash  ./bash_scripts/run_ExpSet2.bash ExpSet_Distributed.py **nNodes** **nThreads** "*
      -  Replace **nNodes** and **nThreads** by the number of number of machines and threads you would like the experiments to use.  
   -  Run *bash  ./plots_scripts/plot_Fig2.sh* (Generate **Figure 2, 5** plots and store in ***"plots/Fig2/"***)
   -  Run *bash  ./plots_scripts/plot_Fig6Large.sh* (Generate **Figure 6** (bottom row) plots and store in ***"plots/Fig3Large/"***)

 **To generate data for the distributed dataset scalability experiments:** (**Figure 3**)
   -  Run *bash  ./bash_scripts/run_ExpSet3.bash ExpSet3_Distributed_NPZ.py DASH **nThreads** "*
      -  Replace **nThreads** by the number of threads you would like the experiments to use (**nNodes** is preset to [**1,2,4,8**])
   -  Run *bash  ./plots_scripts/plot_Fig3.sh* (Generate **Figure 3** plots and store in ***"plots/Fig3/"***)

 **To obtain the results illustrated in *Table 2***
   - Generate the result for the centralized dataset experiments 
   - Run *bash  ./bash_scripts/run_perf_Fig1_DASH.bash*
   - Generate the result for the distributed dataset experiments
   - Run *bash  ./bash_scripts/run_perf_Fig2.bash*

**To run all appendix experiments in sequential manner**
   -  Run *bash run_Appendix.bash* from the base directory
      -  Replace **nThreads** in the script *./bash_scripts/runExpSet_Appendix.sh* by the number of threads you would like the experiments to use.
      -  Replace **nNodes** in the script *./bash_scripts/runExpSet_Appendix.sh* by the number of machines (distributed) you would like the experiments to use.
   -  Run 
      - *bash  ./plots_scripts/plot_Fig6Small.sh* (Generates **Fig. 6**(a-e, Centralized) and store in ***plots/Fig6Small/***), 
      - *bash  ./plots_scripts/plot_Fig6Large.sh* (Generates **Fig. 6**(f-j, Distributed) and store in ***plots/Fig6Large/***) , 
      - *bash  ./plots_scripts/plot_FigApp_MEDDASH.sh* (Generates **Fig. 7** and store in ***plots/FigApp/MEDDASH/***) ,
      - *bash  ./plots_scripts/plot_FigApp_RGLAG.sh* (Generates **Fig. 8-9** and store in ***plots/FigApp/RGLAB/***), 
      - *bash  ./plots_scripts/plot_FigApp_RG_8v32.sh* (Generates **Fig. 10** and store in ***plots/FigApp/RG_8v32/***)
 <!-- **To obtain the results illustrated in *Table 3***
   - Generate the result for the centralized dataset experiments 
   - Run *bash  ./bash_scripts/run_perf_Fig3.bash* -->


 Results data will be automatically saved as CSV files in the ***"experiment_results_output_data"*** directory and the plots will be automatically saved as PDF files in the ***"/plots"*** directory.


### Implemented Distributed Algorithms: ###

<!-- ### Submodular Maximization Algorithms from "DASH: Distributed Adaptive Sequencing Heuristic for Submodular Maximization": ### -->

- **DASH** (Algorithm 1).

- **G-DASH** (Algorithm 2).

- **T-DASH** (Algorithm 3).

- **MED** (Algorithm 4).

- **RandGreedI** (Barbosa et. al. 2015).

- **ParallelAlgo** (Barbosa et. al. 2016).

- **BiCriteriaGreedy** (Epasto et. al. 2017).

- **Distributed Distorted** (Kazemi et. al. 2021).

- **ParallelGreedyBoost()** (Chen et. al. 2021).

 All the centralized data procedures are implemented in ***src/submodular_DASH_Centralized.py*** and all the distributed data procedures are implemented in ***src/submodular_DASH_Distributed.py***


### Objective functions: ###
We include the following monotone submodular objective functions:
- **Image Summarization;** 
- **Influence Maximization;**
- **Network Max Cover;**
- **Revenue Maximization**





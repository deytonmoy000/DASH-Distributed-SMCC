#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
##SBATCH --export=ALL               # use ALL for non-Intel MPI (foss) and NONE for Intel MPI
#SBATCH --job-name=Journal1RDASH         #Set the job name to "TestLDASH"
#SBATCH --time=0-07:16:00           #Set the wall clock limit to 0Day, 7hr, 10min and 0secs
#SBATCH --nodes=64 	     	        #Request 64 nodes
#SBATCH --ntasks-per-node=1         #Request 1 tasks/cores per node
#SBATCH --cpus-per-task=32	        #Request 32 cores/threads per task
#SBATCH --mem=32G                   #Request 32GB (32GB) per node 
#SBATCH --output=Journal1RDASH.%j        #Send stdout/err to "TestDASH.[jobID]"

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --account=132704168404      #Set billing account to 132704168404 (primary) / 132704169032 (test)
##SBATCH --mail-type=ALL            #Send email on all job events
##SBATCH --mail-user=tdey@fsu.edu   #Send all emails to email_address 

#First Executable Line
module purge
module load Anaconda3/2022.05
module load icc/2018.3.222-GCC-7.3.0-2.30 impi/2018.3.222
module load mpi4py/3.0.1-Python-3.6.6
module load GCC/10.3.0
module load networkx/2.2-Python-3.6.6

module list

mpirun --version

alg=$1
# obj=$2
declare -a seeds=("35" "8" "14" "25") # ("42" "8" "14" "25" "35") Please use any seed value as you prefer
declare -a objs=("IFM" "RVM" "MCV")

for j in ${!seeds[@]};
do
    for k in ${!objs[@]};
    do
        seed=${seeds[$j]}
        obj=${objs[$k]}
        cmd="mpirun --bind-to none python3 -W ignore ExpSet_Journal1.py ${obj} ${alg} 32 ${seed}"
        echo ${cmd}
        $cmd
        echo "${alg} - Data generated for '${obj}' with seed = '${seed}'"
    done
done
